#!/usr/bin/env python3
"""sFNX 1P1D (atlas1 D:8093 + P:8094) JSON snapshots -> TWO pushgateway entries
(1119): instance=sfnx (the D row, one merged statistic) and instance=sfnx-p
(the P engine's own row).

History: 1110 aggregated a 4-D pool into one 'sfnx' entry; 1113 shrank the
pool to a single-machine 1P1D; 1119 the user asked for the P engine to be on
the board too - as its own row, NOT folded into sfnx, because P's counters
would double-count hand-off tokens and pollute TTFT / hit-rate (a hand-off's
prefill shows up on the D side as 'reused', which is the fleet-visible truth).

Polls GET /metrics?requests=all every 15s, merges each group into one
Prometheus metric set and pushes it through the local tunnel
(127.0.0.1:18091 -> 24.25:9091) as job=sfnx, instances sfnx / sfnx-p.  The
fleet dashboard panels all aggregate with sum/avg by (instance), so both rows
work with zero query changes - just $instance values added on the board.

Merge rules:
  running/waiting/queued/parallel/outside_slots/tok_s : SUM
  tok_s_mean / prefill_tok_s_mean                      : mean (off-panel, ad-hoc)
  held_tokens_sum                                      : SUM (per-slot rows dropped)
  gpu_util{gpu}/vram_used_mib{gpu}                     : one row per card,
        gpu label = "<tag>-g<idx>" (a1-93-g0 .. a8-94-g3, 16 cards total)
  cache_bytes / cache_budget_bytes                     : SUM
  totals counters (requests/prompt/reused/output/drafts/ms): per-engine
        MONOTONE CONTINUATION (1113) - the raw SUM had a lie in it: any engine
        restart (1113 11:19-11:24 Q8Top16 rollout put a fake 560 tok/s spike on
        the board) or one failed fetch drops the merged counter, and Prometheus
        rate() reads a dip as a reset and adds the whole post-dip value.  The
        puller now keeps a cumulative per engine: delta when the engine's
        lifetime counter rises, +cur when it decreases (new lifetime), and the
        last-known cum when a fetch fails.  Survives puller restarts via
        .sfnx_puller_state.json.
  ttft_seconds/count_total                             : SUM of per-instance
        accumulators (dedup by request id within one engine)
  sfnx:pool_up                                         : how many instances
        made it into this push (0..4) - partial-failure visibility

Zombie self-heal (1023 lesson): per GROUP, only when ALL of its instances fail
3 consecutive rounds do we DELETE that group's pushgateway entry - a single
down instance must not blank the model row; pool_up dropping is the signal.

--once / --dry-run are for manual checks.  All HTTP is proxy-less.

--once / --dry-run are for manual checks.  All HTTP is proxy-less.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

# (push tag, gpu tag, url) - the D pool only; P stays out of the merge
# 1113 1P1D shrink: one D (atlas1:8093, cards 0-3) + one P (same box :8094).
INSTANCES = [
    ("a1-8093", "a1-93", "http://<FLEET_HOST>:8093/metrics?requests=all"),
]
# 1119 user call ("put in the 2 qwen3.8 flash next instances"): the P engine
# (:8094, cards 4-7) gets its OWN board row, instance=sfnx-p - NOT folded into
# the merged sfnx (D) row, because P's counters would double-count hand-off
# tokens (1110 lesson; D sees a hand-off's prefill as 'reused' = fleet truth).
# Tag "a1-94p" deliberately != RETIRED "a1-8094": the frozen D-era counters of
# that name keep feeding the sfnx row's history and must not be resurrected.
P_INSTANCE = ("a1-94p", "a1-94", "http://<FLEET_HOST>:8094/metrics?requests=all")
P_AGG_INSTANCE = "sfnx-p"
# Engines retired by the shrink.  Their lifetime counters stay frozen in
# .sfnx_puller_state.json and keep counting toward the merged totals - dropping
# them outright would dip the sum and rate() would paint a fake reset spike
# (same family as the 1113 restart-spike fix).  Never fetched, never re-saved
# as zero: save_state/total() below cover ALL of `state`, not just INSTANCES.
RETIRED = ["a1-8094", "a8-8093", "a8-8094"]
PUSH_BASE = "http://127.0.0.1:18091"      # vllm-mon-tunnel -> 24.25:9091 (pushgateway)
JOB = "sfnx"
AGG_INSTANCE = "sfnx"                     # the ONE statistic the fleet board shows
INTERVAL = 15.0
FETCH_TIMEOUT = 10.0
FAIL_DELETE_AFTER = 3
# the engine's process-level lifetime counters that must reach Prometheus monotone
COUNTER_KEYS = ("requests", "output_tokens", "prompt_tokens", "reused",
                "drafts_offered", "drafts_accepted", "prompt_ms", "decode_ms")
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sfnx_puller_state.json")

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never a proxy


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def num(v):
    """A float for anything numeric; None/absent/NaN -> 0.0 (a series must always exist)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


def fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.6f}"


class Family:
    """One metric family: name, type, samples."""

    def __init__(self, name, mtype):
        self.name, self.mtype, self.rows = name, mtype, []

    def add(self, value, **labels):
        lab = "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}" if labels else ""
        self.rows.append((lab, fmt(value)))


def merge_lines(snaps, scope=None):
    """snaps = [(tag, gputag, snapshot)] -> the aggregated metric lines.
    scope = None (default): totals/TTFT sum over ALL of state - the sfnx (D)
    row keeps the retired engines' frozen history in its counters.  A tag set
    (the P row) sums only over those tags, so P's counters and TTFT never leak
    into the D row and vice versa."""
    fams = []

    def fam(name, mtype):
        f = Family(name, mtype)
        fams.append(f)
        return f

    def live_sum(key):
        return sum(num((s.get("live") or {}).get(key)) for _, _, s in snaps)

    for name, key in (("sfnx:running", "running"), ("sfnx:waiting", "waiting"),
                      ("sfnx:queued", "queued"), ("sfnx:parallel", "parallel"),
                      ("sfnx:tok_s", "tok_s"), ("sfnx:outside_slots", "outside_slots")):
        fam(name, "gauge").add(live_sum(key))
    for name, key in (("sfnx:tok_s_mean", "tok_s_mean"),
                      ("sfnx:prefill_tok_s", "prefill_tok_s_mean")):
        fam(name, "gauge").add(live_sum(key) / len(snaps))
    fam("sfnx:held_tokens_sum", "gauge").add(
        sum(sum(num(s_.get("held_tokens")) for s_ in (snap.get("live") or {}).get("slots") or [])
            for _, _, snap in snaps))

    gpu_util = fam("sfnx:gpu_util", "gauge")
    vram = fam("sfnx:vram_used_mib", "gauge")
    for _, gputag, snap in snaps:
        for g in (snap.get("hardware") or {}).get("gpus") or []:
            gpu_util.add(num(g.get("util")), gpu=f"{gputag}-g{g.get('index')}")
            vram.add(num(g.get("mem_used")) / 1048576.0, gpu=f"{gputag}-g{g.get('index')}")

    cache_bytes = sum(num((s.get("conversation_cache") or {}).get("bytes")) for _, _, s in snaps)
    cache_budget = sum(num((s.get("conversation_cache") or {}).get("budget_mib")) * 1048576.0
                       for _, _, s in snaps)
    fam("sfnx:cache_bytes", "gauge").add(cache_bytes)
    fam("sfnx:cache_budget_bytes", "gauge").add(cache_budget)

    fold_counters(snaps)

    def total(key):
        # monotone continuation, not the raw sum - see fold_counters
        # over the scope's slice of state: live engines delta up, retired ones stay frozen
        tags = state.keys() if scope is None else (t for t in scope if t in state)
        return sum(state[t]["cum"].get(key, 0.0) for t in tags)

    for name, key, div in (("sfnx:requests_total", "requests", 1.0),
                           ("sfnx:output_tokens_total", "output_tokens", 1.0),
                           ("sfnx:prompt_tokens_total", "prompt_tokens", 1.0),
                           ("sfnx:reused_tokens_total", "reused", 1.0),
                           ("sfnx:drafts_offered_total", "drafts_offered", 1.0),
                           ("sfnx:drafts_accepted_total", "drafts_accepted", 1.0),
                           ("sfnx:prompt_seconds_total", "prompt_ms", 1000.0),
                           ("sfnx:decode_seconds_total", "decode_ms", 1000.0)):
        fam(name, "counter").add(total(key) / div)

    # TTFT: no engine total - accumulate from the finished-request window,
    # dedup by request id within each engine, then sum the accumulators.
    if scope is None:
        ttft_tags = [t for t, st in state.items() if st.get("in_push")]
    else:
        ttft_tags = [t for t in scope if t in state and state[t].get("in_push")]
    ttft_s = sum(state[t]["ttft"]["s"] for t in ttft_tags)
    ttft_n = sum(state[t]["ttft"]["n"] for t in ttft_tags)
    fam("sfnx:ttft_seconds_total", "counter").add(ttft_s)
    fam("sfnx:ttft_count_total", "counter").add(ttft_n)

    fam("sfnx:pool_up", "gauge").add(len(snaps))

    lines = []
    for f in fams:
        lines.append(f"# TYPE {f.name} {f.mtype}")
        lines += [f"{f.name}{lab} {val}" for lab, val in f.rows]
    return lines


def push_path(instance=AGG_INSTANCE):
    return f"{PUSH_BASE}/metrics/job/{JOB}/instance/{urllib.parse.quote(instance, safe='')}"


def fetch(url):
    with _opener.open(url, timeout=FETCH_TIMEOUT) as r:
        snap = json.loads(r.read().decode())
    if not isinstance(snap, dict) or "live" not in snap:
        raise ValueError("snapshot has no 'live' block")
    return snap


def push(lines, instance=AGG_INSTANCE):
    body = ("\n".join(lines) + "\n").encode()
    req = urllib.request.Request(push_path(instance), data=body, method="POST",
                                 headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"})
    with _opener.open(req, timeout=15) as r:
        r.read()


def accumulate_ttft(tag, snap):
    """Fold this snapshot's finished-request window into the per-instance TTFT state."""
    ttft = state[tag]["ttft"]
    cur_ids, fresh = set(), []
    for r in snap.get("requests") or []:
        rid = r.get("id")
        if rid is None:
            continue
        cur_ids.add(str(rid))
        if r.get("first_engine_token_s") is not None and str(rid) not in ttft["ids"]:
            fresh.append(r)
    for r in fresh:
        ttft["s"] += num(r.get("first_engine_token_s"))
        ttft["n"] += 1
    ttft["ids"] = cur_ids


state = {tag: {"fails": 0, "in_push": False, "ttft": {"ids": set(), "s": 0.0, "n": 0},
               "last": {}, "cum": {}}
         for tag, _, _ in INSTANCES + [P_INSTANCE]}
state.update({tag: {"fails": 0, "in_push": False, "ttft": {"ids": set(), "s": 0.0, "n": 0},
                    "last": {}, "cum": {}}
              for tag in RETIRED})


def fold_counters(snaps):
    """Keep each engine's lifetime counters as one monotone cumulative per key.

    cur >= last  -> the process kept running: add the delta.
    cur <  last  -> the process restarted: its new lifetime counts from 0, add cur.
    no snapshot  -> keep the last cum untouched (a flaky fetch must not dip the
                    merged sum - that dip is what rate() misreads as a reset).
    """
    for tag, _, snap in snaps:
        st = state[tag]
        tot = snap.get("totals") or {}
        for k in COUNTER_KEYS:
            cur = num(tot.get(k))
            last = st["last"].get(k)
            if last is None:
                st["cum"][k] = cur
            elif cur < last:
                st["cum"][k] += cur
            else:
                st["cum"][k] += cur - last
            st["last"][k] = cur


def save_state():
    """Persist the continuations (and the TTFT dedup) so a puller restart changes
    nothing on the board.  Atomic: tmp + rename."""
    doc = {"counters": {t: {"last": st["last"], "cum": st["cum"]} for t, st in state.items()},
           "ttft": {t: {"ids": sorted(st["ttft"]["ids"]),
                        "s": st["ttft"]["s"], "n": st["ttft"]["n"]} for t, st in state.items()}}
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, STATE_PATH)


def load_state():
    try:
        with open(STATE_PATH) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return
    for tag, c in (doc.get("counters") or {}).items():
        if tag in state:
            state[tag]["last"] = dict(c.get("last") or {})
            state[tag]["cum"] = dict(c.get("cum") or {})
    for tag, t in (doc.get("ttft") or {}).items():
        if tag in state:
            state[tag]["ttft"] = {"ids": set(t.get("ids") or ()), "s": num(t.get("s")), "n": num(t.get("n"))}


def run_group(insts, agg_instance, scope, dry):
    """One polling round for one push group (a set of engines -> one board row).
    Returns True when the round produced (or would produce) a push."""
    snaps = []
    for tag, gputag, url in insts:
        st = state[tag]
        st["in_push"] = False
        try:
            snap = fetch(url)
        except Exception as e:                                       # noqa: BLE001 - any fetch error is one fail
            st["fails"] += 1
            log(f"{tag}: fetch fail #{st['fails']}: {e!r}")
            continue
        st["fails"] = 0
        st["in_push"] = True
        accumulate_ttft(tag, snap)
        snaps.append((tag, gputag, snap))
    if not snaps:
        fails = [state[t]["fails"] for t, _, _ in insts]
        if min(fails) >= FAIL_DELETE_AFTER:
            try:
                with _opener.open(urllib.request.Request(push_path(agg_instance), method="DELETE"),
                                  timeout=15) as r:
                    r.read()
                log(f"{agg_instance}: ALL instances down ({fails} fails) - pushgateway entry deleted")
            except Exception as e:                                   # noqa: BLE001
                log(f"{agg_instance}: DELETE failed: {e!r}")
            # reset TTFT dedup ONLY for this group's engines, not the other row's
            for t, _, _ in insts:
                state[t]["ttft"] = {"ids": set(), "s": 0.0, "n": 0}
        return False
    lines = merge_lines(snaps, scope=scope)
    if dry:
        print(f"--- {agg_instance} ({len(snaps)}/{len(insts)} instances, {len(lines)} lines) ---")
        print("\n".join(lines))
        return True
    try:
        push(lines, agg_instance)
        save_state()
        down = [t for t, _, _ in insts if not state[t]["in_push"]]
        log(f"{agg_instance}: pushed {len(lines)} lines "
            f"({len(snaps)}/{len(insts)} up{', down: ' + ','.join(down) if down else ''})")
        return True
    except Exception as e:                                           # noqa: BLE001
        log(f"{agg_instance}: PUSH FAIL: {e!r}")
        save_state()   # the fold already happened; persist it so the next round deltas are right
        return False


# (instances, board instance name, counter/TTFT scope tags)
# sfnx (D row) scope = live D engines + RETIRED frozen history; the P tag is
# deliberately absent so P's counters never leak into the D row (and vice versa).
GROUPS = [
    (INSTANCES, AGG_INSTANCE, [t for t, _, _ in INSTANCES] + RETIRED),
    ([P_INSTANCE], P_AGG_INSTANCE, [P_INSTANCE[0]]),
]


def main():
    once, dry = "--once" in sys.argv, "--dry-run" in sys.argv
    if not dry:
        load_state()
    while True:
        ok = all(run_group(insts, agg, scope, dry) for insts, agg, scope in GROUPS)
        if dry or once:
            sys.exit(0 if ok else 1)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
