"""front/prom.py - GET /prom: the /metrics snapshot as Prometheus text (1104).

Read-only addition for the 24.25 fleet monitor: renders exactly the metric
names the ang-side puller (vllm-monitor/sfnx_puller.py) pushes to the
pushgateway, straight from the service's own snapshot (what GET /metrics with
?requests=all serves), so Prometheus can later scrape this port directly
without renaming anything.  The JSON /metrics output is untouched.

Counters come from the snapshot's "totals" block (the engine's own truth since
the process started).  TTFT has no total: sfnx:ttft_seconds_total and
sfnx:ttft_count_total accumulate in-process from the finished-request history,
deduplicated by request id; a server restart re-zeroes them and rate() absorbs
the reset.

Nulls read as 0 (a series must always exist between scrapes).
"""
from __future__ import annotations

import threading

# gauges from the snapshot's "live" block: metric name <- snapshot key
LIVE_GAUGES = (("sfnx:running", "running"), ("sfnx:waiting", "waiting"),
               ("sfnx:queued", "queued"), ("sfnx:parallel", "parallel"),
               ("sfnx:tok_s", "tok_s"), ("sfnx:tok_s_mean", "tok_s_mean"),
               ("sfnx:prefill_tok_s", "prefill_tok_s_mean"),
               ("sfnx:outside_slots", "outside_slots"))

# counters from the snapshot's "totals" block: metric name <- key <- divisor
TOTALS = (("sfnx:requests_total", "requests", 1.0),
          ("sfnx:output_tokens_total", "output_tokens", 1.0),
          ("sfnx:prompt_tokens_total", "prompt_tokens", 1.0),
          ("sfnx:reused_tokens_total", "reused", 1.0),
          ("sfnx:drafts_offered_total", "drafts_offered", 1.0),
          ("sfnx:drafts_accepted_total", "drafts_accepted", 1.0),
          ("sfnx:prompt_seconds_total", "prompt_ms", 1000.0),
          ("sfnx:decode_seconds_total", "decode_ms", 1000.0))

_lock = threading.Lock()          # scrapes can arrive concurrently
_ttft = {"ids": set(), "s": 0.0, "n": 0}


def _num(v):
    """A float for anything numeric; None/absent/NaN -> 0.0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f


def _fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else f"{v:.6f}"


def render(svc) -> str:
    """The exposition document: gauges and counters, same names as the puller's."""
    snap = svc.metrics(all_requests=True)     # the same snapshot /metrics serves
    lines = []

    def emit(name, mtype, samples):
        lines.append(f"# TYPE {name} {mtype}")
        for labels, value in samples:
            lab = ("{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}") if labels else ""
            lines.append(f"{name}{lab} {_fmt(value)}")

    live = snap.get("live") or {}
    for name, key in LIVE_GAUGES:
        emit(name, "gauge", [({}, _num(live.get(key)))])
    held, held_sum = [], 0.0
    for s in live.get("slots") or []:
        held.append(({"slot": s.get("slot")}, _num(s.get("held_tokens"))))
        held_sum += _num(s.get("held_tokens"))
    emit("sfnx:held_tokens", "gauge", held)
    emit("sfnx:held_tokens_sum", "gauge", [({}, held_sum)])
    gpus, vram = [], []
    for g in (snap.get("hardware") or {}).get("gpus") or []:
        gpus.append(({"gpu": g.get("index")}, _num(g.get("util"))))
        vram.append(({"gpu": g.get("index")}, _num(g.get("mem_used")) / 1048576.0))
    emit("sfnx:gpu_util", "gauge", gpus)
    emit("sfnx:vram_used_mib", "gauge", vram)

    cache = snap.get("conversation_cache") or {}
    emit("sfnx:cache_bytes", "gauge", [({}, _num(cache.get("bytes")))])
    emit("sfnx:cache_budget_bytes", "gauge", [({}, _num(cache.get("budget_mib")) * 1048576.0)])

    totals = snap.get("totals") or {}
    for name, key, div in TOTALS:
        emit(name, "counter", [({}, _num(totals.get(key)) / div)])

    with _lock:                   # TTFT: no engine total - dedup the history by id
        ids, fresh = set(), []
        for r in snap.get("requests") or []:
            rid = r.get("id")
            if rid is None:
                continue
            ids.add(str(rid))
            if r.get("first_engine_token_s") is not None and str(rid) not in _ttft["ids"]:
                fresh.append(r)
        for r in fresh:
            _ttft["s"] += _num(r.get("first_engine_token_s"))
            _ttft["n"] += 1
        _ttft["ids"] = ids
        emit("sfnx:ttft_seconds_total", "counter", [({}, _ttft["s"])])
        emit("sfnx:ttft_count_total", "counter", [({}, _ttft["n"])])

    return "\n".join(lines) + "\n"
