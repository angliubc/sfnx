# MAINLINE — read this first (the tree is written for agents, not for humans)

## state (2026-11-19)

- fork of `johnfink8/sfnx`, base = upstream `main` @ `9808733` (2026-11-07).
- upstream is frozen: 0 commits on any branch since 11-07, the maintainer's whole GitHub account has been silent
  since 11-07 23:14, 4 community PRs sit unreviewed. This branch is the mainline from 11-19.
- production runs `/data/sfnx-prod` on one 8-card V100 box (atlas1): D `:8093` (cards 0-3) + P `:8094` (cards 4-7),
  hand-off over the local IPoIB loopback. **Read `DIVERGENCE` below before rolling anything to it.**
- read order: this file → `AGENTS.md` (upstream's build/exactness discipline) → `docs/design.md`.

## delta map — feature → files → why → evidence

### 1. `vision` — the picture tower inside the engine (`5f68acd`)
- files: `src/vit.cu` `src/vit.hpp` `src/vit_def.hpp` `src/vit_host.cpp` `src/vit_host.hpp` `src/stb_image.h`;
  selected by `--vision` / `serve.*.vision.engine_tower`. Upstream's `vision/sfnx_vision.cpp` (llama.cpp's mtmd
  subprocess) stays; the in-engine tower is the one production uses.
- two numerics bugs, both found by oracle comparison, both silent (no crash, no NaN):
  - `k_vit_rope`: write-write race — `r` and `r+18` wrote the same channel pair. Fix: `r` indexes the pair, `0..35`.
  - `k_vit_pos_add`: the position table was read transposed. Correct stride `c + t*1152`, not `i*2304`.
- evidence: final embedding vs oracle `cos = 0.999465`; the residual `3.3e-2` is the **oracle's own** flash-attn F16
  noise (oracle with flash off: `3.4e-4`) — the engine's fp32 path is the more exact of the two. An `<=1e-4` target
  against that oracle is unreachable and is not a bug.
- pitfall (cost a full day): **BF16 gguf tensors are 2 bytes/element.** A reader using `n*4` poisons every numpy
  reference value at once and every downstream number becomes garbage that still looks plausible.

### 2. `pd` — prefill/decode hand-off (`9142134`, `94e44cd`)
- engine: `PDOFFER` / `PDUMP` / `PLOAD` (+ park) in `tools/sfnx.cpp`; the front drives them over
  `/pd/offer` `/pd/blob` `/pd/pread` `/pd/pread_stream` `/pd/dump` `/pd/pread_dump` `/pd/busy`.
- `PDSTREAM` (`/pd/pread_stream`): P pushes the K/V straight to D over TCP, zero landing on disk.
- `rope_std_` (`src/engine.cu|hpp`): a hand-off's K/V is computed with the **plain** rotary whatever scaling this
  engine runs, so a dump crosses a YaRN engine and an unscaled one. The bit joins `graph_key` — the two rotaries'
  window graphs are captured apart, a mismatch would replay the other's baked-in frequencies. `rope_scaled()`
  refuses a plain-rotary dump loading onto a YaRN engine.
- `reload_dump()` (`src/session.cpp|hpp`): a PLOAD'd dump carries no checkpoints, so without one the conversation
  parks with nothing to restore to and its next turn re-reads everything. **Measured 12.3s → 0.7s** on a
  12k-token turn.
- `park_and_unload`: parked conversations survive eviction pressure; 88% reuse, 8.7× faster turns. The only tax is
  P holding nothing (by design).
- `pd_peers` is a **pool** (a conversation keys on its first user message, full text); `pd_peer_prefill` /
  `pd_peer_timeout` / `pd_peer_busy` govern the ask; `park_lock` keeps a parked state from being evicted mid-hand-off.

### 3. `hc` — the inject weight rides f32, not bf16 (`9142134`)
- loader `f32_inj` (`src/model.cpp`, F32/BF16/Q8_0) + consumers (`src/hc.cuh`, `src/kernels.cu|hpp`, `src/prefill.cu`).
- why: the last 8-mantissa-bit weights. On 4 cards the bf16 inject compounded with the prompt ring's per-hop f16
  rounding and blew the `code1` exactness bar: **KL mean 6.06e-2 → 5.6e-4**, `edit1` rows byte-identical to the
  reference. Cost +2 MB/card.
- open caveat (upstream issue #11, ours, unanswered): a freshly started holder's **first** `exact` run shows ~6e-2
  on *both* trees — run-order sensitivity, so part of the original 6.06e-2 report was that, not the inject alone.

### 4. `front` — slot accounting, /prom, premature-stop guard (`94e44cd`, `39e982f` era hunks)
- `(#1869)`: a request ending at EOS inside a batch slot takes the **slot leg's** figures, not the admission's
  single token (the background wait used to eat the leg).
- `front/prom.py`: `GET /prom` renders the `/metrics` snapshot as Prometheus text; the JSON `/metrics` is untouched.
- `front/test_premature_stop_1113.py` pins the 1113 premature-stop class. Sampling in production pins
  `presence_penalty: 1.0` — the 1110 fix for the class of 271-token early EOS.

### 5. `build` / `lab` (`067773d`)
- default toolchain is the fleet's: `-arch=sm_70` + `g++-12` (CUDA 12.9 boxes); ViT objects join the engine link.
- `FNX_LAB_GPUS` picks the card set for `tools/lab.sh` on multi-rig beds.

## build / run (exact, as production does it)

```
make -j8 FNX_NCARD=4                      # engine binaries into bin/
llama.cpp: clone at a37cc46 (only the picture path and ref/ tools need it)
/data/sfnx-prod/.venv/bin/python -m front.server --engine sfnx --config serve/sfnx-8093.json
units in ops/units/, the sticky router in ops/sfnx-gw.py, the relay in ops/relay_sfnx.py.
```

## verification status (2026-11-19, this branch)

- `python3 -m pytest front/ -q`: **264 passed, 1 failed, 3 skipped, 6206 subtests passed**.
- The one failure (`front/test_responses.py::OverHttp::test_json_schema_text_format`, `TypeError: NoneType not
  subscriptable`) is **pre-existing on upstream `main` @ `9808733`** — reproduced in a clean worktree. Not ours.
- conversation cache, measured through the router: turn1 `create=7262 read=0` 3.95s → turn2 `read=7258` **0.67s**
  → turn3 `read=7278` 0.60s (99.7% reuse on a 7.2k-token conversation).

## pitfalls — each one cost a session

1. **The front's `/pd/*` routes always exist; the engine's `PDSTREAM` support is compile-time.** Check
   `strings bin/sfnx | grep PDSTREAM` before cloning a deployment. A P built without it dead-waits 300s per hand-off
   and the outage looks like a network fault (1116, total outage).
2. **Changing the shared gateway's config requires restarting it**, not just the small router in front of the
   instances — dead entries keep being advertised from memory for days (1116: 3 days).
3. The gateway's model advertisement has a **300s refresh lag** after a config change; a stale `/v1/models` right
   after a delete does not mean the delete failed.
4. `simple_backends` in the gateway config is a **list**, not a dict.
5. The P role line (`PD: prefill side`) is in the **systemd journal**, not the log file.
6. Under light load a single P shows **zero hand-off traffic by design** (an idle D reads locally faster). Judge P
   liveness by endpoint 200 + the journal role line, never by traffic.
7. The sticky key is the **first user message's full text** — a client that rewrites its first message changes engine.
8. Cloning a blueprint config: `FNX_EXPERT_PROFILE` carries an **absolute** path into the source tree.
9. The monitor puller keeps a `RETIRED` list: dropping a retired instance outright makes `rate()` show a false reset
   spike (its lifetime counters must keep being summed).
10. `/tmp` on the workstations is wiped by power events; durable artifacts go to `~/claude/…`, on atlas11 to `/data`.

## DIVERGENCE — production vs this mainline (do not skip)

- **`front/server.py`**: production runs the **pre-#16** slot variant (`_bdone_after_eos` without the queue param, no
  `SlotLost`/`SLOT_GONE`). This branch merged **upstream's #16** (batch-slot claim handover) with our hunks; our
  `(#1869)` accounting fix survives at the call site and the front suite is green. **Roll to production only after an
  on-engine A/B** — the two variants differ in who frees a slot a request no longer owns.
- production's tree is not a git repo (rsync'd copies); this branch is the truth source from 11-19.
- the ViT comparison bed (`atlas11:/data/sfnx-vit`, `COMMAND-LOG.md`, the oracle dumps) is not in this tree — that
  host refused ssh on 11-19. The two fixes themselves are in `src/vit.cu`.

## upstream — what is parked there

- issue #4 (ours, open): 8-card numbers + the `cudaDeviceEnablePeerAccess` tolerance patch for partial-mesh boxes.
  Main still hard-fails: `src/engine.cu:143` `ck(e, "peer access")`.
- issue #11 (ours, open): the first-execution effect correction. Zero maintainer response since 11-01.
- worth sending back when they wake: the f32 inject fix (their issue #11 territory), `reload_dump()`, the ViT tower.
