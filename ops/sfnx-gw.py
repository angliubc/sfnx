#!/usr/bin/env python3
"""sfnx-gw: atlas11 qwen model-side gateway (1104 architecture directive).

Thin router for the sFNX fleet on this host. What lives here (model-side):
  - conversation affinity: first-user-message -> sticky backend (least-loaded
    placement on first sight, KV/radix hit afterwards)
  - health: active probe, DOWN members bypassed
  - failover: connect error / 5xx before any bytes relayed -> retry a healthy peer
  - audit: one JSON line per request + the 1104 ledger alarm (engine done tokens
    vs what this gateway actually forwarded - a vanish like done=172/visible=34
    now leaves a trace at the hop where it happens)
What deliberately does NOT live here (stays upstream or in front.server):
  thinking policy, budget clamps, effort stripping, model-name rewriting,
  loop guards.  No request-body rewriting of any kind.

Backends: sfnx-gw.json next to this file:
  [{"name": "sfnx-8093", "url": "http://127.0.0.1:8093", "pool": true}, ...]
  pool=false lanes (ctx-heterogeneous dev instances, e.g. the 262K 8095) are
  excluded from affinity routing - reachable only via X-Sfnx-Backend header.
  (1102 lesson: a 64K copy inside a 1M pool pins sticky sessions that then
  hard-fail on /compact.)
"""
import hashlib
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "sfnx-gw.json")
LOG_DIR = os.path.join(HERE, "gw-logs")
LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
CONNECT_TIMEOUT = 8
READ_TIMEOUT = 20          # per-read (SSE); on expiry we send an SSE comment and retry
READ_TIMEOUT_NONSSE = 120  # 1110: non-streaming = zero bytes until the engine finishes
                          # the whole generation (thinking 30-90s+), so getresponse()
                          # needs room; trade-off: dead-backend failover also 120s
                          # for non-SSE (rare path, CC/scripts stream)
HEALTH_EVERY_S = 15
HEALTH_FAILS = 2

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_backends = {}            # name -> {"url", "host", "port", "pool"}
_health = {}              # name -> bool
_fail_streak = {}
_sticky = {}              # conv key -> backend name (persisted: a gw restart must not orphan
                          # 1M-ctx sessions whose KV sits warm on one D - 1111)
_last_seen = {}           # conv key -> (ts, backend): the ACTIVE signal for placement (1111 -
                          # the old lifetime sticky count let dead sessions outvote the living
                          # and two fresh heavies could still land on one machine)
_lock = threading.Lock()
STICKY_FILE = os.path.join(HERE, "sticky.json")
STICKY_SAVE_S = 60        # dump cadence; the boot load is one read
ACTIVE_WINDOW_S = 1800    # a conversation silent this long no longer counts toward placement


def parse_url(url):
    rest = url.split("//", 1)[-1].rstrip("/")
    host, _, path = rest.partition("/")
    if ":" in host:
        h, p = host.split(":")
        return h, int(p), path
    return host, 80, path


def load_config():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = json.load(f)
    with _lock:
        for b in cfg.get("backends", []):
            h, p, _ = parse_url(b["url"])
            _backends[b["name"]] = {"url": b["url"].rstrip("/"), "host": h, "port": p,
                                    "pool": bool(b.get("pool", True))}
            _health.setdefault(b["name"], True)
    # 1111: sticky sessions survive a restart.  A 1M-ctx conversation's KV sits warm on
    # one D; a gw restart that forgot it would re-place by (empty) activity and could
    # bounce the fleet's heaviest sessions onto cold engines - a full re-read each.
    # _last_seen rides along: without it the restored heavies count as silent until their
    # next request, and a fresh conversation could land on the same machine in that window.
    try:
        with open(STICKY_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        keep = {k: v for k, v in (saved.get("sticky") or {}).items() if v in _backends}
        seen = {k: (ts, v) for k, (ts, v) in (saved.get("last_seen") or {}).items() if v in _backends}
        with _lock:
            _sticky.update(keep)
            _last_seen.update(seen)
        log(f"sticky: restored {len(keep)} sessions, {len(seen)} activity marks")
    except FileNotFoundError:
        pass
    except Exception as e:
        log(f"sticky restore skipped ({e})")
    return cfg


def sticky_save_loop():
    while True:
        time.sleep(STICKY_SAVE_S)
        try:
            with _lock:
                snap = dict(_sticky)
                # keep _last_seen flat: entries past the active window are dead weight
                cutoff = time.time() - ACTIVE_WINDOW_S
                for k2 in [k for k, (ts, _) in _last_seen.items() if ts < cutoff]:
                    _last_seen.pop(k2, None)
                seen = dict(_last_seen)
            tmp = STICKY_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"sticky": snap, "last_seen": {k: list(v) for k, v in seen.items()}}, f)
            os.replace(tmp, STICKY_FILE)
        except Exception as e:
            log(f"sticky save failed ({e})")


def log(*a):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def audit(obj):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "sfnx-gw-%s.log" % datetime.now().strftime("%Y%m%d"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ── conversation key (ported from the 24.23 gateway, incl. 1030 blocks-array fix) ──

def conv_key(body):
    try:
        data = json.loads(body)
    except Exception:
        return "noconv"
    for m in data.get("messages", []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and c:
            return "conv:" + hashlib.md5(c.encode()).hexdigest()
        if isinstance(c, list) and c:
            texts = [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
            joined = "".join(texts)
            if joined:
                return "conv:" + hashlib.md5(joined.encode()).hexdigest()
            return "conv:" + hashlib.md5(json.dumps(m, sort_keys=True).encode()).hexdigest()
        if c:
            return "conv:" + hashlib.md5(str(c).encode()).hexdigest()
    return "noconv"


def route(key, pin=None):
    """(name, backend dict) or (None, None). Pin first, then sticky, then least-loaded."""
    with _lock:
        if pin and pin in _backends and _health.get(pin):
            return pin, _backends[pin]
        pool = sorted(n for n, b in _backends.items() if b["pool"] and _health.get(n))
        if not pool:
            return None, None
        name = _sticky.get(key)
        if name not in pool:
            # 1110: a NEW key joins the backend with the fewest sticky sessions,
            # not its hash slot. Hash rolls dice: 6 fresh subagent sessions over
            # 6 backends = 1.5% perfect spread (expected max bucket ~2.4) and a
            # collision pins for the whole conversation. Least-loaded makes
            # N<=len(pool) concurrent sessions land on N distinct backends.
            # Placement only - once in _sticky the session never moves, so
            # KV/radix affinity is unchanged after the first request.
            # 1111: "fewest" now counts sessions ACTIVE in the last 30 min, not
            # lifetime - dead conversations had outvoted the living and two
            # fresh heavy sessions could still land on one machine's two lanes
            # (which then fed one P: the day the pool half died).
            now = time.time()
            load = {n: 0 for n in pool}
            for k2, (ts, v) in _last_seen.items():
                if now - ts <= ACTIVE_WINDOW_S and v in load and k2 != key:
                    load[v] += 1
            name = min(pool, key=lambda n: (load[n], n))
            # 1111 evening: a session PINNED to a backend that is merely DOWN detours
            # to the least-loaded healthy member for THIS request only - the pin
            # survives, and the conversation returns when the backend does.  The
            # boot race once re-placed every pin while its D was still loading
            # (rolling restart + gateway restart in the same minute), and 7
            # concurrent sessions then ran on one D for an hour while another
            # idled.  A pin to a backend RETIRED from config (not in _backends at
            # all) still re-places permanently.
            if key in _sticky and _sticky[key] in _backends:
                log(f"sticky: {key[:18]} pinned to {_sticky[key]} (down) - detour to {name}")
                _last_seen[key] = (time.time(), name)
                return name, _backends[name]
            _sticky[key] = name
            if len(_sticky) > 100000:
                _sticky.pop(next(iter(_sticky)))
        _last_seen[key] = (time.time(), name)
        return name, _backends[name]


def health_loop():
    while True:
        with _lock:
            snapshot = {n: dict(b) for n, b in _backends.items()}
        for n, b in snapshot.items():
            ok = False
            try:
                req = urllib.request.Request(b["url"] + "/v1/models")
                with _opener.open(req, timeout=3) as r:
                    ok = r.status == 200
            except Exception:
                ok = False
            with _lock:
                _fail_streak[n] = 0 if ok else _fail_streak.get(n, 0) + 1
                was = _health.get(n)
                _health[n] = not (_fail_streak[n] >= HEALTH_FAILS)
                if was and not _health[n]:
                    log(f"backend {n} DOWN ({b['url']})")
                elif not was and _health[n]:
                    log(f"backend {n} UP ({b['url']})")
        time.sleep(HEALTH_EVERY_S)


# ── relay ────────────────────────────────────────────────────────────

class UpstreamError(Exception):
    pass


def relay(handler, be, method, path, body, headers_raw):
    """Stream backend->client. Returns (status, raw). Raises UpstreamError if the
    backend died before sending anything (safe to retry a peer)."""
    # 1110: pick the per-request clock — non-streaming bodies get the long timeout
    # (see READ_TIMEOUT_NONSSE), streaming/GET keep the 20s keepalive cadence.
    _timeout = READ_TIMEOUT
    if body:
        try:
            _timeout = READ_TIMEOUT if json.loads(body).get("stream") else READ_TIMEOUT_NONSSE
        except Exception:
            pass
    conn = http.client.HTTPConnection(be["host"], be["port"], timeout=_timeout)
    fwd = {}
    for k, v in headers_raw:
        if k.lower() in ("host", "content-length", "connection", "transfer-encoding",
                         "accept-encoding", "x-sfnx-backend"):
            continue
        fwd[k] = v
    fwd["Host"] = "%s:%d" % (be["host"], be["port"])
    fwd["Connection"] = "close"
    fwd["Accept-Encoding"] = "identity"
    if body:
        fwd["Content-Length"] = str(len(body))
    try:
        conn.request(method, path, body=body, headers=fwd)
        resp = conn.getresponse()
    except (OSError, socket.timeout) as e:
        conn.close()
        raise UpstreamError(str(e))
    handler.send_response(resp.status)
    is_sse = "text/event-stream" in (resp.getheader("Content-Type") or "")
    for k, v in resp.getheaders():
        if k.lower() in ("transfer-encoding", "connection", "content-length", "keep-alive"):
            continue
        handler.send_header(k, v)
    handler.send_header("Connection", "close")
    handler.close_connection = True
    if is_sse:
        handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    raw = bytearray()
    try:
        while True:
            try:
                chunk = resp.read1(65536)
            except socket.timeout:
                if is_sse:                      # engine thinking: keep the client warm
                    fp = getattr(resp, "fp", None)          # 1110: a read timeout latches
                    sockio = getattr(fp, "raw", None)        # socketio._timeout_occurred; clear
                    if sockio is not None:                   # it or the next read1() raises
                        sockio._timeout_occurred = False     # OSError on a "timed out object"
                    handler.wfile.write(b": keepalive\n\n")   # (and never shadow `raw` here -
                    handler.wfile.flush()                      # it is the audit accumulator)
                    continue
                raise UpstreamError("read timeout (non-SSE)")
            except OSError as e:                # 1110: the flag raced us once anyway
                if is_sse and "timed out" in str(e):
                    fp = getattr(resp, "fp", None)
                    sockio = getattr(fp, "raw", None)
                    if sockio is not None:
                        sockio._timeout_occurred = False
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    continue
                raise
            if not chunk:
                break
            raw += chunk
            handler.wfile.write(chunk)
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass                                    # client went away; still return what we relayed
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return resp.status, bytes(raw)


# ── audit extraction (anthropic and openai SSE shapes) ───────────────

def summarize(raw):
    out_chars, tool_events, usage, finish = 0, 0, {}, None
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        if t == "content_block_start":
            if (ev.get("content_block") or {}).get("type") == "tool_use":
                tool_events += 1
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            if d.get("type") in ("text_delta", "thinking_delta"):
                out_chars += len(d.get("text") or d.get("thinking") or "")
        elif t == "message_delta":
            d = ev.get("delta") or {}
            if d.get("stop_reason"):
                finish = d["stop_reason"]
            if isinstance(ev.get("usage"), dict):
                usage.update(ev["usage"])
        elif t == "message_start":
            m = ev.get("message") or {}
            if isinstance(m.get("usage"), dict):
                usage.update(m["usage"])
        if isinstance(ev.get("usage"), dict):
            usage.update(ev["usage"])
        for c in ev.get("choices") or []:
            d = c.get("delta") or {}
            if d.get("content"):
                out_chars += len(d["content"])
            if d.get("reasoning_content"):
                out_chars += len(d["reasoning_content"])
            if d.get("tool_calls"):
                tool_events += 1
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    return out_chars, tool_events, usage or None, finish


def ledger_alarm(name, usage, out_chars, tool_events, finish):
    done = (usage or {}).get("output_tokens") or (usage or {}).get("completion_tokens") or 0
    try:
        done = int(done)
    except (TypeError, ValueError):
        return
    if done >= 50 and out_chars < done / 4 and tool_events == 0:
        log(f"[ledger] {name}: engine done {done} tokens but this gateway forwarded only "
            f"{out_chars} chars, {tool_events} tool events (finish={finish}) - where did the rest go?")


# ── colon-EOS guard (1112, ported from the 8300 gateway-local.py) ─────────
# Qwen3.8-Flash-Next (UD-Q4_K_XL @ sfnx engines) occasionally mis-samples
# <|im_end|> right after a lead-in ending in ':' / '：' instead of the expected
# \n\n<tool_call>: the turn dies with stop_reason=end_turn and zero tool blocks
# (1112 logged 3 real cases in one day). Clients that reach the fleet through
# this gateway directly (17.5 CC/pmate, scripts) never pass the 8300 guard, so
# the rescue lives here too - at the single choke point for qwen3.8-flash-next.
# Policy (identical to 8300): /v1/messages only. SSE passes every event through
# untouched except the final message_delta + message_stop, which are held until
# the turn's shape is known. If the stream ended on a dangling colon with no
# tool block, keep the client connection open (ping), POST a continuation to
# the SAME backend (its D-slot KV is hot for this conversation) with the
# produced assistant blocks + a nudge, drop the continuation's message_start,
# shift its content_block indexes, merge output_tokens, then flush one final
# message_delta + message_stop. Max 2 continuations. The original request is
# never rewritten - only a follow-up is issued on rescue (doctrine check: this
# gateway still does zero request-body rewriting; the nudge rides a NEW POST).
import re

COLON_GUARD_ENABLED = os.environ.get("SFNX_COLON_GUARD", "1") != "0"
COLON_GUARD_MAX_CONT = 2
COLON_GUARD_MARK = "[sfnx-gw-colon-guard]"
COLON_GUARD_NUDGE = (
    f"{COLON_GUARD_MARK} 你的上一条回复在冒号（: / ：）处意外提前终止，"
    "未发出后续工具调用或正文。禁止在冒号处结束回合！请紧接该冒号之后立即发出"
    "对应的工具调用或继续输出后续内容，切勿重复冒号前已经输出过的文字。"
)
PREAMBLE_GUARD_NUDGE = (
    f"{COLON_GUARD_MARK} 你的上一条回复只输出了行动引导句（表示将要检查/读取/执行）就意外提前终止，"
    "未发出任何工具调用。禁止只说要做某事而不调用工具就结束回合！请立即发出对应的工具调用，"
    "切勿重复刚才已经输出过的引导句。"
)
_ACTION_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"(?:I'll|I will|Let's|Let me|Now let me|First,? let me|First,? I'll)\s+"
    r"(?:start by|begin by|check|read|fetch|inspect|examine|search|grep|run|look at|list|discover|verify|pull|open|query|scan|test)\b"
    r"|(?:我先|现在我先|接下来我先|下面我先|让我先|先来)(?:检查|查看|读取|拉取|获取|看看|跑|执行|查一下|核对|探测|扫描)"
    r")",
    re.IGNORECASE,
)
_TOOL_BLOCK_TYPES = ('tool_use', 'server_tool_use', 'mcp_tool_use')
WEB_SEARCH_PREFIX = 'Perform a web search for the query: '
_PING_EVENT = b'event: ping\ndata: {"type": "ping"}\n\n'


def cg_in_open_code(text):
    if text.count('```') % 2:
        return True
    line = text.rsplit('\n', 1)[-1].replace('```', '')
    return line.count('`') % 2 == 1


def cg_ends_with_dangling_colon(text):
    if not text or cg_in_open_code(text):
        return False
    s = text.rstrip()
    s = s.rstrip('*_').rstrip()
    return bool(s) and s[-1] in (':', '：')


def cg_premature_stop_kind(text, has_tools):
    """1113: returns 'colon' or 'preamble' when text ended prematurely without tools."""
    if not text or cg_in_open_code(text):
        return None
    s = text.rstrip().rstrip('*_').rstrip()
    if not s:
        return None
    if s[-1] in (':', '：'):
        return 'colon'
    if has_tools and len(s) <= 220 and '\n' not in s and s[-1] in ('.', '。'):
        if _ACTION_PREAMBLE_RE.search(s):
            return 'preamble'
    return None


def cg_last_user_text(data, cap=8192):
    try:
        for m in reversed(data.get('messages') or []):
            if m.get('role') == 'user':
                c = m.get('content')
                if isinstance(c, str):
                    return c[:cap]
                if isinstance(c, list):
                    return (' '.join(b.get('text', '') for b in c
                                     if isinstance(b, dict)))[:cap]
    except Exception:
        pass
    return ''


def cg_eligible(data):
    """POST body of /v1/messages (path gate done by caller). Mirrors 8300."""
    if not COLON_GUARD_ENABLED or not isinstance(data, dict):
        return False
    msgs = data.get('messages')
    if not isinstance(msgs, list) or not msgs:
        return False
    if (data.get('max_tokens') or 1024) < 128:
        return False
    last_u = cg_last_user_text(data)
    if 'Do not use tools.' in last_u or WEB_SEARCH_PREFIX in last_u:
        return False
    return True


def cg_pop_sse_events(buf):
    """Split complete SSE events on \\n\\n / \\r\\n\\r\\n keeping raw bytes."""
    events = []
    pos = 0
    n = len(buf)
    while pos < n:
        i_lf = buf.find(b'\n\n', pos)
        i_crlf = buf.find(b'\r\n\r\n', pos)
        if i_lf == -1 and i_crlf == -1:
            break
        if i_crlf != -1 and (i_lf == -1 or i_crlf < i_lf):
            end = i_crlf + 4
        else:
            end = i_lf + 2
        events.append(buf[pos:end])
        pos = end
    return events, buf[pos:]


def cg_parse_sse_json(ev_bytes):
    ev_name = None
    data_lines = []
    for line in ev_bytes.splitlines():
        if line.startswith(b'event:'):
            ev_name = line[6:].strip().decode('utf-8', 'replace')
        elif line.startswith(b'data:'):
            data_lines.append(line[5:].lstrip(b' '))
    if not data_lines:
        return ev_name, None
    raw_data = b'\n'.join(data_lines).strip()
    if not raw_data or raw_data == b'[DONE]':
        return ev_name, None
    try:
        obj = json.loads(raw_data)
        if isinstance(obj, dict):
            return ev_name or obj.get('type'), obj
    except Exception:
        pass
    return ev_name, None


def cg_is_done_frame(ev_bytes):
    for line in ev_bytes.splitlines():
        if line.strip() in (b'data:[DONE]', b'data: [DONE]'):
            return True
    return False


def _cg_clear_read_timeout(resp):
    # 1110: a read timeout latches socketio._timeout_occurred; clear it or the
    # next read1() raises OSError on a "timed out object" (see relay()).
    fp = getattr(resp, "fp", None)
    sockio = getattr(fp, "raw", None)
    if sockio is not None:
        sockio._timeout_occurred = False


def cg_connect(be, path, body_bytes, headers_raw, timeout):
    """POST to a backend, return (conn, resp). Raises UpstreamError (no client
    bytes involved - caller must only call this before sending, or handle it)."""
    conn = http.client.HTTPConnection(be["host"], be["port"], timeout=timeout)
    fwd = {}
    for k, v in headers_raw:
        if k.lower() in ("host", "content-length", "connection", "transfer-encoding",
                         "accept-encoding", "x-sfnx-backend"):
            continue
        fwd[k] = v
    fwd["Host"] = "%s:%d" % (be["host"], be["port"])
    fwd["Connection"] = "close"
    fwd["Accept-Encoding"] = "identity"
    if body_bytes:
        fwd["Content-Length"] = str(len(body_bytes))
    try:
        conn.request("POST", path, body=body_bytes, headers=fwd)
        resp = conn.getresponse()
    except OSError as e:
        try:
            conn.close()
        except Exception:
            pass
        raise UpstreamError(str(e))
    return conn, resp


def relay_guarded_sse(handler, be, be_name, path, body, headers_raw, data):
    """Stream /v1/messages with held final frames + colon-EOS rescue.
    Returns (status, raw_emitted, cont_used). UpstreamError only before any
    byte goes to the client (same contract as relay())."""
    conn, resp = cg_connect(be, path, body, headers_raw, READ_TIMEOUT)
    status = resp.status
    is_sse = "text/event-stream" in (resp.getheader("Content-Type") or "")
    handler.send_response(status)
    for k, v in resp.getheaders():
        if k.lower() in ("transfer-encoding", "connection", "content-length", "keep-alive"):
            continue
        handler.send_header(k, v)
    handler.send_header("Connection", "close")
    handler.close_connection = True
    if is_sse:
        handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()

    raw = bytearray()
    cont_used = 0
    client_gone = [False]

    def emit(b):
        if not b:
            return
        raw.extend(b)   # += would rebind raw as a local of this closure
        try:
            handler.wfile.write(b)
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            client_gone[0] = True

    if status != 200 or not is_sse:
        # error page / wrong shape: plain passthrough, nothing to rescue
        try:
            rest = resp.read()
            emit(rest)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return status, bytes(raw), 0

    sent_data = data
    cont_messages = list(sent_data.get('messages') or [])
    idx_offset = 0
    max_idx = -1
    total_out_tokens = 0
    base_usage = None
    held_delta = None          # (ev_name, obj, orig_bytes)
    held_stop = None           # raw bytes
    held_done = None           # trailing [DONE] frame if the engine sends one

    try:
        for cont_idx in range(COLON_GUARD_MAX_CONT + 1):
            carry = b''
            pass_text = []
            pass_thinking = []
            pass_sig = []
            pass_used_tool = False
            pass_out_tokens = 0
            pass_delta = None
            pass_stop = None

            while not client_gone[0]:
                try:
                    chunk = resp.read1(65536)
                except socket.timeout:
                    _cg_clear_read_timeout(resp)
                    emit(b": keepalive\n\n")
                    continue
                except OSError as e:
                    if "timed out" in str(e):
                        _cg_clear_read_timeout(resp)
                        emit(b": keepalive\n\n")
                        continue
                    raise
                if not chunk:
                    break
                carry += chunk
                events, carry = cg_pop_sse_events(carry)
                for ev in events:
                    ev_name, obj = cg_parse_sse_json(ev)
                    if obj is None:
                        if cg_is_done_frame(ev):
                            if cont_idx == 0:
                                held_done = ev
                        else:
                            emit(ev)
                        continue
                    etype = obj.get('type') or ev_name
                    if etype == 'message_start':
                        if cont_idx == 0:
                            emit(ev)
                        # continuation passes: drop the duplicate message_start
                    elif etype in ('content_block_start', 'content_block_delta',
                                   'content_block_stop'):
                        orig_i = obj.get('index', 0)
                        orig_i = orig_i if isinstance(orig_i, int) else 0
                        new_i = orig_i + idx_offset
                        if new_i > max_idx:
                            max_idx = new_i
                        if etype == 'content_block_start':
                            cb = obj.get('content_block') or {}
                            bt = cb.get('type')
                            if bt in _TOOL_BLOCK_TYPES:
                                pass_used_tool = True
                            elif bt == 'thinking':
                                if cb.get('thinking'):
                                    pass_thinking.append(cb['thinking'])
                                if cb.get('signature'):
                                    pass_sig.append(cb['signature'])
                            elif bt == 'text' and cb.get('text'):
                                pass_text.append(cb['text'])
                        elif etype == 'content_block_delta':
                            d = obj.get('delta') or {}
                            dt = d.get('type')
                            if dt == 'thinking_delta' and d.get('thinking'):
                                pass_thinking.append(d['thinking'])
                            elif dt == 'signature_delta' and d.get('signature'):
                                pass_sig.append(d['signature'])
                            elif dt == 'text_delta' and d.get('text'):
                                pass_text.append(d['text'])
                            elif dt == 'input_json_delta':
                                pass_used_tool = True
                        if idx_offset != 0 and 'index' in obj:
                            obj['index'] = new_i
                            emit((f"event: {ev_name or etype}\n"
                                  f"data: {json.dumps(obj, ensure_ascii=False)}\n\n").encode('utf-8'))
                        else:
                            emit(ev)
                    elif etype == 'message_delta':
                        pass_delta = (ev_name or 'message_delta', obj, ev)
                        u = obj.get('usage') or {}
                        try:
                            pass_out_tokens = int(u.get('output_tokens') or 0)
                        except (TypeError, ValueError):
                            pass_out_tokens = 0
                        if cont_idx == 0:
                            base_usage = dict(u)
                    elif etype == 'message_stop':
                        pass_stop = ev
                    else:
                        emit(ev)

            total_out_tokens += pass_out_tokens
            if pass_delta is not None:
                held_delta = pass_delta
            if pass_stop is not None:
                held_stop = pass_stop

            stop_reason = ((held_delta[1].get('delta') or {}).get('stop_reason')
                           if held_delta else None)
            full_text = ''.join(pass_text)
            pkind = cg_premature_stop_kind(full_text, bool(sent_data.get('tools')))
            needs_rescue = (held_delta is not None
                            and stop_reason == 'end_turn'
                            and not pass_used_tool
                            and pkind is not None
                            and cont_idx < COLON_GUARD_MAX_CONT
                            and not client_gone[0])
            if not needs_rescue:
                break

            # colon/preamble + end_turn with no tools: keep the stream open, continue on
            try:
                conn.close()
            except Exception:
                pass
            emit(_PING_EVENT)
            asst_blocks = []
            if pass_thinking:
                asst_blocks.append({'type': 'thinking',
                                    'thinking': ''.join(pass_thinking),
                                    'signature': ''.join(pass_sig)})
            asst_blocks.append({'type': 'text', 'text': full_text})
            nudge = PREAMBLE_GUARD_NUDGE if pkind == 'preamble' else COLON_GUARD_NUDGE
            cont_messages = cont_messages + [
                {'role': 'assistant', 'content': asst_blocks},
                {'role': 'user', 'content': [{'type': 'text', 'text': nudge}]},
            ]
            payload = dict(sent_data)
            payload['messages'] = cont_messages
            payload['stream'] = True
            cbody = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            log(f"{COLON_GUARD_MARK} {be_name}: ended on {pkind} "
                f"({full_text.strip()[-50:]!r}, out={pass_out_tokens} tok, tools=0); "
                f"continuing mid-stream ({cont_idx + 1}/{COLON_GUARD_MAX_CONT})")
            try:
                conn, resp = cg_connect(be, path, cbody, headers_raw, READ_TIMEOUT)
            except UpstreamError as e:
                log(f"{COLON_GUARD_MARK} {be_name}: continuation connect failed "
                    f"({e}); flushing held end_turn")
                break
            if resp.status != 200 or 'event-stream' not in (resp.getheader('Content-Type') or ''):
                log(f"{COLON_GUARD_MARK} {be_name}: continuation status={resp.status}; "
                    "flushing held end_turn")
                try:
                    conn.close()
                except Exception:
                    pass
                break
            cont_used += 1
            idx_offset = max_idx + 1
    except (BrokenPipeError, ConnectionResetError):
        client_gone[0] = True
    except (OSError, UpstreamError) as e:
        # upstream died mid-stream: fall through to the held-frame flush so the
        # client at least gets a well-formed (if truncated) turn end
        log(f"{COLON_GUARD_MARK} {be_name}: upstream error mid-stream "
            f"({type(e).__name__}: {e}); flushing held frames")

    if held_delta is not None:
        if cont_used == 0:
            emit(held_delta[2])
        else:
            ev_n, d_obj, _ = held_delta
            merged = dict(d_obj)
            u = dict(base_usage) if base_usage else dict(merged.get('usage') or {})
            u['output_tokens'] = total_out_tokens
            merged['usage'] = u
            emit((f"event: {ev_n}\ndata: {json.dumps(merged, ensure_ascii=False)}\n\n").encode('utf-8'))
            log(f"{COLON_GUARD_MARK} {be_name}: rescued, conts={cont_used}, "
                f"final_stop={(merged.get('delta') or {}).get('stop_reason')}, "
                f"total_out={total_out_tokens}")
    if held_stop is not None:
        emit(held_stop)
    if held_done is not None:
        emit(held_done)
    try:
        conn.close()
    except Exception:
        pass
    return status, bytes(raw), cont_used


def _cg_send_response(handler, status, resp_headers, body_bytes):
    handler.send_response(status)
    for k, v in resp_headers:
        if k.lower() in ("transfer-encoding", "connection", "content-length", "keep-alive"):
            continue
        handler.send_header(k, v)
    handler.send_header("Content-Length", str(len(body_bytes)))
    handler.send_header("Connection", "close")
    handler.close_connection = True
    handler.end_headers()
    try:
        handler.wfile.write(body_bytes)
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass


def relay_guarded_nonstream(handler, be, be_name, path, body, headers_raw, data):
    """Buffered non-streaming /v1/messages + colon-EOS rescue. The whole engine
    response is read before any byte goes out, so a rescue can stitch the JSON
    and a mid-read failure still allows clean failover (UpstreamError)."""
    conn, resp = cg_connect(be, path, body, headers_raw, READ_TIMEOUT_NONSSE)
    status = resp.status
    hdrs = list(resp.getheaders())
    try:
        rbody = resp.read()
    except OSError as e:
        try:
            conn.close()
        except Exception:
            pass
        raise UpstreamError(f"non-SSE read failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if status != 200:
        _cg_send_response(handler, status, hdrs, rbody)
        return status, rbody, 0
    try:
        j = json.loads(rbody)
        if not isinstance(j, dict) or not isinstance(j.get('content'), list):
            raise ValueError("not an anthropic message object")
    except Exception:
        _cg_send_response(handler, status, hdrs, rbody)
        return status, rbody, 0

    base_usage = dict(j.get('usage') or {})
    try:
        total_out = int(base_usage.get('output_tokens') or 0)
    except (TypeError, ValueError):
        total_out = 0
    cont_messages = list(data.get('messages') or [])
    cur_blocks = list(j.get('content') or [])
    cur_stop = j.get('stop_reason')
    cont_used = 0

    for ci in range(COLON_GUARD_MAX_CONT):
        used_tool = any(isinstance(b, dict) and b.get('type') in _TOOL_BLOCK_TYPES
                        for b in cur_blocks)
        txt = ''.join(b.get('text', '') for b in cur_blocks
                      if isinstance(b, dict) and b.get('type') == 'text')
        pkind = cg_premature_stop_kind(txt, bool(data.get('tools')))
        if not (cur_stop == 'end_turn' and not used_tool
                and pkind is not None):
            break
        nudge = PREAMBLE_GUARD_NUDGE if pkind == 'preamble' else COLON_GUARD_NUDGE
        cont_messages = cont_messages + [
            {'role': 'assistant', 'content': cur_blocks},
            {'role': 'user', 'content': [{'type': 'text', 'text': nudge}]},
        ]
        payload = dict(data)
        payload['messages'] = cont_messages
        payload['stream'] = False
        cbody = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        log(f"{COLON_GUARD_MARK} {be_name}: nonstream ended on {pkind} "
            f"({txt.strip()[-50:]!r}); continuing ({ci + 1}/{COLON_GUARD_MAX_CONT})")
        c2 = None
        try:
            c2, r2 = cg_connect(be, path, cbody, headers_raw, READ_TIMEOUT_NONSSE)
            b2 = r2.read()
        except (UpstreamError, OSError) as e:
            log(f"{COLON_GUARD_MARK} {be_name}: nonstream continuation failed "
                f"({e}); flushing original")
            break
        finally:
            if c2 is not None:
                try:
                    c2.close()
                except Exception:
                    pass
        if r2.status != 200:
            log(f"{COLON_GUARD_MARK} {be_name}: nonstream continuation "
                f"status={r2.status}; flushing original")
            break
        try:
            j2 = json.loads(b2)
        except Exception:
            break
        nb = list(j2.get('content') or [])
        j['content'].extend(nb)
        cur_blocks = nb
        cur_stop = j2.get('stop_reason')
        j['stop_reason'] = cur_stop
        total_out += int((j2.get('usage') or {}).get('output_tokens') or 0)
        cont_used += 1

    if cont_used > 0:
        base_usage['output_tokens'] = total_out
        j['usage'] = base_usage
        rbody = json.dumps(j, ensure_ascii=False).encode('utf-8')
        log(f"{COLON_GUARD_MARK} {be_name}: nonstream rescued, conts={cont_used}, "
            f"final_stop={cur_stop}, total_out={total_out}")
    _cg_send_response(handler, status, hdrs, rbody)
    return status, rbody, cont_used


def relay_guarded(handler, be, be_name, path, body, headers_raw, data):
    if data.get("stream"):
        return relay_guarded_sse(handler, be, be_name, path, body, headers_raw, data)
    return relay_guarded_nonstream(handler, be, be_name, path, body, headers_raw, data)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _pick_and_relay(self, method, body=None):
        path = self.path
        pin = self.headers.get("X-Sfnx-Backend")
        key = conv_key(body or b"")
        name, be = route(key, pin=pin)
        if not name:
            self.send_error(503, "No healthy sFNX backends")
            return None, None, None, None, None, 0
        cg_data = None
        if (method == "POST" and body and COLON_GUARD_ENABLED
                and path.split("?", 1)[0] == "/v1/messages"):
            try:
                d = json.loads(body)
            except Exception:
                d = None
            if cg_eligible(d):
                cg_data = d
        t0 = time.time()
        cont_used = 0
        try:
            if cg_data is not None:
                status, raw, cont_used = relay_guarded(self, be, name, path, body,
                                                       list(self.headers.items()), cg_data)
            else:
                status, raw = relay(self, be, method, path, body, list(self.headers.items()))
        except UpstreamError as e:
            alt, altbe = route(key, pin=None)
            if alt and alt != name:
                log(f"failover {name} -> {alt} ({e})")
                try:
                    if cg_data is not None:
                        status, raw, cont_used = relay_guarded(self, altbe, alt, path, body,
                                                               list(self.headers.items()), cg_data)
                    else:
                        status, raw = relay(self, altbe, method, path, body, list(self.headers.items()))
                    name = alt
                except UpstreamError as e2:
                    self.send_error(502, f"sFNX backends unreachable ({e2})")
                    return None, None, None, None, None, 0
            else:
                self.send_error(502, f"sFNX backend {name} unreachable ({e})")
                return None, None, None, None, None, 0
        return name, status, raw, key, t0, cont_used

    def do_GET(self):
        if self.path == "/health":
            with _lock:
                body = json.dumps({"backends": {n: {"url": b["url"], "pool": b["pool"],
                                                    "up": _health.get(n)} for n, b in _backends.items()},
                                   "sticky": len(_sticky)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            self.wfile.write(body)
            return
        r = self._pick_and_relay("GET")
        if r[0] is None:
            return
        name, status, raw, key, t0, _cg = r
        audit({"ts": datetime.now().strftime("%H:%M:%S"), "ip": self.client_address[0], "path": self.path,
               "key": key[:16], "be": name, "status": status, "ms": int((time.time() - t0) * 1000),
               "fwd": len(raw)})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        r = self._pick_and_relay("POST", body)
        if r[0] is None:
            return
        name, status, raw, key, t0, cg_conts = r
        out_chars, tool_events, usage, finish = summarize(raw)
        entry = {"ts": datetime.now().strftime("%H:%M:%S"), "ip": self.client_address[0], "path": self.path,
                 "key": key[:16], "be": name, "status": status, "ms": int((time.time() - t0) * 1000),
                 "fwd": len(raw), "out_chars": out_chars, "tools": tool_events,
                 "done_tok": (usage or {}).get("output_tokens") or (usage or {}).get("completion_tokens"),
                 "finish": finish, "usage": usage}
        if cg_conts:
            entry["cg"] = cg_conts   # colon-guard continuations used this request
        audit(entry)
        ledger_alarm(name, usage, out_chars, tool_events, finish)


if __name__ == "__main__":
    load_config()
    threading.Thread(target=health_loop, daemon=True).start()
    threading.Thread(target=sticky_save_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    log(f"sfnx-gw :{LISTEN_PORT} | backends={sorted(_backends)} | affinity=first-user-msg | no body rewriting")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
