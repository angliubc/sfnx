#!/usr/bin/env python3
"""PD E2E probe — run ON the decode (D) machine:  pd_e2e_probe.py [port] [filler_reps]

Long prompt (filler + 17*23) through the D front; reports first-token latency and
the answer. Working hand-off = fast first token (remote prefill + PLOAD); a failed
hand-off silently falls back to local prefill — correctness never at risk, only
TTFT. Confirm which path fired via `PD hand-off: N tokens' K/V from <peer>` in
serve/front-<port>.log. urllib only; empty ProxyHandler because these hosts
export http_proxy which breaks localhost peer calls.
"""
import json, sys, time, urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "8093"
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 60

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
model = json.load(op.open(f"http://127.0.0.1:{port}/v1/models", timeout=10))["data"][0]["id"]

filler = "这是一段用于测试预填充分流的长文本。模型需要通读全部内容后回答最后的问题。" * reps
content = filler + "\n\n问题：17*23 等于多少？只回答数字。"

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"model": model,
                     "messages": [{"role": "user", "content": content}],
                     "max_tokens": 200, "temperature": 0, "stream": True}).encode(),
    headers={"Content-Type": "application/json"})

t0, first, parts = time.time(), None, []
with op.open(req, timeout=600) as r:
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        d = json.loads(line[6:])
        ch = (d.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or ""
        if piece:
            if first is None:
                first = time.time() - t0
            parts.append(piece)

text = "".join(parts)
ft = f"{first:.2f}s" if first is not None else "n/a"
print(f"model={model}  first_token={ft}  total={time.time()-t0:.1f}s")
print("tail:", text[-160:].replace("\n", " "))
print("VERDICT:", "ANSWER_391_OK" if "391" in text else "NO_391_INSPECT_ABOVE")
