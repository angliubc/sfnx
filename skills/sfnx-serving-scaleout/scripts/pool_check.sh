#!/bin/bash
# 1110 sfnx 聚合池验收：经本机 8300 打 N 笔唯一首消息的小请求，
# 再统计 sfnx-gw 审计日志里这段时间 be= 的后端分布（应铺满 6 台）。
N=${1:-12}
START=$(date +%H:%M:%S)
for i in $(seq 1 "$N"); do
  curl -s -m 150 http://127.0.0.1:8300/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8-flash-next\",\"messages\":[{\"role\":\"user\",\"content\":\"unique-probe-1110-$i-$RANDOM: reply with the single word ok\"}],\"max_tokens\":64,\"stream\":false}" \
    -o /dev/null &
done
wait
echo "sent $N probes since $START; backend distribution:"
ssh p40-23 "tail -n 400 /home/xh/sfnx-gw/gw-logs/sfnx-gw-$(date +%Y%m%d).log" \
  | START="$START" python3 -Ic '
import sys, os, json, collections
start = os.environ["START"]
c = collections.Counter()
for line in sys.stdin:
    try:
        d = json.loads(line)
    except Exception:
        continue
    if str(d.get("ts", "")) >= start:
        c[d.get("be")] += 1
for k, v in c.most_common():
    print(f"  {v:4d}  {k}")
'
