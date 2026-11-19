#!/usr/bin/env python3
"""sfnx 长上下文工具调用验证：深埋检索 + tool call 格式双考验
用法: tool_longctx_test.py <base_url> <target_prompt_tokens> <label>
例:   tool_longctx_test.py http://<FLEET_HOST>:8093 300000 atlas11-300K
判定: tool_calls 存在 + 函数名对 + 参数 JSON 可解析 + 检索值精确正确
覆盖场景: 用户报“长上下文精度下降/工具调用失败”时的端到端回归
生成器校准: ~63.5 tok/段(中文填充段), 目标事实深埋 ~55% 处, 带同形干扰项(其它客户订单号)
注: 单发 PASS 不等于无统计性发散(序列特异 bug)——KL exact 才是铁证, 本脚本验证实战场景
"""
import json, time, sys, urllib.request

BASE, TARGET, LABEL = sys.argv[1], int(sys.argv[2]), sys.argv[3]

FILLER = ("物流仓库第 {} 号货架盘点记录：今日入库批次 A-{}，数量 {} 件，湿度 {}%，复核人王某。"
          "气象站 {} 点观测：气温 {}.{} 度，风速 {} 米每秒。")
DECOY = "普通客户{}的订单编号 ORD-{}，客户识别号 CUST-{}，已出库。"

def build(target_tokens):
    n = int(target_tokens / 63.5) + 40
    paras, planted = [], False
    plant_at = int(n * 0.55)
    for i in range(n):
        paras.append(FILLER.format(i, i % 89 + 1, i * 7 % 400, i % 60 + 20,
                                   i % 24, i % 35, i % 10, i % 12 + 1))
        if i % 97 == 13:
            paras.append(DECOY.format("李娜", 10000 + i, 20000 + i))
        if i % 97 == 47:
            paras.append(DECOY.format("王强", 30000 + i, 40000 + i))
        if i == plant_at:
            paras.append("特别记录：VIP 客户张伟的订单编号 ORD-77219，"
                         "客户识别号 CUST-55308，已进入分拣中心，优先处理。")
            planted = True
    assert planted
    paras.append("请调用工具查询 VIP 客户张伟的订单状态。")
    return "\n".join(paras)

TOOLS = [{"type": "function", "function": {
    "name": "get_order_status",
    "description": "查询订单的当前物流状态",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string", "description": "订单编号"},
        "customer_id": {"type": "string", "description": "客户识别号"}},
        "required": ["order_id", "customer_id"]}}}]

body = {"model": "qwen3.8-flash-next", "max_tokens": 3000,
        "messages": [{"role": "user", "content": build(TARGET)}],
        "tools": TOOLS, "tool_choice": "auto"}
req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.monotonic()
with urllib.request.urlopen(req, timeout=900) as r:
    res = json.load(r)
t = time.monotonic() - t0

u = res.get("usage", {})
pt = u.get("prompt_tokens", 0)
msg = res["choices"][0]["message"]
finish = res["choices"][0].get("finish_reason")
tcs = msg.get("tool_calls") or []

verdict, detail = "FAIL", ""
if tcs:
    fn = tcs[0]["function"]
    try:
        args = json.loads(fn["arguments"])
        if fn["name"] == "get_order_status" and args.get("order_id") == "ORD-77219" \
           and args.get("customer_id") == "CUST-55308":
            verdict, detail = "PASS", f"args={args}"
        else:
            detail = f"wrong: name={fn['name']} args={fn['arguments'][:200]}"
    except Exception as e:
        detail = f"unparseable args: {fn.get('arguments','')[:200]} ({e})"
else:
    c = (msg.get("content") or "")[:150].replace("\n", " ")
    detail = f"no tool_call, finish={finish}, content={c!r}"

print(f"[{LABEL}] {pt} tok ctx | {t:.1f}s | prefill≈{pt/max(t-2,0.1):.0f} tok/s | "
      f"out={u.get('completion_tokens')} | {verdict} | {detail}")
