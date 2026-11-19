---
name: sfnx-pd-handoff-verify
description: sfnx PD hand-off 链路健康验证与排障——三件套判读法、KV 复用矩阵判据（"第二次还慢"专用）、受控 A/B 测法与占位异步坑、引擎二进制能力对拍（克隆部署必做）、CLOSE-WAIT 错误指纹、拒绝路径快撤验证、1116 两连案（全灭+PLOAD 零 ckpt）的完整复用流程
---

# sfnx PD hand-off 验证与排障

触发：用户报 sfnx/flash-next "负载不均"“P 不干活”“D 压死”；克隆/换机部署 P 或 D 后验收；PD 链行为可疑（hand-off 决策与实际完成对不上）；任何 `PD stream skipped` 日志行。

背景：1116 案——1113 克隆 P 到 atlas1 时 front 层照搬但引擎二进制用本地版（无 PDSTREAM），35 次 hand-off 全 502 静默回退且每次白等 300s。front 能力 ≠ 引擎能力是克隆部署的系统性陷阱。

## 一、三件套判读法（判链路死活，勿看 P CPU 单指标）

```bash
# 件1: D 侧决策（atlas1）
ssh atlas1 'journalctl -u sfnx-8093 --since "-2 hours" --no-pager | grep -E "PD decision|handing off" | tail -20'
# 件2: P 侧推流（缺=链断在 P）
ssh atlas1 'journalctl -u sfnx-p-8094 --since "-2 hours" --no-pager | grep streamed'
# 件3: D 侧完成（缺=链断在回程/PLOAD）
ssh atlas1 'journalctl -u sfnx-8093 --since "-2 hours" --no-pager | grep "hand-off:"'
```

三件齐=链通。件1 有决策但件2/3 空 = hand-off 全灭（见二）。决策三形态分布判"该不该忙"：`idle engine ... local`（闲走本地，正确）／`N tokens, N slot(s) busy ... handing off`（忙甩 P，正确）／`no decode here, the P's serial lane is mid-read - local`（P 占道回避，正确）。

## 一点五、KV 复用矩阵（"第二次还要 N 秒"专用判据）

判复用**勿看客户端时延**（含排队/流式开销），看引擎记账行：

```bash
# D 侧逐请求复用（serve/sfnx-8093.log 尾部）
ssh atlas1 'grep -E "PLOAD slot|slot [0-9]: prompt" /data/sfnx-prod/serve/sfnx-8093.log | tail -5'
# P 侧推流复用（serve/sfnx-p-8094.log，P 引擎行全在此文件不在 journald）
ssh atlas1 'grep PDSTREAM /data/sfnx-prod/serve/sfnx-p-8094.log | tail -5'
```

三路矩阵（同对话同形状 follow-up，45K 实测基准）：

| 路径 | 健康指纹 | 病态 |
|---|---|---|
| read→park→local 恢复 | `prompt N = ~N reused + 少量 read in <1s` | `0 reused + 全量 read ~12s` |
| hand-off→PLOAD→再 hand-off | P 侧 `PDSTREAM: N tokens (~N reused)`，推流 1.5-3s | `(0 reused)` 首发全价属正常 |
| hand-off→PLOAD→park→local 恢复 | `PLOAD slot` 后 `~N reused in <1s`（1116 reload_dump 补丁后） | **补丁前 0 reused 全量重读**——即"第二次还慢"病根 |

读法：`slot 0: prompt 45131 tokens = 45083 reused + 48 read in 372 ms`。发散点在 assistant 回合是常态（引擎历史含完整生成文本，HTTP 回传只有 content）——恢复靠 ≤发散点的 checkpoint；`0 reused` 一律先查 ckpt 链（dump 字节格式不含 ckpt，全靠装载侧重建）。

## 二、引擎二进制能力对拍（克隆部署必做；front 恒有路由≠引擎支持）

```bash
# 现役引擎能力：PDSTREAM（P 侧推流）+ PLOAD（D 侧装载）都应 >0
ssh atlas1 'strings -a /data/sfnx-prod/bin/sfnx-pd | grep -c "^PDSTREAM"; strings -a /data/sfnx-prod/bin/sfnx | grep -c "^PLOAD"'
# 8300 网关 /score 里的后端名单要与实际部署一致（死条目=谎报，见四）
```

PDSTREAM 参考实现二进制：atlas0 `/data/sfnx0/sfnx-prod/bin/sfnx`（11-11 版，md5 4c70094c…）。config 的 `"exe"` 字段决定引擎路径，改后须重启单元（启动读一次铁律）。

## 三、直接探针（1 分钟出真相，比读日志快）

## 三、直接探针（1 分钟出真相，比读日志快）

### 受控 A/B（测复用率的正确姿势，两个时序坑）

- **占位请求必须异步 fire**（threading.Thread/Timer），`time.sleep(4)` 后再打 B——写错成同步 call 会阻塞 14s，B 到达时引擎已闲、全走 local，白测一轮（1116 实际踩过）。
- **测 local 路须等空闲窗口**：发 follow-up 前检查 `journalctl -u sfnx-8093 --since -10s` 无 `PD decision|done:|hand-off` 行；否则一发生产流量在途就把你的测试件挤到 hand-off 路。
- follow-up 的正确形状：**首条 user 消息逐字节不动**，新内容追加在 assistant 回合之后（对话键=首条消息）。消息内追加（同一 user 消息里改尾部）不会命中缓存条目。
- 判定：跑完拉引擎 slot 行（见一点五），不要看客户端秒数。

```bash
# 从 D 同机发真实 /pd/pread_stream：监听 socket + POST offer
# 见 /tmp/patch 脚本同款——关键是 listener 先起、offer 线程后发（顺序反了死锁）
# 健康回包: {"ok": true, "tokens": 20000, ..., "bytes": ...} + NONCE 首行 + dump 字节流
# 病态回包: 502 {"ok": false, "error": "ERR PDSTREAM: not supported by sfnx"} = 引擎无 PDSTREAM
```

并发实弹（A 占 decode 槽 + B 大 prompt 到达 → 应甩 P）：

```bash
ssh atlas1 'python3 - <<PYEOF
import json, threading, time, urllib.request
URL = "http://127.0.0.1:8093/v1/chat/completions"
def fire(name, prompt, mt, delay=0):
    def run():
        t0 = time.time()
        body = json.dumps({"model":"qwen3.8-flash-next","messages":[{"role":"user","content":prompt}],"max_tokens":mt,"stream":False}).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={"Content-Type":"application/json"}), timeout=180) as r:
                u = json.loads(r.read()).get("usage", {})
                print(name, round(time.time()-t0,1), "s ptok=", u.get("prompt_tokens"))
        except Exception as e: print(name, "FAIL", repr(e))
    threading.Timer(delay, run).start()
pa = "数理逻辑是研究推理形式的学科。" * 1400        # ~11K tok 槽位占用
pb = ("人工智能发展史涵盖符号主义、连接主义与行为主义三条脉络，各自兴衰交替。"
      "从图灵测试到深度学习浪潮，每一次范式转移都重塑了行业格局。") * 4300   # ~155K tok
fire("A-slot", pa, 3000, 0); time.sleep(8); fire("B-fresh", pb, 64, 0)
time.sleep(90)
PYEOF'
```

## 四、指纹与坑（Pitfalls）

- **CLOSE-WAIT + recv-q 62 字节**：`ss -tn state close-wait | grep 8094`——P 已回错误响应（62 字节 JSON）关闭连接、D front 不读。每条 = 一次被拒的 hand-off。
- **应用层失败零日志**：`pd_handoff_stream` 的 offer not-ok 原版静默 `return None`；只有 socket 异常才打 `PD stream skipped`。修复版（1116，server.py a76838d0）accept 5s 轮询、拒绝即打 `the peer refused (...)` 并秒退（原版白等 pd_peer_timeout 300s）。
- **P 侧 journal 只在 systemd 打**：引擎 log 文件（serve/sfnx-p-*.log）只记启动横幅，服务行全在 journald；P 引擎日志行不带时间戳。
- **`done: 48 tokens in 308 s`** = decode 只花 0.3s、其余全是死等超时——300s 病的典型指纹，grep 慢请求即见。
- **重启 P 前看流量**：轻载窗口重启无感；P 引擎加载 ~95s，`/v1/models` 200 才算 ready。
- **8300 /score 是 24.23 主网关的代理**：glm5.3-b tier 死条目源头在 24.23 `gateway@8000` 内存——改 `gateway-config.json` 后**主网关也要重启**（1113 漏了这刀，死条目谎报到 1116）。curl 探测必须 `--noproxy "*"`（24.23 有全局代理劫持 localhost，503 假象）。
- **轻载 idle-local 不是病**（1110 设计：闲 D 本地 3600 tok/s 更快）；判病看忙时三件套，勿看瞬时 P CPU。
- **make 会原地覆盖运行中的 bin/sfnx 但不 ETXTBSY**（nvcc 链接走 temp+rename）；旧二进制被覆盖后**从 `/proc/<engine-pid>/exe` 捞回**（运行进程持旧 inode）再谈回滚。引擎 pid 用 `ps aux | grep "bin/sfnx --serve"` 找（systemd 单元 MainPID 是 front 的 python）。
- **reload_dump 的 checkpoint 必须从引擎造**（`e_.save_state(false)`：tokens=history、只抓尾页、与 load_state 对称）——复用 dump 的 SeqState 伪造 kv=false/空 bytes，恢复时 load_state 校验布局即抛异常。
- 对生产树打补丁前先 `g++-12 -fsyntax-only`/nvcc `-c -o /dev/null` 预检 session.cpp——全量 make ~25 分钟，语法错别等链接才发现。

## 五、验证步骤（收尾清单）

1. 三件套 15 分钟窗口：决策三形态都有、`hand-off: N tokens' K/V` 与 `PD streamed out` 成对出现。
2. `ss -tn state close-wait | grep -c 8094` = 0。
3. 死端口快撤单测（拒绝路径 ≤5s 回退，原 300s）：见 1116 档案 `sfnx-pd-handoff-dead-binary-1116.md` 的 stub 测试。
4. 8300 全链探针：`curl http://127.0.0.1:8300/v1/chat/completions -d '{"model":"qwen3.8-flash-next",...}'` 回"正常"。
5. `nvidia-smi`：忙时 D 卡（0-3）与 P 卡（4-7）都有利用率；闲时全 0 是正常回落。

## 部署变更记录（本 skill 出处）

- 1116：P 引擎换 `bin/sfnx-pd`（atlas0 版）+ config `.bak-1116-pdbin`；D front accept 快撤补丁 + `.bak-1116-pdaccept`；24.23 主网关重启清 /score 死条目。
- 1116（二连案，"第二次 8 秒"）：D 引擎 `reload_dump` 补丁（session.hpp:119/session.cpp:349/tools/sfnx.cpp:1130，备份 `.bak-1116-ploadckpt` 三文件+bin）；D 二进制 4d96c146（旧 7adf87d4 存 `bin/sfnx.bak-1116-pdckpt`，从 /proc/exe 捞回）；只重启 sfnx-8093。P 引擎无此问题未动。
- 参考档案：memory `sfnx-pd-handoff-dead-binary-1116.md` + `sfnx-pload-ckpt-restore-1116.md`；skill:sfnx-pd-disaggregation（PREAD/PDUMP 机制总纲）。
