---
name: sfnx-serving-scaleout
description: "sfnx（qwen3.8-flash-next 1M 引擎）生产实例扩容并聚合进 sfnx-gw 的完整操作规程：换机复刻生产树、权重软链零拷贝、cuda 版本对齐、setsid 起实例、本机中继加端口、sfnx-gw 一次性切六后端、24.23 容量与分组改写、分布/吞吐验收。触发：任何『再起 N 个 sfnx / flash-next 实例』『把实例加进聚合池』『收编池外 sfnx 实例』『换机复刻 1M 引擎』『扩 sfnx-gw 后端』任务——先查本条勿重新侦察。"
metadata:
  type: reference
---

# sfnx 生产实例扩容 + 聚合进 sfnx-gw

> **现役拓扑（1113 定版）**：单机 1P1D = atlas1 同机 D:8093（卡 0-3，--vision）+ P:8094（卡 4-7，pd_peers 走本机 IPoIB `<FLEET_HOST>:8094`）。sfnx-gw 单后端 atlas1-8093（中继 18095），max_concurrent=4（=单 D batch）。atlas0/8/10 全退役。PD 机制见 skill:sfnx-pd-disaggregation，缩编档案见 memory `sfnx-1p1d-single-node-1113`。下文六实例表为历史形态，扩容步骤仍适用。

1110 定版（六实例案，见 memory `sfnx-6pool-atlas1-atlas8-1110`、网关机制 `sfnx-atlas2-relay-1105`）。

## 触发条件
用户要"再部署 N 个 qwen3.8-flash-next 实例/一起聚合/加进池/收编池外实例"，或某台 sfnx 机要重建。

## 事实表（先核对是否仍成立，别当永恒真理）
| 项 | 值 |
|---|---|
| 聚合网关 | `sfnx-gw` @ 24.23(p40-23) `/home/xh/sfnx-gw/`，systemd `sfnx-gw.service`，监听 :8090 |
| 上游 | 24.23 `gateway.py 8000`（配置 `/home/xh/gateway-config.json`，**300s 自动刷新**）；本机 8300 → 24.23 |
| 现役后端 | 1110 起六台：atlas2(:223) 8093/8094 ← 中继 18093/18094；atlas1(:222) 8093/8094 ← 18095/18096；atlas8(:229) 8093/8094 ← 18097/18098 |
| 生产树蓝本 | `atlas2:/data/sfnx`（bin md5 `7adf87d46d6a683abe6477e2791f3fda`）；参数 `--max-context 1001390 --spec 6 --lookup 15 --conversation-cache-mib 65536 --batch 6 --rope-scale 3.82 --vision` |
| 权重 | GGUF 四片 UD-Q4_K_XL（shard-1 md5 `1958631667f5320c7969d047b43b5829`）+ `mmproj-F16`(=BF16 改名)；atlas1 本地已有 `/data/strata-local/gguf`，fs00 原件 `/fs00/models/Qwen3.8-Flash-Next-GGUF/` |
| 单实例资源 | 4×V100(32G 全占) + host：29.5G 锁内存 PLE + 64G parked cache；装载 ~90s，就绪行 `vit: the tower is up on 4 cards` |
| 卡组 | 必须同 PIX 组：`nvidia-smi topo -m` 看，惯例 GPU0-3 一组 / 4-7 一组（跨 switch = prefill 毒药，1028 三定律） |

## 步骤

### 0. 侦察（三条命令定型，勿多查）
```bash
ssh p40-23 'curl -s --noproxy "*" http://127.0.0.1:8090/health'      # 现役池
for h in atlas1 atlas2 atlas8; do echo "## $h"; timeout 25 ssh $h 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader|tr "\n" " "; free -g|sed -n 2p; df -h /data|tail -1'; done
ssh <新机> 'ss -tlnp | grep -E ":(809[0-9])"; ls -d /data/sfnx* /data/strata-local/gguf /data/strata/Strata-data/mtp/rt /data/sfnx-py/.venv 2>/dev/null'
```

### 1. 搬生产树（新机建**新目录**，绝不覆盖旧树）
两 atlas 机互不能 ssh（host key/密钥没配），必须经本机管道：
```bash
ssh atlas2 'tar -C /data/sfnx -cf - --exclude=models --exclude=.venv --exclude=pd-dumps .' \
 | ssh <新机> 'mkdir -p /data/sfnx-prod && tar -C /data/sfnx-prod -xf -'
ssh <新机> md5sum /data/sfnx-prod/bin/sfnx    # 必须 == 蓝本 md5
```

### 2. 权重/venv/mtp 全部软链（零拷贝）
```bash
ssh <新机> 'ln -sfn /data/sfnx-py/.venv /data/sfnx-prod/.venv
 mkdir -p /data/sfnx-prod/models/UD-Q4_K_XL
 for i in 1 2 3 4; do ln -sfn /data/strata-local/gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000${i}-of-00004.gguf /data/sfnx-prod/models/UD-Q4_K_XL/; done
 ln -sfn /data/strata-local/gguf/mmproj-BF16.gguf /data/sfnx-prod/models/mmproj-F16.gguf'
# 逐条 readlink -f 验证可解析；md5 对 mmproj（新机 vs atlas2）必须一致，不同就停
```
`data/mtp-rt` 是指向 `/data/strata/Strata-data/mtp/rt` 的绝对软链——新机若该路径不存在，先把 rt 目录（~2.2G）落到新机再改链。

### 3. serve 配置两处必改（其余照抄生产）
```bash
ssh <新机> 'cd /data/sfnx-prod/serve && for f in sfnx-8093 sfnx-8094; do
  sed -i "s#/data/sfnx/data/expert-routes.bin#/data/sfnx-prod/data/expert-routes.bin#" $f.json; done'
# lib_dirs 必须指到新机实际 cuda（生产机 /usr/local/cuda→12.9，atlas1→12.4！照抄会拿错 cudart）
ssh <新机> 'ls -d /usr/local/cuda-*'
```
用 python 改 `lib_dirs=["/usr/local/cuda-12.9/bin","/usr/local/cuda-12.9/lib64"]`（按上一步结果选），并核 `gpu` 字段分组正确、`port` 未被占。校验：`.venv/bin/python -m json.tool serve/sfnx-8093.json`；依赖体检：
```bash
ssh <新机> 'LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 ldd /data/sfnx-prod/bin/sfnx | grep -i "not found"'   # 必须空
```

### 4. 起实例（setsid 脱离，别在前台跑）
```bash
ssh <新机> 'cd /data/sfnx-prod && setsid nohup .venv/bin/python -m front.server --engine sfnx --config serve/sfnx-8093.json >> serve/start-8093.out 2>&1 </dev/null &'
# 8094 同。**不要**手工传 CUDA_VISIBLE_DEVICES——front.server 按 config 的 gpu 自设 + CUDA_DEVICE_ORDER=PCI_BUS_ID
ssh <新机> 'tail -f /data/sfnx-prod/serve/sfnx-8093.log'   # 等 "the tower is up"，~90s
```

### 5. 重启自愈单元（enable 但**不 start**——现役 setsid 进程占着卡）
`/etc/systemd/system/sfnx-<port>.service`：`User=atlas`、`WorkingDirectory=<树目录>`、`ExecStart=<树>/.venv/bin/python -m front.server --engine sfnx --config serve/<配置名>.json`、`LimitMEMLOCK=infinity`（装载要锁 29.5G）、`Restart=on-failure`、`StartLimitIntervalSec=600`/`StartLimitBurst=3`、`After=network-online.target`。`systemctl enable` 后确认 `is-active` = inactive（正常，开机才接管）；`sudo systemd-analyze verify <unit>` 只看自己那行（snapd 的 RestartMode 警告是系统噪音）。
现状：atlas1(`/data/sfnx-prod`) 与 atlas8(`/data/sfnx`) 四台已有单元；**atlas2 两台仍裸跑**。建单元/改配置都不会碰现役进程（config 只在启动读一次），但改完务必核 `ps -o etime` 引擎 pid 没变短。

### 6. 本机中继加端口（1110 起 `~/relay_sfnx.py` = **全池唯一中继**，覆盖 18093-18098）
在 `PORTS` 表加 `180xx: ("<新机IP>", 8093/8094)` 再 restart。restart 会掐断**所有**正在流的会话（老 relay_a2.py 已退役；它的冷备 unit 改名 `relay-a2.service.retired-1110`，别再 enable，会撞端口——mask 对已存在文件无效只能挪走）。安静窗口纪律：
```bash
ss -tn state established '( sport = :18093 or sport = :18094 or sport = :18095 or sport = :18096 or sport = :18097 or sport = :18098 )' | tail -n +2 | wc -l   # 必须 0
python3 -m py_compile ~/relay_sfnx.py && systemctl --user restart relay-sfnx.service
ss -tln | grep -oE ':1809[3-8]' | sort          # 六个都在
for pt in 18093 18094 18095 18096 18097 18098; do printf "$pt "; ssh p40-23 "curl -s -m 6 --noproxy '*' -o /dev/null -w '%{http_code}\n' http://<FLEET_HOST>:$pt/v1/models"; done   # 全 200
```
单元 = `relay-sfnx.service`（enabled + WantedBy=default.target，本机 linger=yes）→ 本机重启全池自愈。若从手起的旧进程接管：老进程持端口时 SO_REUSEPORT 也救不了，只能 `kill <老pid>; systemctl --user restart relay-sfnx` 连着发（~1s 空窗）。

### 7. sfnx-gw 加后端 + 一次性 restart（配置只在启动时读，无热加载）
```bash
ssh p40-23 'cd /home/xh/sfnx-gw && cp sfnx-gw.json sfnx-gw.json.bak-<日期>-<N>pool'
# 写新 sfnx-gw.json：backends 数组，每项 {"name":"<机>-<port>","url":"http://<FLEET_HOST>:180xx","pool":true}
# 下刀前看在途：
ss -tn state established '( sport = :18093 or sport = :18094 )' | tail -n +2 | wc -l   # 争取 0
ssh p40-23 'sudo systemctl restart sfnx-gw && sleep 4 && curl -s --noproxy "*" http://127.0.0.1:8090/health'
```

### 8. 24.23 容量与分组（改完等 300s 刷新，**不重启主网关**）
`/home/xh/gateway-config.json`（先 cp 备份）：`simple_backends[sfnx-gw].max_concurrent` = 实例数 × batch(6)；被收编的机名从其它 `session_groups.backends` 里摘掉（防双线抢卡）。

## Pitfalls
- **本机 `relay_sfnx.py` 是全池唯一通路**：它一挂六台同时 DOWN（gw 探测 15s×2 连败标 DOWN → 503 "No healthy sFNX backends"）。已 enabled+linger 做到重启自愈，但 restart/崩溃窗口内全池无服务——加端口/改代码前先按步骤 6 查六端口在途=0。
- **24.23 没有到 <FLEET_HOST>/24 的路由**：不加 `--noproxy` 时 curl 会走 `/etc/environment` 的全局代理（<FLEET_HOST>:1085）返回"看起来正常"的 200，骗你以为能直连。判别法：`curl --noproxy '*'` = 000 → 无真路由，**后端 URL 必须写本机中继地址**。
- **atlas 机 shell 也带 http_proxy**（atlas2→<FLEET_HOST>:1085）：在机上 `curl localhost:8093` 得到 **HTTP=000 假死**，加 `--noproxy '*'` 才通；别据此判实例死了。
- **sfnx-gw restart 清 `_sticky`**：被重新哈希的会话付一次冷 prefill；且 restart 瞬间健康表为空，会有极短 503 窗口。
- **`gpu` 字段 ≠ 实际卡位**：front.server 先用 `gpu` 推导 `CUDA_VISIBLE_DEVICES`，随后 config 的 `env` 块**覆盖**它——所以有的机（atlas8 a8-8094）gpu 写错却靠 env 一直跑对，一旦有人删 env 或照 gpu 重启就把两台挤同组卡爆显存。收编/建单元前一律以进程实况为准：`tr '\0' '\n' < /proc/<engine pid>/environ | grep CUDA_VISIBLE`，两处对齐后再动手（改 config 只影响下次重启，安全）。
- **引擎日志路径撞车**：atlas8 曾两台共用 `serve/sfnx-8093.log`（a8-8094.json 的 `log` 字段没改过来）→ 排障串味。收编时顺手核对 `log` 与端口一一对应。
- **同名旧树绝不覆盖**：atlas1/atlas8 上的 `/data/sfnx` 是另一代树（bin md5 不同），一律新目录部署。
- **验收别用小编号 max_tokens**：thinking 档下 `max_tokens=64~80` 会被 reasoning 吃满 → HTTP 200 + `content=null` 假空返；正确性断言用 `17*23=391`，max_tokens ≥512。
- 非流式请求零字节等到引擎生成完，`READ_TIMEOUT_NONSSE=120`（在 sfnx-gw 内，不在 8300/nginx）；超 120s 的活要么走流式，要么改该常量。
- cuda 版本：`/usr/local/cuda` 各机指向不同（atlas1=12.4 / atlas2=12.9），只认 `ldd` 结果不认惯例。

## 验证
```bash
ssh p40-23 'curl -s --noproxy "*" http://127.0.0.1:8090/health'          # backends=N, up=N
bash ~/sfnx_pool_check.sh 12      # 经 8300 打唯一首消息探针，统计 gw 审计日志 be= 分布（应铺满全部后端）
# 端到端正确性（走真实链路，非机器本地）：
ssh p40-23 'curl -s -m 180 --noproxy "*" http://<FLEET_HOST>:18095/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-flash-next\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with just the number.\"}],\"max_tokens\":512,\"stream\":false}"'   # content=391
# 流式：curl -sN ... -d '{"stream":true,...}' | grep -c '^data:'   （>0 即通）
# 吞吐对拍：同一条 900-token 请求分别打新机与生产机，tok/s 应持平（1110 实测新机 153.5 vs 生产带载 138）
# 错误面：ssh p40-23 'sudo journalctl -u sfnx-gw --since <切换时刻> --no-pager | grep -ciE "error|down|traceback"' → 0
#          grep -c "upstream fail" ~/relay_sfnx.log → 0
```
收尾：更新 memory `sfnx-6pool-atlas1-atlas8-1110`（或当期条目）的后端表/容量数；**别去 grep 24.23 网关 journal 的关键词**——它记录全量上下文载荷，一条能灌爆上下文。
