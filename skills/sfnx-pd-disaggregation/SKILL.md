---
name: sfnx-pd-disaggregation
description: "sFNX PD 分离与跨机搬运：PREAD/PDOFFER/PDUMP/PLOAD、pd_peer 时激活。"
version: 1
author: ang-agent
license: MIT
metadata:
  hermes:
    tags: [sfnx, pd-disaggregation, prefill, kv-transfer]
    related_skills: [sfnx-sm61-p40-port, sfnx-v100-lab, parallel-process-patterns]
---

# sFNX PD 分离（prefill hand-off）

## When to Use

- 部署/测试/排障 sFNX 跨机 prefill hand-off（V100 做 P、P40 做 D）
- 引擎行协议 PREAD/PDOFFER/PDUMP/PLOAD 或 front /pd/* 端点相关开发；YaRN(1M) P 与标准 rope P40 D 互通的双 rope 补丁
- PLOAD 报 "a state saved with another layout"、跨树 dump 不兼容、或 hand-off 回退本地排查
- 分工红线：本流程只管引擎与 PD 链路；网关机（gateway.py / gateway-config.json）的修改归网关侧智能体，割接完成后交端点拓扑即可，勿顺手改网关配置

P 侧（V100，prefill 快）把 prompt 读上 lane → dump 成文件 → D 侧（P40，decode 强）拉取载入 → 本地 GEN 全量复用只 decode。任何一步失败 = 本地生成（正确性永不受损，只慢）。

## 架构（三层）

1. **引擎行协议四命令**（tools/sfnx.cpp 补丁，P/D 用同一二进制）：
   - `PREAD <ids>`：prompt 读上 lane 不生成（`GEN max_tokens=1` 不可替代——生成的 1 token 进 history 污染 usable() 前缀匹配）；应答 `PREAD OK <slot> <tokens> <reused> <ms>`，内部 best_slot+make_room
   - `PDUMP <slot> <path>`：lane → 文件（tokens+每卡 state bytes 含 KV），lane 同时卸载；应答 `PDUMP OK <tokens> <bytes>`
   - `PLOAD <slot> <path>`：文件 → slot 的 lane（先 park 现有会话再 reload）；应答 `PLOAD OK <n>`；池压安全：reload→ensure_pages 自带池短回调+Refused 异常
   - `PDOFFER <ids> <path>`：PREAD+PDUMP 合一原子单行（读用标准 rope 包裹、dump 头 tag=2，见「双 rope 补丁」节）；应答 `PDOFFER OK <slot> <tokens> <reused> <bytes>`
   - 应答自带命令前缀（VENC 同款）防被请求 reader 吃掉；ERR 有两种形态 `<CMD> ERR <why>` / `ERR <CMD>: <why>`——front 泵两种都要路由到 pd_q
2. **front P 侧端点**（config `pd_role:"P"`）：POST /pd/offer（ids→引擎 PDOFFER，读+dump 一行原子完成，主路径）、GET /pd/blob/<name>（文件流，取后即删）；/pd/pread+/pd/dump 两轮旧端点保留但已被 offer 取代。pd_q 泵 startswith 元组须含 `"PDOFFER "`/`"ERR PDOFFER"`
3. **front D 侧编排**（config `pd_role:"D"`+`pd_peer`）：首段 generate 前决策（≥PD_MIN_TOKENS 且本地 prefill 慢于 pd_peer_prefill）→ pd_handoff（单轮 HTTP /pd/offer→blob 拉取到 pd_dir）→ 本地 `PLOAD 0 <file>` → GEN。仅 solo 路径才 hand-off（batch slots 全闲且无人排队）——PLOAD 固定载 slot 0 的 lane，batch 忙时 GEN 由引擎 best_slot 落到别的 lane 对不上；带图请求（emb 非空）也不 hand-off

关键语义：P 侧读**全量 prompt ids**（front/hand-off 不要自己切末 token）——引擎 read(M) 自身挂起末 token 为 decode 首行、环内落 M-1，恰是 GEN 的 "history 少 1" 精确前缀分支：D 侧 PLOAD 后 GEN 纯 resume（RESUME=M-1、零续读行）、prefill 0ms。P 若只喂 N-1（环 N-2），D 的 GEN 须一次续读 2 行——该路径与全新计算不逐 token 一致（实测第 3 token 起发散；KV dump 字节对比全等已排除 rope 数学，是多行续读路径本身不精确）。勿为此改 session.cpp（曾加续写分支补丁是地雷，已回滚）。

## 为什么不用 vLLM/Mooncake 那套 PD connector 机制（被问 "sfnx 为何不用现成 PD 框架 / P 也当 D" 时）

- vLLM 系 KVConnector（Mooncake/AscendStore 插件）依赖**分页 KV 的块级寻址**（BlockManager / remote_block_ids，按块增量传输）；sFNX 的 KV 是整条 lane 的序列化 state，布局被 NCARD/KV 量化/模型/FNX_PF_ROWS/rope 五重绑定（跨机兼容四条件）——没有 block 粒度就没有增量传输，只能整段 dump/推流（~132KiB/token），所以传输自写成引擎行协议 + front 编排，connector 插件形态不可用
- 自适应均衡分层实现（910B qfn PD 把同类逻辑做在 proxy 层——connector 现成只需路由；sFNX 必须自己搬 KV，均衡 = sfnx-gw 对 D 池 least-loaded + D 侧 front 自决 hand-off：空闲→本地 / <40K 且 P 忙→本地 / 其余 hand-off / 任一步失败→本地重算，正确性永不受损）
- **P 刻意不进服务池（"P 也当 D"被设计排除）**：P 的 prefill 速率是 hand-off 决策阈值（pd_peer_prefill）的标定基线，P 混跑 decode 拖垮 prefill 速率 → hand-off 经济学塌掉；lane 资源竞争另有事故史（并发 PREAD 互驱逐出 0-token 假 dump）。pd_role 由 front 启动期 config 定死，无双角色模式

## 跨机兼容四条件（缺一 = PLOAD ERR "another layout" 回退本地）

1. 同 NCARD（双侧 make FNX_NCARD=4）
2. 同 KV 量化（FNX_KV8 缺省 1=int8；两侧都别设 0）
3. 同模型（UD-Q4_K_XL）
4. **同 qsa_ring 行数**：`qsa_ring = max(2048, 2*FNX_PF_ROWS)`，state_pieces 布局逐字节依赖它。sm_61 移植树 MAXT=16、V100 原树 MAXT=18——`FNX_PF_ROWS=4096` 在 V100 树启动即崩（%18≠0），且两侧不同值则 dump 字节数不匹配。**统一 FNX_PF_ROWS=4176**（LCM(16,18)=144 的倍数，两树都整除、ring 同为 8352）

## 双 rope 补丁（YaRN P ↔ 标准 rope D 互操作）

YaRN 是按旋转 pair 的频率混合（ramp 挂 pair-index、与位置无关）——每个位置的旋转都被改，"262K 内标准、之外 yarn"数学上不存在；D 侧把 yarn KV 转标准（旋转校正+int8 roundtrip+重池化）也太脏。唯一干净解：**P 侧为 hand-off 的读单独用标准 rope 算 KV**（Rope 是 attn_prep_append/qsa_index 的 kernel 参数，可 per-call 换），dump 与标准 rope D 逐字节兼容，P 自身 GEN 仍走 yarn（1M 直服不受影响）：

- Engine：`rope_std_`（同 base、scaled=false、mscale=1）+ 开关 `rope_std_next_`（public `rope_std_next(bool)`/`rope_scaled()`）；`rope_cur()` 按开关二选一
- **调用点全树清点**：rope_of 不止 engine.cu 6 处（decode/draft/qsa），**prefill.cu 还有 4 处**（prefill_draft+prefill_pass）——PREAD 的 KV 正是在 prefill.cu 算的，漏改则 std 开关只管 decode、dump 出的仍是 yarn KV 且无守卫拦截 → D 侧静默错误答案。改引擎前 `grep -rn rope_of src/` 清点全部调用点
- graph_key 加 std 位：窗口 graph replay 参数 bake，两套 rope 的 graph 必须各自缓存，混用会 replay 进错频率
- PDOFFER 读包裹 std 开关，catch 先关再 rethrow（异常安全防开关悬挂）
- dump 头 word[3]=rope 域 tag：1=引擎自身 rope（PDUMP）、2=标准 rope（PDOFFER）；PLOAD 守卫：tag=2 且 `rope_scaled()` 则拒（"dumped on the plain rotary, this engine runs YaRN"）。`--rope-scale 1.0` 经 choose_rope（env_S=0）→ scaled=false：1.0 引擎即"标准"侧，守卫语义正确
- 语法闸门目标按源文件：engine.cu→`make FNX_NCARD=4 build/engine.o`、prefill.cu→`build/prefill.o`——`build/kernels_p0.o` 只依赖 kernels.cu，改 engine.cu/prefill.cu 不触发它重编（报 up to date ≠ 编译过）

## 验证流程（A / A2 / E2E / pd4）

- Phase A（同机跨进程）：A 生成→PDUMP→B PLOAD→续写 vs C 全新对照。**必须 temperature=0**——缺省采样是随机的，B≠C 未必是 bug（曾误判）。PASS = D==C 逐 token 相等 + D RESUME=n-1
- Phase A2（PREAD 版）：P 只 PREAD 不生成，流程更干净
- Phase pd4（跨 rope，双 rope 补丁验证）：YaRN P（--rope-scale 3.82）PDOFFER → P 自 PLOAD 同文件须被守卫拒（plain rotary 错）→ 标准 D（1.0）PLOAD+GEN → C 全新对照，greedy D==C 逐 token。**测试引擎串行占卡**：一机一引擎（每实例 ~23G 贴满 24G 卡，并行 spawn 两引擎 = 加载期 cudaMalloc OOM）；kill 后轮询 nvidia-smi 全卡 <500MiB 再起下一个。断言按引擎读语义写（读 M 个 ids 进环 M-1，末 token 挂起为 decode 首行）：PDOFFER **全量 N 个** → `PLOAD OK <N-1>`、GEN resume=N-1 零续读——按自己算术写 assert 会误报失败
- **KV 数学等价用 dump 字节对比定案**（比 D==C token 对照便宜且隔离变量）：同 build 同 ids 出两份 dump——yarn 引擎 PDOFFER（std-rope）vs 标准 rope 引擎 PREAD+PDUMP——`cmp <(tail -c +17 a.bin) <(tail -c +17 b.bin)`（跳 16B 头；word[3] tag 2 vs 1 是唯一合法差异）逐字节相等 ⟺ std-rope 数学精确。D==C 对照同时考 KV 数学与续读路径两件事，FAIL 时先字节对比二分归因
- 跨机 E2E：D 侧发真实 chat 请求；front 日志出现 `PD hand-off: N tokens' K/V from <peer>`；用 17*23=391 类数学题验答案。回退也 PASS（答案对，只是无提速）。探针 `scripts/pd_e2e_probe.py`（D 机上直接跑：自动取模型名、长 prompt 内嵌 17*23、报首 token 延迟与答案），割接后复验即用。复验直接用现成探针，勿现场内联写 SSE 解析：末行 `data: [DONE]` 非法 JSON 会让逐行 json.loads 崩、thinking 模型首 token 藏在 reasoning_content 里（只盯 content 会误判无输出）——现写必踩这两个坑

验证结果（实测）：Phase A2（26 同机）PASS——PREAD 2000 tok 4.5s → PDUMP 1999 tok 253MiB/510ms → PLOAD → GEN RESUME=1999、prefill 0ms、decode 满速 66.5；greedy D==C 逐 token 一致。跨机 E2E（26 D↔atlas11 P，修 layout 后）PASS——681 tok（232MiB）hand-off，TTFT 6.45s→2.65s（2.4×），答案 391 对；修复前 layout 拒绝时优雅回退本地，答案照样对（正确性永不受损）

双 rope 补丁（实测）：YaRN P PDOFFER 1999 tok（std-rope KV）5.1s/267MB、守卫自拒实证；**dump 字节对比**：yarn-std dump 与标准 rope 引擎 dump 头后逐字节全等——std-rope 数学证毕。front 曾按 ids[:-1] 切送（环 1998、D 一次续读 2 行）→ D==C 第 3 token 起发散；归因后 front 改送全量 ids（pd3 同款纯 resume 形态）；生产形态终验 PASS：yarn P PDOFFER 全量 ids → 标准 D PLOAD → GEN 纯 resume（prefill 0ms、对照全新计算 4.8s），greedy D==C 逐 token 一致。

生产定版：27 双实例（A:8093/B:8094，rope 二进制=26 同件，pd_peer→atlas11:8095，FNX_PF_ROWS=4176）E2E 双 PASS：TTFT 2.28s、681 tok hand-off 双侧实证；22→23 网关→p40 组→27 全链 3.33s、391 对。8095 割接已执行：P 转 1M（rope 3.82、n_ctx 1001390），27-A/B 双 D 经它 hand-off 复验双 PASS（TTFT 2.11/2.65s、1326 tok/242MiB）；8093 树是他人实验，永不触碰。**26 第三 D 节点已上线**（卡 2,3,4,5，config serve/sfnx-26-8093.json，PD 键与 27 同款，pd_peer→atlas11:8095）。**atlas8 第四/五 D 节点已上线**（V100 8 卡，树从 atlas2 /data/sfnx 整树同步，双实例 8093/8094 卡 0-3/4-7，标准 rope 262144，**纯本地 prefill**——atlas8 到 atlas11:8095 TCP 不通（76 网段→190.13 路由墙），config 无 PD 键）。23 网关 -p40 组现役 = 27-a/27-b/26-a + atlas8-a/atlas8-b 五成员（两集群混编）。现役拓扑：PD 链 = 8095(1M P) ← 27 A/B + 26 (D)；atlas8 = 无 P 链直服；聚合层 sfnx-gw :8090（atlas11，serve/sfnx-gw.json）：4 池 = 本机 8093/8095 + atlas2:8093/8094——池成员须 ctx 同质（4 实例现已全 1M）；atlas2 两实例是 sfnx-gw 直服池成员，仍非 pd_role P、不服务 P40 hand-off。跨机寻址、IB 双轨与中继、BMC 带外排障、重启预检与卡引导处置见 references/v100-ib-topology.md。P 侧（atlas11）整机故障时的保命组合（用户明令时越过分工红线执行）：-p40/别名组指 27 双 D + 26 + atlas8 双 D（本地回退实测可用，长文 TTFT 6s 上下、答案对）+ 1M 组经本机中继 18093/18094 指 atlas2 双 1M（23 与 atlas2 互不可达时本机是唯一桥）；步骤与验证陷阱见 references/v100-ib-topology.md「24.23 应急重指」+「本机中转恢复 1M」节。27 树无 llama.cpp 子树（本编不了）——换二进制从 26 拷，scp 直写运行中文件报 Failure，走临时名+mv 原子替换

## 部署配方

- P 实例：atlas11 8095（sfnx32 树，卡 4-7）或 8093（sfnx 树）皆可——双 rope 补丁后 P 直跑 1M（`--max-context 1001390 --rope-scale 3.82` + env `FNX_YARN_ORIG_CTX=262144`），hand-off KV 由 PDOFFER 用标准 rope 算、与 P40 D 兼容；P 的 1M 直服与 PD 兼容并存（补丁前才被迫 1.0/262144 对齐）。env `FNX_MTP_EXPERTS=q4_0` + `FNX_PF_ROWS=4176`，config 加 `pd_role:"P"` + `pd_dir`
- D 实例：P40 侧 config 加 `pd_role:"D"` + `pd_peer` + `pd_peer_prefill`（P40 ~260 vs V100 ~900 tok/s，设 800）
- 独立 1M 直服实例（不进 P 池）：种子树从已验明身份的 rope 树全树 rsync（排除 build/ 与模型权重）→ 模型本地化 → 二进制全量重编或从构建机拷；config 无 PD 键、--max-context 1001390 --rope-scale 3.82、env FNX_PF_ROWS=4176 + FNX_EXPERT_PROFILE 指本地树路径；双实例 gpu 0-3/4-7 各一份 config，启停一律脚本文件
- 新节点直服实例（无 P 链，如 atlas8）：从同架构现役树（V100 用 atlas2 /data/sfnx）机器对机器 rsync → **补传树内软链真身**（见 Pitfalls）→ config 删 PD 五键（pd_role/pd_peer/pd_peer_prefill/pd_peer_timeout/pd_dir）+ env CUDA_VISIBLE_DEVICES 按新机验证过的卡组改 → 双实例端口 8093/8094 → 冷载后直连 /v1/messages 冒烟（17*23=391）。挂网关侧见 p40-gateway-ops 的 session group 扩容配方
- 27 单实例滚动更新：引擎是 front 的子进程——杀 front（pgrep -f "sfnx-<port>.json"）后 `pgrep -P <front_pid>` 拿引擎 PID 精确杀，另一 port 实例不受影响；mv 换二进制后单起该 port 的 front
- 27 双实例 D 上量：两实例 config 同加 PD 四键 + `env.FNX_PF_ROWS=4176`（layout 对齐 P 侧）；换新二进制（含 GDN 分块 kernel）同时删 env `FNX_PF_GDN_CHUNK` 拐杖；二进制从 26 拷（见 Pitfalls），重启脚本杀 front+engine→双 nohup，冷载 36+ 分钟挂长看护
- 建配置用 python3 json load/dump 脚本文件——ssh heredoc 内联 python 的转义会被 shell 吃
- 重启用脚本文件（见 Pitfalls 第 1 条），启动命令同款：`cd <树> && (STRATA_ENGINE_READY_S=0 nohup .venv/bin/python -m front.server --engine sfnx --config serve/<cfg>.json > serve/front-<port>.log 2>&1 &)`
- 双侧 ready 判据：curl --noproxy "*" /v1/models=200 + front 日志含 `PD: prefill side`/`PD: decode side`
- 复用/清空一台机器的 GPU 前枚举自启源，不只看进程：`systemctl list-unit-files --state=enabled` + `systemctl --failed` + `crontab -l` + rc.local——杀进程不防复活，systemd 单元须 `disable --now`（常驻实验服务、ollama 类都做过自启源；清理后 nvidia-smi 全卡归零再验）

## Pitfalls

- **pgrep 自杀（[s] 标记失效场景）**：kill 模式若是启动命令的子串（如 config 文件名 sfnx-8095-pd.json 含 "sfnx-8095"），`[s]` 标记防不住——外层 bash -c 的 cmdline 里有全串，kill 杀掉自身导致 nohup 从未执行（表现为日志 mtime 不更新）。把 kill+start 逻辑放进脚本文件（模式不在任何 cmdline）再 `bash /tmp/xxx.sh`
- front 的 Service 方法（run()）内引用 `self`，不是 handler 闭包的 `svc`（NameError：请求 500）
- front/server.py 新增 config 键必须同步登记 `CONFIG_KEYS` frozenset——check_config_keys 对未登记键直接 SystemExit（unknown config key），front 起不来
- 26/27 shell 带 http_proxy：front 拉 peer 一律用 `urllib.request.build_opener(ProxyHandler({}))` 建的全局 opener，urlopen 会走代理连不通内网 peer
- 带图片的请求（emb 非空）绝不 hand-off：视觉塔 KV 由 GENI 生成，PLOAD 的文本 KV 复用会错
- atlas11 树 `make` 全量 rc=2（llama.cpp/.git 缺失检查失败）但 `make FNX_NCARD=4 bin/sfnx` 单目标不依赖 llama.cpp——tools/sfnx.cpp 改动后单目标链接即可（~1 分钟）
- 27 树无 llama.cpp 子树（tar 传输排除）不能本地编译：27 换引擎二进制 = 26 编好后 scp 临时名再 mv 原子替换（直写 bin/sfnx 报 ETXTBSY；27 双实例冷载 ~36 分钟）
- **跨机传文件：机器互通（ssh 可达）就机器→机器直传，绝不经本机工作站中转**（用户纠正）——26↔27 已配双向 id_rsa key，26→atlas11 用 `~/.ssh/strata_a11`，26 上直接 `scp bin/sfnx xh@<FLEET_HOST>:...`；两机互不可达时经远端中继机两跳（仍是机器对机器，不经本机工作站）。例外仅限：两端互不可达且用户明令时，服务面可走本机中转过渡（见 references/v100-ib-topology.md「本机中转恢复 1M」节）
- **CUDA_VISIBLE_DEVICES 组合不能照搬他机**：sfnx 引擎 ring init 要求组内全卡 peer access，同一台 P40 机器上不同 4 卡组合的 P2P 拓扑不同——照抄 27 的 `0,1,2,3` 到 26 上引擎加载完即崩（`peer access is not supported between these two devices`）。每台机器的设备组是验证过的机器态事实（26 = `2,3,4,5`），新机/换卡组起引擎前先用原机现役 config 的设备组，别按序号美观选卡
- 装 authorized_keys 勿用 shell 变量手工拼长公钥行——静默错一字符=指纹不符 Permission denied 且难查；用已通的机器间通道原文搬运（`ssh <peer> "cat ~/.ssh/id_rsa.pub" >> authorized_keys`），`ssh-keygen -lf` 验指纹
- 看护循环里的远端命令必须整体包进同一条 ssh——循环体是本地 shell，裸 `python3 /tmp/x.py` 会跑在本机（Connection refused）
- 单引擎冷加载 ~6 分钟（26 P40 / atlas11 V100 同量级）；双侧 E2E 先等双双 ready
- PDUMP ≈132KiB/token（2000 tok=253MiB，dump 0.5s）；P40↔atlas11 10GbE 实测 9.2Gbps，253MiB 传 0.23s——网络非瓶颈，PREAD 本身（~4.5s/2000tok）才是 hand-off 主要成本
- 提速边界：PD 只救 TTFT（2000 tok：本地 prefill ~4.5s → hand-off 后 ~0s+传输 0.2s），decode 速度不变（P40 满速 61-66 tok/s 保留）
- **僵尸拼装二进制（ABI 错配）**：改 engine.hpp（Engine 类布局）后凡引用 Engine 的 .o（tool_sfnx.o/session.o 等）都必须重编——单目标语法闸门只验改动的源文件，拿它当发布闸门链接出混编二进制：加载正常、首个 GEN 信号死（front 日志 exit code None、dmesg segfault at 0），表象酷似运行时 bug。发布件必须全量 `make FNX_NCARD=<N> bin/sfnx`；换机部署前核对 build/ 各 .o mtime 同代
- **FNX_DIR 编译期烤入**：二进制携带构建树路径（nvcc -D），换机部署后该路径不存在 → expert-routes fopen 失败被误报成 `not 49152 tallies`——config env `FNX_EXPERT_PROFILE` 显式指本地树路径根治。克隆老 config 还会缺 `FNX_PF_ROWS` 键（V100 树 MAXT=18、默认 4096 启动即崩）→ 搬运二进制的 config 必补 env 4176
- **树同步后文件清单 diff**：rsync 退出码干净 ≠ 树完整（多跳中继曾静默丢 1.4G experts 件与 tokenizer）；`find <树> -type f -printf '%s %p\n' | sort` 两侧落盘 diff，缺件在引擎启动时报成不相干错误（tallies/tokenizer 类），按清单补推再起
- **树内软链指向树外绝对路径，rsync 只传断链**：`data/mtp-rt -> /data/strata/Strata-data/mtp/rt`、`.venv -> /data/sfnx-py/.venv` 这类软链 rsync 原样复制链接本身，目标端真身不存在 → 引擎启动报 `mtp: cannot open data/mtp-rt/dense.bin`（错误表象像缺件，真因是链接目标没传）。树同步后 `find <树> -type l` 清点全部软链，逐个把 target 真身按原绝对路径补传（含父目录授权）
- **目标机 /data root 属主**：rsync 首笔就报 `mkdir ... Permission denied (13)`——先 `sudo -n mkdir -p <路径> && sudo -n chown <user>: <路径>` 建好授权目录再传，不是网络问题
- **新节点到 P 网络隔离 → 删 PD 键走纯本地 prefill**：判法 `ip route get <P-ip>` + `bash -c 'echo > /dev/tcp/<P-ip>/8095'`；不通则 config 删 PD 五键，正确性无损（PD 只优化 TTFT），V100 本地 prefill ~900 tok/s 可接受。别为拉通网络在防火墙/路由上烧时间——PD 是加速项不是正确性依赖
- **front 重启端口竞态**：杀旧 front 立刻起新 front → 旧进程未死透、新 front "address already in use" 退出，端口随后空出但无人监听（探针连不上而进程表干净）。重启脚本杀完轮询 `ss -tln` 到端口消失再起；已踩空时纯启动（无 kill）即可恢复
- **同步源先验身份**：内网 IP 与主机易错配（相邻 IP 是不同机器），同步命令拷错 IP = 整树拉错 vintage 全程白跑；写机器间同步命令前先在目标上 `hostname` + 关键文件 mtime 探针核实身份
- **set -e 脚本里 pgrep 无匹配即自杀**：pgrep 零匹配退出码 1，set -e 让割接脚本静默死半途（服务已停、config 未改、端口无人监听）——可合法零匹配的 pgrep 一律 `|| true`，割接脚本每步打进度回显；静默半途死是最难排查的现场
- **Lustre 节点绝不 `rmmod mlx5_ib`**（/fs00 挂载走 o2ib 的机器，如 atlas1/11）：LNet 持模块引用，rmmod 卡 D 态不可杀、模块楔在 Going、该 HCA 的 IPoIB netdev 注销到重启才回来。修 IPoIB 用 bounce netdev；已楔死时现有设备照常工作（无害楔子），别再动模块，netplan 埋好 IP 择机重启自愈（详见 references/v100-ib-topology.md）

## 1110 生产定版(2P4D,当日上线——本节覆盖上文旧生产段)

**现役拓扑**:P = atlas0 双实例(8×V100-SXM2 NVLink,专职 prefill 不进池);D = atlas1+atlas8 四实例经 sfnx-gw(24.23:8090, 4 后端 least-loaded);atlas2 双实例下线(IB fabric-0 口硬件死)。D↔P 走 IPoIB fabric-0(<FLEET_HOST>/16,单流 1.35-1.74 GiB/s);上游链 CC→8300→24.23:8000→sfnx-gw 不变。

**三处架构修订(上文旧段已被取代)**:
1. **solo-only 已废**:pd_ready 改"存在空闲 lane"(PLOAD 选空闲槽+best_slot 前缀匹配落 lane,不再写死 slot 0);batch 忙时 hand-off 照常工作
2. **YaRN 通路**:池里 D 全 1M YaRN(rope 3.82),std-rope offer 通路(双 rope 补丁)用不上;生产走新端点 `/pd/pread_dump`(PREAD+PDUMP 单 HTTP 轮,引擎自身 rope,tag=1 dump)——**配 _PD_SEQ 锁**(两 D 并发 PREAD 互 park lane → 0-token 假 dump 实锤案)+0-token 守卫
3. **D 侧 hand-off 线程化+5s ping**:hand-off 本体 30-60s 静默(hand-off 在 opening 后 generator 前同步执行),中间代理 20s 超时掐流;线程跑+yield ("ping",None) 维持 SSE(front 内置通道)

**自适应均衡(无需人工切换)**:pd_peer_prefill 设 3601(=ACS 修复后单机 prefill 速率);空闲 D 本地读速 ≈P → 本地赢走本地(24s/90K);**D 忙时有效读速跌 → hand-off 自动接管**。PD 的价值=busy-D 干扰隔离,不是冷启动提速。

**当日三案排障指纹**(全在生产实弹撞出):
- 网关 relay `cannot read from timed out object`:SSE 超时→keepalive→复用死 response;修=清 socketio `_timeout_occurred` 毒旗(socket 超时后 conn.sock 被 None,别 settimeout)
- P 侧 0-token dump(1.7GiB 假成功):并发 PREAD 互驱逐 lane;修=_PD_SEQ 锁
- D 静默掐流:见上第 3 条
- 附:**config json 与 server.py 都是启动读一次,改任何都要重启 front**(1110 晚间双重实锤:①17:12 落盘的三修,16:47 起的 front 没吃到;②pd_dir 改 /dev/shm 后 dump 仍落老路径——勿再信任何"热载");atlas0=GLIBC 2.31 跑不了生产 binary(要 2.32+),必须本地编译(cuda 链恰好 12.9 一致)

## 1110 深夜 PDSTREAM 直推上线(v1 TCP,覆盖 pread_dump 文件路径)

**链路**:D front 单发监听器(bind pd_stream_host:0+nonce)→ P `/pd/pread_stream`(_PD_SEQ 包引擎 PDSTREAM)→ 引擎 PREAD+socket 直推 PDUMP 字节序列(D 侧落 /dev/shm 供 PLOAD,用完 unlink)。**P 侧零落盘**;文件路径(pread_dump)留作回落(超 260K 的大件/老引擎/流失败),三段兜底:流→文件→本地。

**引擎 PDSTREAM**(tools/sfnx.cpp ~90 行,md5 9d0cd687):pd_tcp_connect(非阻塞+poll,TCP_NODELAY,SO_SNDTIMEO 120s)→ 写 `NONCE <hex>\n` → 复用 PDUMP 全套(best_slot/make_room/rope_std 包夹)→ pd_write_all 逐字节同 PDUMP 序列 → shutdown(SHUT_WR) → `PDSTREAM OK <slot> <tokens> <reused> <bytes>`。

**front 三坑(全在 rig+生产实弹撞出)**:
1. **offer POST 必须 daemon 线程**:P 引擎在 POST 处理中就开始推,顺序 POST-then-accept 死锁在 socket 缓冲(指纹=TimeoutError+P 侧 PDSTREAM 已开始)
2. **pd_stream_host 必须显式写 D 的 IPoIB 地址**:空串 bind 0.0.0.0 后把 "0.0.0.0" 当回调地址送 P,连不上
3. **solo GEN 不占 slot_busy**:idle 判定必须 `len(live_reqs)-1` 计入 solo 请求(否则连串 180K 大件误判 idle 全走本地,把 D 饿到 3 tok/s)

**决策终版**(pd_hands_off):空闲(含 solo,数 live_reqs)→本地;<40K 且 P 忙→本地;其余 hand-off。pd_ready 只判"能否承载 PLOAD"(batch>0)——排队中不再拦 hand-off,道满由 PLOAD 拒绝兜底。流超时随 prompt 伸缩:`max(pd_peer_timeout, 120+len/1800)`。

完整字节格式/验证档案/备份链见 memory:sfnx-pdstream-direct-push-1110。

完整部署配方+地雷清单见 memory:sfnx-2p4d-pd-production-1110。
