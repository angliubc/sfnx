---
name: sfnx-sm61-p40-port
description: "在 P40(sm_61) 上部署或排障 sFNX 引擎：移植层、九关卡、验证。"
---

# sFNX on P40 (sm_61) 移植部署

生产参照 atlas11（V100 8×32G，/data/sfnx）。目标机 p40-26（xh@<FLEET_HOST>，密码查 <local credential ledger>，sudo -n NOPASSWD）。移植已完成并真 chat 验证。本 skill 同时是重启/排障手册。

## 现役部署快照

- **硬件真相**：26 是 6×P40 且**卡 0/1 孤岛对、卡 2/3/4/5 全互通**（各自 PLX 域，跨组无 P2P）→ 只能 4 卡跑
- 服务：`http://<FLEET_HOST>:8093/v1`（qwen3.8-flash-next, ctx 262144, MTP spec6, int8 KV）
- 树：`/data/strata/sfnx26/sfnx`（FNX_NCARD=4 构建；模型 models/UD-Q4_K_XL；mtp-rt 已实体拷贝非软链）
- config `serve/sfnx-8093.json` env：`FNX_MTP_EXPERTS=q4_0` + `CUDA_VISIBLE_DEVICES=2,3,4,5` + `FNX_ARENA_RESERVE_MIB=2300` + `FNX_PF_ROWS=4176`（GDN 分块已入二进制，FNX_PF_GDN_CHUNK 拐杖已删）。26 已转测试机，现跑 pd_role:"D" 测试实例指 atlas11:8095
- args 关键：`--max-context 262144 --rope-scale 1.0`（1M ctx 的 KV 池 4×24G 装不下）
- 无 systemd 单元，nohup 裸进程
- **p40-27（xh@<FLEET_HOST>，8×P40）双 4 卡实例**：{0,1,2,3} 与 {4,5,6,7} 两个 NUMA 组跨组无 P2P，FNX_NCARD=8 物理不可行 → `http://<FLEET_HOST>:8093/v1` + `:8094/v1` 双实例（原 strata-140 双实例已下线）；树在 /data/strata/sfnx26/sfnx（同名路径复用 26 的 venv/python 树），模型链 /strata/models/UD-Q4_K_XL、mtp-rt 链 /strata/mtp/rt；**PD 分离已上生产**（双实例 pd_role:"D" 指 atlas11:8095 P 实例，env 加 FNX_PF_ROWS=4176，详见 sfnx-pd-disaggregation skill）

## 启动 / 重启

```
cd /data/strata/sfnx26/sfnx
# 杀旧：按 PID kill front.server 与 bin/sfnx（勿 pkill python，网关拦截）
STRATA_ENGINE_READY_S=0 nohup .venv/bin/python -m front.server --engine sfnx --config serve/sfnx-8093.json > serve/front-8093.log 2>&1 &
```

加载 ~6 分钟（81G 权重 @0.23GB/s 机械盘）+ arena ~2 分钟，别提前判死。**27 双实例冷盘首载 36+ 分钟**（单块 HDD 双实例竞争 ~0.04GB/s/个，各读 94G 后走 page cache 才加速；双实例同起共享 page cache）。27 重启后双实例拉起：

```
cd /data/strata/sfnx26/sfnx; : > serve/sfnx-8093.log; : > serve/sfnx-8094.log
STRATA_ENGINE_READY_S=0 nohup .venv/bin/python -m front.server --engine sfnx --config serve/sfnx-8093.json > serve/front-8093.log 2>&1 &
cd /data/strata/sfnx26/sfnx; STRATA_ENGINE_READY_S=0 nohup .venv/bin/python -m front.server --engine sfnx --config serve/sfnx-8094.json > serve/front-8094.log 2>&1 &
```

27 实测（双实例 2026-11-04，1501 tok 稳态）：A/B decode 61.8/61.9 tok/s、draft 33/54（61%）、prefill 264/260 tok/s、答案正确——略优于 26 单实例，双实例总吞吐翻倍。**27 也需要 grub iommu=pt**（新机首启同样报 DMA-FQ 拒启，门卡 1 每台都要过）。

## 构建

```
cd /data/strata/sfnx26/sfnx && rm -rf build bin && make FNX_NCARD=4 -j48   # ~25 分钟，日志 /tmp/sfnx_buildX.log
```
Makefile 已改：sm_61、g++-13/gcc-13、`-G "Unix Makefiles"`（Ninja 坏）、CMAKE_CUDA_ARCHITECTURES 61。FNX_NCARD 必须与实际互通卡数一致。⚠️ **任何 make 调用（含 `make build/kernels_p0.o` 这类单目标语法闸门、甚至 `make -n`）都会把 build/ncard restamp 成该次 FNX_NCARD（缺省=3）**——后续不带 FNX_NCARD=4 的 make 会全量重编成 3 卡件，引擎报 "built for 3 cards (FNX_NCARD), 4 visible"。语法闸门也要带 `make FNX_NCARD=4 build/kernels_p0.o`。

## 迁移部署到新 P40 机（复用现成树）

选机第一步跑 scripts/p2p_test.cu：P40 服务器常按 NUMA 分两组 4 卡且跨组无 P2P，全互通组多大就 FNX_NCARD=几——8 卡机多半只能双 4 卡实例。模型/mtp 目标机已有则只传树：

1. 源机打包（~124M，**同路径落地免重建 venv**）：`.venv bin src tools front Makefile serve data/{draft_vocab_en.bin,expert-routes.bin,tokenizer}` + venv 指向的 python 树（.venv/bin/python 是 symlink，pyvenv.cfg home= 指向它，26 上是 sfnx-py/）
2. **跨机 scp 须本机中转**：远端再套一层 sshpass 引号必坏（秒回空输出）；拉回本机再推目标机
3. 资产映射：`models/UD-Q4_K_XL/` 子目录放模型软链 + `mmproj-F16.gguf` 链（config 路径硬编码该子目录，平铺报 `gguf: cannot open`）；`data/mtp-rt` 链本地 mtp 目录——**旧 mtp 目录常缺 `experts-q4_0.bin`（FNX_MTP_EXPERTS=q4_0 必需，1.4G），从现役机拷**
4. config 复制成双份（python json 改）：port/log/CUDA_VISIBLE_DEVICES 各自改——**port 字段抄自源 config，忘改=两实例抢同一端口**
5. 双实例同起（每段显式 cd，`;` 分隔不用 `&&`）；先 `free -g` 核实 RAM ≥ 2×130G（每实例锁 ~101G embedding）

## 移植层三件套（src/，母本 ~/.hermes-tg3/cache/scratch/）

1. `sm61_wmma.h`：nvcuda::wmma 16×16×16 f16→f32 模拟。**布局铁律：lane l 持 tile 行 l>>1、列 (l&1)*8+j**——load（matrix_a row / matrix_b col / matrix_b row 三个重载）与 store 全按此寻址，与 mma_sync 的 shuffle 约定一致。mma_sync：k 循环 5×shfl + 8×fma，全掩码。
2. `sm61_compat.h`：nanosleep（clock64 自旋）/ match_any_sync（32 路 shfl 扫）shim。
3. `patch_sm61.py`：锚点补丁——`__CUDA_ARCH__<700` 时 include sm61_wmma.h 替代 `<mma.h>`，sm_70+ 零改动。

**改 wmma 层必须先跑单元测试再全量重编**（全量 25 分钟，测试 5 分钟）：scripts/wmma_test.cu，A×B 两种布局 + store 两种布局 vs CPU 参考，要求 max abs err = 0。选机拓扑预检用 scripts/p2p_test.cu（`nvcc -arch=sm_61 -ccbin g++-13 -Wno-deprecated-gpu-targets`）。

## 启动九关卡（报错→根因→修法，按时序）

1. **IOMMU 拒启**（DMAR PTE Write / "need identity domains"）→ engine.cu 要求 GPU iommu_group/type=identity → grub `GRUB_CMDLINE_LINUX_DEFAULT` 加 `iommu=pt` + `update-grub`，**grep grub.cfg 数到 3 处才算生效**（首跑可能 0 处，重跑一次）
2. **peer access is not supported between these two devices** → 引擎要求全对全 P2P → 写 10 行 nvcc 小程序（`-arch=sm_61 -ccbin g++-13`）枚举 `cudaDeviceCanAccessPeer` 全 15 对（**别信 nvidia-smi topo 的 PIX/PHB 推断**）→ 选全互通卡组重编 FNX_NCARD=N + config env CUDA_VISIBLE_DEVICES
3. **vit cols OOM**（KV 池挤爆）→ KV 页池按 max_ctx 全量计，4×24G 摊 1M ctx 装不下 → `--max-context 262144`（模型原生长度）+ `--rope-scale 1.0`
4. **FNX_YARN_ORIG_CTX is set without FNX_ROPE_SCALE** → 降 rope-scale 后 YaRN env 残留冲突 → 删 env `FNX_YARN_ORIG_CTX`
5. **vit cols OOM 二现** → arena（conversation-cache）吃光余量，vit 塔在 arena 之后分配（每卡 628MiB）→ env `FNX_ARENA_RESERVE_MIB=2300`
6. **gdn_prompt: shared memory: invalid argument** → k_gdn_ch_rec 动态 smem 72.6KB > sm_61 48KB 硬上限 → env `FNX_PF_GDN_CHUNK=0`（走逐行 k_gdn_pf_rec）
7. **attn_prompt: illegal memory access** → 移植层 wmma 布局 bug（见铁律）；若布局已对，用 FNX_SYNC_LAUNCH 定位真凶 kernel
8. **window/step: an hc read's grid barrier timed out** → hc.cuh `H_BLOCKS=80` 按 V100 80 SM 定制（软件 grid barrier 需全块同时驻留，`__launch_bounds__(H_DT,1)`），P40 仅 30 SM → 手术：`H_RPG 96→352`（H_RG=1 → 20 块）+ `H_DT=H_DRI(96)×H_TPRD(8)=768` + down 段加 r0 行循环 + up 段加 H_UPT(=H_UPB/32) tile 循环；H_BAR_MOD 按块数自适应无需改。**任何带软件 grid barrier 的 kernel 移架构，先查块数 ≤ 目标 SM 数**
9. **HTTP 503 假象** → 26 机 http_proxy 劫持（响应头 `Proxy-Connection` 是指纹），127.0.0.1 也走代理 → `curl --noproxy "*"`

## Pitfalls

- **假 READY**：/v1/models 由 front 应答，引擎死了也能回。判活必须真 chat + `nvidia-smi` 显存双校验
- **STRATA_ENGINE_READY_S 是 front 自读 env**：写进 config 的 env 字段只传引擎进程，无效；须放启动 shell
- **ACS setpci 清零后 CUDA 报 cudaGetDeviceCount unknown error**：写寄存器打坏绑定态驱动。修 = 杀占用后 `rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia` → `modprobe nvidia nvidia_uvm`（**忘补 uvm 一样 unknown error**）。fuser -k /dev/nvidia* 会误杀自己的 ssh 会话（输出空 = 被杀）
- **CUDA_LAUNCH_BLOCKING=1** 会让 warmup 慢到 900s READY 超时；定位用 FNX_SYNC_LAUNCH（kernels.cu 的 ck_launch 已打 env 门控补丁，原始件 src/kernels.cu.bak-synclaunch）
- compute-sanitizer memcheck 慢 ~10×，加载后 25 分钟未达出错点——先用 SYNC_LAUNCH
- 引擎报错在 `serve/sfnx-8093.log` 尾（`sfnx: failed: <kernel>: <err>`）；front 超时/重启历史在 `serve/front-8093.log`
- "N cards ... not been validated on hardware" 是提示不是错误
- 看护轮询用 `pgrep -f "python -m [f]ront.server"` 防自匹配；间隔 60-90s
- **SSH 一条命令里先 kill 后 spawn 会自杀**：pgrep 模式匹配到本条 bash -c 命令行，刚起的进程被自己杀掉（现象=spawn 秒死且无报错）。拆两次调用：先收集 PID kill，确认后再单独 spawn；spawn 行永远不与 pgrep/kill 同命令
- **`&&` 链遇 `&` 截断**：`cd X && nohup … & cd X && nohup … &` 第二段实际在 home 目录执行（报 front-XXXX.log No such file）——每个后台段前显式 `cd`，用 `;` 分隔
- 远端 python 脚本同样吃 http_proxy：`urllib.request.build_opener(ProxyHandler({}))`，与 curl --noproxy 等价
- 停旧 strata 生产：按 PID kill `serve/server.py --engine strata` 与 `strata-vision`，显存落回 ~5.6G 基线才算停净（defunct 僵尸无害）

## 验证（部署后必做）

```
curl --noproxy "*" http://<FLEET_HOST>:8093/v1/models    # status=loaded
curl --noproxy "*" http://<FLEET_HOST>:8093/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","max_tokens":60,"temperature":0,"messages":[{"role":"user","content":"17*23=? 简答"}]}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -4   # 4×~23.1G
```
判据：**数学对（391）+ 中文流畅**（模拟层错 = 乱码）；timings `draft_n_accepted>0`（MTP 工作）；decode 50-56 tok/s。

## 性能水位（4×P40 vs 8×V100 atlas11，同 prompt 实测）

- **prefill 255 vs 904 tok/s（3.5×）；decode 50-62 vs 221 tok/s（4×）；MTP 接受率 37-61% vs 76%
- 构成：硬件 2.6×（30 SM/347GB/s vs 80 SM/900GB/s）+ 移植损失（wmma shuffle 模拟、hc 20 块、4 卡 expert spill）——**这是预期水位不是 bug**
- **GDN 分块移植实验（1104 实测定案）**：k_gdn_ch_rec 的 ks/qs 出 smem 直读 xc（L2 驻留，smem 72.6→41.3KB ≤ 48KB），GCH=32 算法/tile 零改动，正确性全过、引擎稳定——**但 prefill 实测 261.6/258.6 vs 旧逐行 255.4/259.9-264，持平**。定论：GDN 逐行 fallback 不是 prefill 损失源（旧归因错），真瓶颈=expert spill（PCIe）+ wmma 模拟 + 硬件。分块版价值=V100 算法对齐+免 FNX_PF_GDN_CHUNK=0 拐杖；已在 26 验证（新二进制、env 已删 FNX_PF_GDN_CHUNK）；**27 二进制已换 26 同款并重启，FNX_PF_GDN_CHUNK 已删**——27 树无 llama.cpp 子树不能本地编译，换二进制一律 26 编好后拷：scp 到 `bin/sfnx.new` 临时名再 mv（直写在跑二进制报 ETXTBSY "dest open Failure"；运行进程持旧 inode 不受影响，重启后生效），md5 双端核对 + `strings bin/sfnx | grep -c "PREAD slot"` 验特性再重启
- 短 prompt 的 prompt_per_second 偏低是固定开销假象；测 prefill 用 ≥1500 tok 稳态（复制一段中文×60）
