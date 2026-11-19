---
name: "sfnx-v100-lab"
description: "sFNX引擎复现/exactness/多卡A/B对照实验床。涉及sfnx或lab.sh exact时激活。"
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  tags: [inference, sfnx, v100, precision, lab]
  related_skills: [infra, infra-misc, cc-session-ops]
---

# sFNX V100 Lab：复现·exactness·A/B

## When to Use

- 在生产机（atlas11）之外的 V100 节点复刻 sFNX 环境做实验（生产机卡满/不能折腾）
- 把 sFNX 移植到 P40（sm_61）等非 Volta 硬件并部署服务：模拟层设计/树传输/服务/IOMMU 闸门 → `references/p40-sm61-port.md`（启动六关全破 IOMMU→全 mesh P2P→KV 池预算→vit/arena 余量→GDN prefill smem→wmma 布局：定版 = NCARD=4 全互通子组 + 262144 ctx + reserve 2300 + GDN chunk=0；第六关根因 = sm61_wmma.h load/store 与 mma_sync 布局约定不一致（lanes 16-31 越界写 smem = attn_prompt illegal access），修正 + 单元测试 err=0（scripts/wmma_sm61_test.cu）后重编）
- 补跑或新跑 `tools/lab.sh exact` 精度验收（llama.cpp 参考生成是前置，生产机从未建过）
- 定位多卡（FNX_NCARD）相关数值 bug：需要 N 卡 vs N±1 卡同机对照
- 精度恢复改动的 A/B 验证（改一处算子/精度开关，验证是否真有收益）

sFNX 是自研 CUDA 引擎（sm_70/CUDA12.9，Qwen3.8-Flash-Next），生产在 atlas11（PCIe V100），上游 AGENTS.md/docs 只覆盖三卡生产形态。环境地图/构建补丁/已测数字 → `references/atlas0-bed-map.md`。

## 流程（首次在新 V100 节点建床）

1. **复刻生产树**（atlas11 → /fs00 → 目标节点，禁 rsync 写 /fs00）：
   - 大文件（GGUF 分片）直接 `cp` 到 /fs00 stage；目录树（源码+llama.cpp+mtp-rt）打单个 `bundle.trees.tar`（`tar -C / --exclude sfnx/build --exclude bin --exclude .venv -cf ... data/sfnx data/llama.cpp data/strata/Strata-data/mtp/rt`），目标本地盘解包后 `mv` 重排——squash 挂载上解包必死，本地盘无限制
   - 重建链接：`models` → 本地 gguf 目录（UD-Q4_K_XL 分片 + mmproj-F16→BF16 软链）；`llama.cpp` → 本地树；`data/mtp-rt` → 本地 mtp-rt
   - **传输完整性必做双端 sha256**（尺寸对≠内容对）
2. **构建适配**（Ubuntu 20.04/glibc 2.31 节点，如 atlas0）：Makefile 五处 sed——nvcc 链接规则补 `-lpthread`（fnx_hold/fnx_lab/libfnx_lab.so/fnx_run 系）、gguf_test 链接行、`gcc-14→gcc-12`；`sudo apt-get install -y ninja-build`；llama.cpp 树 `git init`（Makefile 只 test -d .git）。全部现成命令见 reference
3. **先生成 llama.cpp 参考**（exact 的尺子；生产机从未建过 ref/lgr）：`make ref` 后 `./ref/fnx_logref $Q4 ref/code1.bin 300 ref/lgr/llama_cuda_code1.lgr`（edit1 同 300、long64 用 200 步；LD_LIBRARY_PATH 指向 llama.cpp/build/bin）。picture 不用生成——仓库自带 `ref/img/sphinx.lgr`（还可用它验证整条管线健康度）
4. **lab 流程**：`export FNX_LAB_DIR=/tmp/fnx-lab-<臂名>`（必须 export！）→ `tools/lab.sh start` → `tools/lab.sh exact base`（写自身基线，全部过线才落盘）→ 改动臂用 `tools/lab.sh exact <name>`，与 base 同目录
5. **多卡对照**：`make FNX_NCARD=N` 会重写 `build/ncard` 并全量重编；换卡数 = 换实验，lab holder 先 `tools/lab.sh unload`
6. **A/B 臂用硬链接树**：`cp -al sfnx sfnx-<arm>`——源文件零拷贝，改动的文件自动断链成独立副本，`make` 只重编改动源的对象；两臂各自 build/ 目录互不污染

## 判读规则

- **隔离变量必同机同参考**：N 卡 bug 的判定实验是同机重编 N±1 卡跑同套参考，不要跨节点比（节点差异/参考差异不可分）
- **排除法矩阵**（定位数值 bug 的顺序）：GGUF sha256 双端一致 → 窗口 1/2/3/11/16 行字节一致（引擎自洽，非随机崩）→ 可疑开关逐个关（如 FNX_RING_F32=1 不救=非 ring 舍入）→ 单 prompt 爆而其余过 = token 序列特异触发，非全局配置错
- **KL 分布形状即指纹**：median 高=全程漂移（求和顺序/精度类）；median 低而 p99/max 高=少数 step 大错（分叉/索引错）
- **负结果也是结果**：精度恢复试点（如权重 bf16→f32）A/B 全面略差即可定案不采纳——求和顺序扰动可能大于原舍入损失，改动本身有代价
- exact base 的 FAIL 不落盘基线（base-new* 保留可读）；判读失败臂时看 vs llama.cpp 行而非只看 FAIL

## Pitfalls

- **FNX_LAB_DIR 必须 export**：前缀赋值（`FNX_LAB_DIR=x tools/lab.sh start`）只作用于 start；后续 exact 走默认目录 → fifo 写阻塞超时 → 顺手杀掉真 worker、新 worker 找不到默认目录 holder 空转退出——症状是 exact 无输出且 out.log 只有 worker 重启行
- **`exact base` 全过才写基线**；失败时文件留 base-new*。改动臂首次跑会报 no base——先跑 base 或直接读 vs llama.cpp 行
- **lab.sh start 传引擎变量**：`tools/lab.sh start "" "-E VAR=value"`（第一参是模型路径占位，空串用默认 Q4）
- **picture 参考仅 Q4 可用**（fnx_run 里 q4 才跑 picture）；Q8 模型 long64 需 FNX_EXACT_REF 指定
- **构建产物 inode 校验**：A/B 两臂的 bin/fnx_run 必须不同 inode（`stat -c %i`）——硬链接树不重编时是同一文件，比了个寂寞
- **跨树同步补丁先全量 md5 清单再 diff，但差异文件必须再逐个 `diff -u` 确认**：两机 locale 不同会让 `find|xargs md5sum` 输出排序不一致产生假阳性差异（同文件出现在 diff 两侧）——以 diff -u 实质内容为准，清单只做候选筛选
- **补丁与本地改造重叠时用三方合并而非直拷**：目标树若在待同步文件上另有补丁（如 MAXT=18 dispatch 扩容），直接覆盖会丢本地件；`diff -u base/<f> fixed/<f> > f.patch` + `cp target/<f> merged/` + `patch --batch merged/<f> < f.patch`，干净合入后 grep 三特征（补丁特征 + 本地特征 + 无回归特征）验身份再部署
- **选 FNX_NCARD 前先核实目标机实卡数**（`nvidia-smi -L | wc -l` + `lspci | grep -i nvidia`）：舰队同型机卡数不同（如 p40 同批有 6 卡与 8 卡），传闻值/历史笔记值不可信；Makefile 只校验合法档位 2/3/4/6/8，不校验机上有那么多卡。**且实卡数≠可用档位**——引擎要求全对 P2P 互通，多交换机背板可能只有子组互通（p40-26 六卡仅 {2,3,4,5} 全互通 → NCARD=4 + CUDA_VISIBLE_DEVICES 圈组），动手前先枚举 P2P 矩阵（gpu-pcie-p2p-acs-ops 的 templates/p2p_test.cu）
- **档案里的速度数字引用前必须实测 A/B**：skill/memory 里的历史 tok/s（含“作者数字/单次个例/不同负载口径”）不可直接当现状基线——同题同参对跑两台现役端点再下结论；历史数字只用于解释量级（如“80-205 t/s 负载依赖”意味着单次 73 t/s 不算异常）

详细环境地图、构建 sed 全文、已测基线数字与 code1 4 卡 bug 证据矩阵 → `references/atlas0-bed-map.md`；长上下文工具调用验证脚本 → `scripts/tool_longctx_test.py`；f32 inject 修复补丁全文与上游回馈 issue 要点 → `references/f32-inject-patch.md`；P40/sm_61 移植全链路（模拟层/构建/悬空链/服务/IOMMU/六关） → `references/p40-sm61-port.md`；wmma 模拟层单元测试 → `scripts/wmma_sm61_test.cu`
