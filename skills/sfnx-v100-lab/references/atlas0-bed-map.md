# atlas0 sFNX 对照床地图与已测数字

## 环境

- **atlas0** (`ssh atlas0`, nvidia@<FLEET_HOST>, sudo 免密): 8×V100-SXM2-32GB, 503G RAM, Ubuntu 20.04/glibc 2.31, /usr/local/cuda-12.9 + g++-12, /fs00 挂载，无公网
- 床布局 `/data/sfnx0/`: `sfnx/`（生产树副本，FNX_NCARD=4 build）、`sfnx-f32/`（hc_inject f32 实验臂，硬链接树）、`llama.cpp/`（a37cc46 树 + CUDA build + libmtmd）、`gguf/`（4 分片 105G + mmproj-BF16 907M）、`mtp-rt/`（dense/experts/experts-q4_0 等 2.1G）
- 链接：`sfnx/models → ../gguf`、`sfnx/llama.cpp → ../llama.cpp`、`sfnx/data/mtp-rt → ../../mtp-rt`；mmproj-F16.gguf 是 mmproj-BF16.gguf 的软链
- GGUF sha256（与 atlas11 /data/strata-local/gguf 双端一致）:
  - 00001: 4448186216b3...  00002: 3f342f1c1580...  00003: 56758f40269c...  00004: 753bda48b98b...
- lab 目录：`/tmp/fnx-lab-1000{,-f32,-3card,-r32}`（基线/f32臂/3卡/r32 各一）

## 构建适配 sed（Ubuntu 20.04/glibc 2.31 专用，生产 22.04 不需要）

```bash
# nvcc 链接补 -lpthread（glibc 2.31 需显式；2.35+ 在 libc 里）
sed -i "s|-o \$@ \$^ -lcuda$|-o \$@ \$^ -lcuda -lpthread|" Makefile
sed -i "s|build/store.o -ldl -lcuda$|build/store.o -ldl -lcuda -lpthread|" Makefile
sed -i "s|-o \$@.new \$^ -lcublas -lcuda &&| -o \$@.new \$^ -lcublas -lcuda -lpthread \&\&|" Makefile
sed -i "s|-o \$@ \$^ -lcublas -lcuda$|-o \$@ \$^ -lcublas -lcuda -lpthread|" Makefile
sed -i "s|g++-12 -O2 -std=c++17 -o \$@ tools/gguf_test.cpp src/gguf.cpp$|& -lpthread|" Makefile
sed -i "s|-DCMAKE_C_COMPILER=gcc-14|-DCMAKE_C_COMPILER=gcc-12|" Makefile
sudo apt-get install -y ninja-build; git init -q llama.cpp   # Makefile 只 test -d .git
```

## 参考生成（首次建床必做）

```bash
export LD_LIBRARY_PATH=/data/sfnx0/llama.cpp/build/bin:$LD_LIBRARY_PATH
Q4=models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
./ref/fnx_logref $Q4 ref/code1.bin 300 ref/lgr/llama_cuda_code1.lgr   # edit1 同样 300 步
./ref/fnx_logref $Q4 ref/long64.bin 200 ref/lgr/llama_cuda_long64.lgr  # long64 是 200 步
# picture 用仓库自带 ref/img/sphinx.lgr，勿重新生成
```

## 已测 exactness 基线（atlas0，自产参考，bar=fnx_run.cpp BARS×1.25；**⚠此表=1028 补丁前树(uint16 inject)所测，1031 定案见下节，勿再当现役基线引用**）

| prompt | 3卡 | 4卡 | 4卡+RING_F32 | 4卡+inject_f32 | bar(KL) |
|---|---|---|---|---|---|
| code1 | 6.78e-4 / 98.3% | **6.06e-2 / 92.67%** | 5.90e-2 / 93.0% | 6.11e-2 / 92.3% | 6.06e-4 |
| edit1 | 1.29e-5 / 300/300 | 4.22e-6 / 300/300 | 1.03e-5 | 6.17e-6 | 5.15e-6 |
| long64 | 1.58e-3 / 196/200 | 1.60e-3 / 198/200 | 1.74e-3 | 1.81e-3 | 无(仅测) |
| picture | 5.25e-3 / 198/200 | 5.40e-3 / 197/200 | 6.37e-3 | 7.28e-3 | 7.98e-3 |

## code1 4 卡 bug 证据矩阵（截至建档，bug 未修）

- 4 卡 code1 KL 6e-2（bar 100×），top-1 92.67%，第 6 步起分叉；median 2.2e-3 + p99 1.05 + max 1.32 = 少数 step 大错型指纹
- 已排除：GGUF 传输损坏（双端 sha256 一致）/ 引擎自不一致（窗口 1/2/3/11/16 行字节全同）/ ring fp16 舍入（RING_F32 不救）/ hc_inject bf16（f32 不救）/ 全局 4 卡配置错（edit1/long64/picture 全过且 edit1 反而更好）
- 3 卡同机同参考 6.78e-4 → 4 卡特有；edit1 与 long64 的跨卡求和链路健康 → 非通用 ring/求和路径
- 疑似方向：4-way split 下某层块分配/索引在特定 token 序列上错（上游 AGENTS.md 自认 4+ 卡 "untested on hardware"）
- 深挖工具：ref/compare.py 逐层 RMS 对拍（fnx_ref 的层 dump 管线）；ViT 侧同期发现的跨卡 ring 循环 bug（vit_ring_dir/k_vit_add_bias）提示 4 卡 ring 计划值得复查

## 1031→1102 code1 4 卡发散定案（三次反转，最终定性=首执行效应，树无关）

- **⚠上面 1028 基线表与"证据矩阵"整节已废**：所有数字=fresh holder 首跑测量，被"首执行效应"污染
- **1102 终态**：exact 装置 code1 路（整段 58tok 单窗 prefill+300 步）在 **fresh holder 第一跑**产出真实坏 logits（~6e-2，.lgr 与暖态内容不同、完整无截断——非比较器竞态）；同 holder 复跑立即干净且此后字节稳定。**两树（uint16/f32 inject）同样中招**；raw 窗口路冷启动首跑=暖态**字节一致**干净=引擎冷启动本体无罪。1029 落树的 f32_inj 补丁=小精度改进（窗口不变性），**非本 bug 修复**
- **判读纪律（血泪定版）**：exact 判定必须 fresh 首跑+同 holder 复跑**双跑对照**，单跑不作数；raw 窗口路是冷启动对照的首选（已知冷启动干净）
- **待破**：首执行未初始化/首用状态在 code1 路的哪一处；3 卡首跑是否也爆（"4 卡特有"未重测）

- **判读纪律**：任何"基线表"引用前先核树版本（git 无仓，用 src mtime/`grep "const uint16_t\* __restrict__ w_inj" src/prefill.cu` 判 uint16/float 版本——**两版本 code1 首跑同爆，树版本与本 bug 无关**）；测量数字溯源=全转写 grep `"stdout":"…数字…"` 找原始工具输出
- **操作坑**：①cp -al 覆盖不断链（GNU cp 对硬链接目标写穿），臂内改文件须 rm+cp；②3 卡 build 须 FNX_LAB_GPUS 显式给 3 张卡（引擎拒 4 可见卡）；③原始 6 文件备份=/tmp/bundle-x/data/sfnx/src/（源自 /fs00/sfnx-atlas0-stage/bundle.trees.tar 1028 22:03 打包）；④**make 默认目标只到 build/，holder 换卡数后须显式 `make FNX_NCARD=N bin/fnx_hold bin/fnx_lab` 重链——否则 holder 仍是旧卡数二进制，静默测错**
- ~~**生产待办**~~ **已完成(1031 下午)**:atlas11 两树均已移植 f32 补丁(实际 7 文件:补丁链 6 文件+rope_gguf.hpp)+重编+重启。终态=**双实例八卡**:8093(1M ctx,GPU0-3,131 t/s)+8094(64K/18 lane,GPU4-7)同机并存;sfnx32 树=MAXT18 版 kernels.cu/model.cpp/model.hpp 与 f32 补丁三方合并(patch --batch 干净合入)。KV 亲和=24.23 网关 session_groups 粘性路由实测命中(同会话二轮 cached_tokens=64,异会话散列分流 3:5)。回滚备份=两树 src.bak-1031-preF32/;atlas0 同日退役对外 serve(实验床保留,复刻法见下节不变)

## atlas11 → atlas0 引擎整体搬迁（2026-10-31 实操定案）

生产 `bin/sfnx`（atlas11 构建，NCARD=4）**不能直接在 atlas0 跑**：要 GLIBC_2.32/2.33/2.34，atlas0(20.04) 只有 2.31 → 必须本机重编：`cd /data/sfnx0/sfnx && PATH=/usr/local/cuda/bin:$PATH make FNX_NCARD=4 bin/sfnx -j32`（Makefile 默认 NCARD=3，先 `cat build/ncard` 对生产值；atlas11 现值=4）。

Python 前端链（front.server）：atlas11 用 `/data/sfnx-py`（独立 py3.14 + .venv）。搬迁=本机中转 tar：`ssh atlas11 'tar cf - -C /data sfnx-py sfnx/.venv' > x.tar` → scp 到 atlas0 **/raid**（/data 根目录 nvidia 用户不可写）→ 解包后 venv 搬家三处修复：①`python3.14` symlink 指向原路径 `/data/sfnx-py/python/bin/python3.14`，重链到实际位置；②`pyvenv.cfg` 里 `home=`/`executable=` sed 替换旧前缀；③`sfnx/.venv` 是指向 `/data/sfnx0/.venv`（不存在）的死链，重指 `/raid/sfnx-py/.venv`。验证 `.venv/bin/python -c "import jinja2,psutil,PIL"`。

起服务（atlas0 无 systemd-run 流程依赖时直接 nohup）：`cd /data/sfnx0/sfnx && ulimit -l unlimited; setsid nohup .venv/bin/python -m front.server --engine sfnx --config serve/sfnx-atlas0.json > serve/atlas0-8093.log 2>&1 &`（config=生产 sfnx-8093.json 副本，GPU[0-3]，port 8093，FNX_MTP_EXPERTS=q4_0/FNX_YARN_ORIG_CTX=262144 在 config env 里）。~60s ready；`/health` + 9.11vs9.9 ✓，单流 decode ~73 t/s（SXM2 4卡 + MTP spec6 + lookup15）。

## 其它定案（精度恢复试点，勿重试）

- **hc_inject bf16→f32**（model.cpp 两处装载 + hc.cuh HcArgs/kernel + prefill.cu k_hcb_prep/k_hcb_mixinj 签名链，共 13 处）：A/B 全面略差（edit1/picture/long64 全部变差，code1 不变）→ 不采纳。bf16 inject 的舍入损失本来可忽略，改动引入的求和扰动是净损失
- **FNX_RING_F32=1 在 4 卡**：全面略差（见上表）→ 4 卡上连作者 "-30% 速度不值" 的判决都不需要，方向本身就负
- MTP experts q2_0→q4_0：draft 接受率 17.0%→17.8%，显存/速度零变化（两模式解包后同为 Q4_0 块）；已上生产双池

## 跨臂实验操作记录（供复制）

```bash
# 硬链接臂 + 补丁 + 重编 + 独立 lab
cd /data/sfnx0 && cp -al sfnx sfnx-<arm>            # 改动文件自动断链
# <打补丁>
cd sfnx-<arm> && PATH=/usr/local/cuda/bin:$PATH make FNX_NCARD=4 -j32 lab test-bin
stat -c %i bin/fnx_run ../sfnx/bin/fnx_run           # inode 必须不同
export FNX_LAB_DIR=/tmp/fnx-lab-1000-<arm>
tools/lab.sh start && tools/lab.sh exact base        # 或 exact <name>
tools/lab.sh unload                                   # 换臂/换卡数前
# 换卡数（同机 N 卡对照）
make FNX_NCARD=3 -j32 lab test-bin                   # build/ncard 重写全量重编
```
