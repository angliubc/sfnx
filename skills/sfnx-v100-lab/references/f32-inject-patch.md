# hc_inject f32 修复：补丁全文与上游回馈 issue

## 修复要点（判读与验证详见 atlas0-bed-map.md 1031 定案节）

根因 = hc_inject 权重 bf16(8位尾数) × 4卡 ring 3跳 fp16 舍入在 code1 类序列复合发散（KL 6.06e-2 / top-1 92.67%）；修复 = inject 权重升 f32 加载 + 两个消费核改纯 f32 乘加，6 文件 +45 行，代价 +2MB/卡。

**半措施陷阱**：只改加载器不改核无效——消费核用 `__uint_as_float((unsigned) w_inj[...] << 16)` 重导 bf16，必须整链改（model.cpp 加载器 + model.hpp/hc.cuh/kernels.hpp 类型链 + prefill.cu 两个核 + kernels.cu 签名）。

## 完整补丁（apply 到任意老树：`patch -p1 < this`，先 grep `const uint16_t\* __restrict__ w_inj` 应为 2 处命中）

```diff
--- a/src/model.cpp
+++ b/src/model.cpp
@@ -259,6 +259,23 @@
             std::memcpy(&h[i * t.ne[0]], t.data + (size_t) rows[i] * t.ne[0] * 4, (size_t) t.ne[0] * 4);
         return put<float>(h.data(), h.size() * 4);
     }
+    // inject.weight -> f32 from F32/BF16/Q8_0 (precision: was bf16)
+    float* f32_inj(const Tensor& t) {
+        const size_t n = (size_t) (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]);
+        std::vector<float> h(n);
+        if (t.type == T_F32) std::memcpy(h.data(), t.data, n * 4);
+        else if (t.type == T_BF16) {
+            const uint16_t* b = (const uint16_t*) t.data;
+            for (size_t i = 0; i < n; ++i) { const uint32_t u = (uint32_t) b[i] << 16; std::memcpy(&h[i], &u, 4); }
+        } else if (t.type == T_Q8_0) {
+            for (size_t bl = 0; bl < n / 32; ++bl) {
+                const uint8_t* blk = t.data + bl * 34; uint16_t dh; std::memcpy(&dh, blk, 2);
+                const float d = f16_to_f32(dh);
+                for (int j = 0; j < 32; ++j) h[bl * 32 + j] = d * (float) (int8_t) blk[2 + j];
+            }
+        } else throw std::runtime_error(t.name + ": f32_inj supports F32, BF16 and Q8_0");
+        return put<float>(h.data(), h.size() * 4);
+    }
     // Q8_0 -> bf16, the whole tensor; F32 -> bf16 likewise
     uint16_t* bf16(const Tensor& t) {
@@ -528,7 +545,7 @@
-            w.hc_inj[h] = up.bf16(L(l, (std::string(hp[h]) + "inject.weight").c_str()));
+            w.hc_inj[h] = up.f32_inj(L(l, (std::string(hp[h]) + "inject.weight").c_str()));
@@ -770,7 +787,11 @@
-        w.hc_inj[h] = up.put<uint16_t>(bf16_of(inj), inj.nbytes);
+        {   std::vector<float> hinj((size_t) inj.nbytes / 2);
+            const uint16_t* bi = bf16_of(inj);
+            for (size_t i = 0; i < hinj.size(); ++i) { const uint32_t u = (uint32_t) bi[i] << 16; std::memcpy(&hinj[i], &u, 4); }
+            w.hc_inj[h] = up.put<float>(hinj.data(), hinj.size() * 4);
+        }
--- a/src/model.hpp
+++ b/src/model.hpp
@@ -96,7 +96,7 @@
-    uint16_t* hc_inj[2] = {};          // bf16 [HC][D]
+    float* hc_inj[2] = {};             // f32 [HC][D] (was bf16: the only 8-mantissa-bit weights left, +2 MB/card)
--- a/src/hc.cuh
+++ b/src/hc.cuh
@@ -32,7 +32,7 @@
-    const uint16_t* w_inj;
+    const float* w_inj;
@@ -184,21 +184,19 @@
-            const uint4* w4 = reinterpret_cast<const uint4*>(a.w_inj + (size_t) (rr - LR) * D + col0);
-            uint4 u[NQ];
+            constexpr int NQF = 2 * NQ;   // a float4 holds 4 f32 where a uint4 held 8 bf16
+            const float4* w4 = reinterpret_cast<const float4*>(a.w_inj + (size_t) (rr - LR) * D + col0);
+            float4 u[NQF];
-            for (int q = 0; q < NQ; ++q) u[q] = __ldg(w4 + q * H_TPRD + j);
+            for (int q = 0; q < NQF; ++q) u[q] = __ldg(w4 + q * H_TPRD + j);
-                for (int q = 0; q < NQ; ++q) {
+                for (int q = 0; q < NQF; ++q) {
-                    const uint16_t* hh = reinterpret_cast<const uint16_t*>(&u[q]);
-                    const float4 x0 = *reinterpret_cast<const float4*>(&xs[k][ch * 8]);
-                    const float4 x1 = *reinterpret_cast<const float4*>(&xs[k][ch * 8 + 4]);
-                    acc += bf(hh[0]) * x0.x + bf(hh[1]) * x0.y + bf(hh[2]) * x0.z + bf(hh[3]) * x0.w + bf(hh[4]) * x1.x +
-                           bf(hh[5]) * x1.y + bf(hh[6]) * x1.z + bf(hh[7]) * x1.w;
+                    const float4 x0 = *reinterpret_cast<const float4*>(&xs[k][ch * 4]);
+                    acc += u[q].x * x0.x + u[q].y * x0.y + u[q].z * x0.z + u[q].w * x0.w;
--- a/src/kernels.cu
+++ b/src/kernels.cu
@@ -1014,7 +1014,7 @@
-             const uint16_t* w_inj, float eps, float* mixed, float* inject, float* scratch, unsigned* err, int T, void* st,
+             const float* w_inj, float eps, float* mixed, float* inject, float* scratch, unsigned* err, int T, void* st,
--- a/src/kernels.hpp
+++ b/src/kernels.hpp
@@ -56,7 +56,7 @@
-             const Q8S& w_up, const uint16_t* w_inj, float eps, float* mixed, float* inject, float* scratch, unsigned* err,
+             const Q8S& w_up, const float* w_inj, float eps, float* mixed, float* inject, float* scratch, unsigned* err,
--- a/src/prefill.cu
+++ b/src/prefill.cu
@@ -126,7 +126,7 @@
-                                                          float eps, const uint16_t* __restrict__ w_inj, float* __restrict__ inject_out) {
+                                                          float eps, const float* __restrict__ w_inj, float* __restrict__ inject_out) {
@@ -155,7 +155,7 @@
-                for (int r = 0; r < HC; ++r) acc[r] = fmaf(__uint_as_float((unsigned) w_inj[(size_t) r * D + c * N + i] << 16), xn, acc[r]);
+                for (int r = 0; r < HC; ++r) acc[r] = fmaf(w_inj[(size_t) r * D + c * N + i], xn, acc[r]);
@@ -184,7 +184,7 @@
-                                                            const uint16_t* __restrict__ w_inj,
+                                                            const float* __restrict__ w_inj,
@@ -205,7 +205,7 @@
-        acc += __uint_as_float((unsigned) w_inj[(size_t) r * D + i] << 16) * (Rt[i] * wn[i] * r_s[i / N]);
+        acc += w_inj[(size_t) r * D + i] * (Rt[i] * wn[i] * r_s[i / N]);
@@ -460,7 +460,7 @@
-                const uint16_t* w_inj, float eps, float* mixed, float* inject, int C, cudaStream_t s) {
+                const float* w_inj, float eps, float* mixed, float* inject, int C, cudaStream_t s) {
```

（上方为压缩示意；完整未前缀补丁在作者可发版时从任一已修复树现生：`for f in model.cpp model.hpp hc.cuh kernels.cu kernels.hpp prefill.cu; do diff -u --label a/src/$f --label b/src/$f <老树>/src/$f <修复树>/src/$f; done > sfnx-f32-inject.patch`。备用完整件副本=atlas11 /tmp/f32sync-1031/）

## 上游回馈（johnfink8/sfnx，用户已令回馈、发布被打断未发）

- gh 已认证（angliubc），仓库无 4 卡发散相关 issue → 新开（`gh issue create -R johnfink8/sfnx -t "Four cards: code1 diverges (KL 6.06e-2) — root cause bf16 hc_inject, fix + A/B evidence" -F <body>`）
- 英文正文完整版（含 A/B 数据表/半措施陷阱/picture 轻微回归/环境段）= 本机 scratch `issue-body-4card-f32.md`（若被清理，从本文件+bed-map 1031 节重写：What we saw → Root cause → Fix + A/B evidence → Pitfall → Patch → Environment）
- 相关既有 issue：#4（8卡 Store/VMM，另案）
