// The tower's device half (vit.hpp): the loader (mmproj -> the cards' slices) and the forward (one picture's
// patches as rows). The graph is llama.cpp's clip_graph_qwen3vl op for op (tools/mtmd/models/qwen3vl.cpp); the
// token order it implies — the patches in "quad" order, token t = 4*cell + dx + 2*dy — and the vision M-RoPE's
// exact angles are derived in the comments at each kernel. Numerics: fp32 everywhere — the weights widened to f32
// once at load and cublasSgemm (sm_70 has no BF16 GEMM), LayerNorm accumulated in double as ggml's CPU kernel does,
// softmax without flash attention.
#include "vit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "env.hpp"
#include "vit_host.hpp"

namespace fnx {

namespace {

void vck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("vit: ") + what + ": " + cudaGetErrorString(e));
}
void vckb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("vit: cublas ") + what + ": " + std::to_string((int) s));
}
void vcud(CUresult e, const char* what) {
    if (e != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(e, &s);
        throw std::runtime_error(std::string("vit: ") + what + ": " + (s ? s : "?"));
    }
}
unsigned vgrid(size_t n) { return (unsigned) std::min<size_t>(4096, (n + 255) / 256); }

float bf16_to_f32_host(uint16_t v) {
    const uint32_t u = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32_host(uint16_t h) {   // ggml's f16 -> f32 (model.cpp's, carried over: the F16 mmproj needs it)
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else {
            int ee = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) { mm <<= 1; --ee; }
            u = (s << 31) | ((uint32_t) (ee + 127 - 14) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) u = (s << 31) | 0x7f800000u | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f32_to_bf16_host(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float f32_of(const Tensor& t, size_t i) {
    if (t.type != T_F32) throw std::runtime_error("vit: " + t.name + " is not F32");
    return ((const float*) t.data)[i];
}

// ---- the kernels
__device__ __forceinline__ float vit_gelu(float x) {   // ggml's ggml_gelu_f32 (the tanh form, CPU and CUDA alike)
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

// LayerNorm with bias over rows of `width` (one block a row): mean and the centered variance both accumulated in
// double, as ggml's norm_f32 does; y = (x - mean) / sqrt(var + eps) * w + b
__global__ void k_vit_ln(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                         float* __restrict__ y, int width, float eps) {
    const float* xr = x + (size_t) blockIdx.x * width;
    float* yr = y + (size_t) blockIdx.x * width;
    __shared__ double red[2][8];
    __shared__ float mean, inv;
    double s = 0.0;
    for (int i = threadIdx.x; i < width; i += blockDim.x) s += (double) xr[i];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if ((threadIdx.x & 31) == 0) red[0][threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        double a = 0.0;
        for (int i = 0; i < (int) ((blockDim.x + 31) / 32); ++i) a += red[0][i];
        mean = (float) (a / width);
    }
    __syncthreads();
    double v = 0.0;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const double d = xr[i] - mean;
        v += d * d;
    }
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0) red[1][threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        double a = 0.0;
        for (int i = 0; i < (int) ((blockDim.x + 31) / 32); ++i) a += red[1][i];
        inv = 1.0f / sqrtf((float) (a / width) + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// the patch conv's im2col: column p (patch px = p % gx, py = p / gx) gathers the 3x16x16 patch's taps in ggml's
// order (ci*256 + ky*16 + kx: KW fastest, then KH, then IC), reading the channel-planar pixels [3][ny][nx]
__global__ void k_vit_conv_im2col(const float* __restrict__ img, float* __restrict__ cols, int nx, int ny, int gx) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= gx * (ny / VIT_PATCH)) return;
    const int px = p % gx, py = p / gx;
    float* col = cols + (size_t) p * (3 * VIT_PATCH * VIT_PATCH);
    for (int ci = 0; ci < 3; ++ci)
        for (int ky = 0; ky < VIT_PATCH; ++ky) {
            const float* row = img + ((size_t) ci * ny + 16 * py + ky) * nx + 16 * px;
            for (int kx = 0; kx < VIT_PATCH; ++kx) col[ci * 256 + ky * 16 + kx] = row[kx];
        }
}

// the patches from row-major (py*gx + px) into the quad order the tower runs in: token t = 4*cell + dx + 2*dy,
// cell = my*(gx/2) + mx, its patch (2*mx + dx, 2*my + dy). This is ggml's permute/cont/reshape dance (derived in
// vit.hpp's header comment) collapsed into one gather; the merger later reads tokens 4m..4m+3 as cell m's 4608
// channels, so the quad order must be exactly this one
__global__ void k_vit_quad_reorder(const float* __restrict__ x, float* __restrict__ dst, int gx, int gy) {
    const int mw = gx / 2, n_cells = mw * (gy / 2);
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) n_cells * 4 * VIT_H;
         e += (size_t) gridDim.x * blockDim.x) {
        const int cell = (int) (e / (4 * VIT_H)), rem = (int) (e % (4 * VIT_H));
        const int quad = rem / VIT_H, i = rem % VIT_H;
        const int mx = cell % mw, my = cell / mw;
        const int px = 2 * mx + (quad & 1), py = 2 * my + (quad >> 1);
        dst[e] = x[(size_t) (py * gx + px) * VIT_H + i];
    }
}

// the learned position table (48x48, column x + y*48) sampled with ggml's bilinear ALIGN_CORNERS interpolation to
// the (gx, gy) patch grid — sf = (out-1)/(in-1), no pixel offset, the four neighbours clamped — and added to token
// t, whose patch (px, py) comes from the quad order above
__global__ void k_vit_pos_add(float* __restrict__ x, const float* __restrict__ table, int gx, int gy) {
    const int t = blockIdx.x;
    const int mw = gx / 2;
    const int cell = t / 4, quad = t % 4;
    const int mx = cell % mw, my = cell / mw;
    const int px = 2 * mx + (quad & 1), py = 2 * my + (quad >> 1);
    const float fx = gx > 1 ? (float) px * 47.0f / (float) (gx - 1) : 0.f;
    const float fy = gy > 1 ? (float) py * 47.0f / (float) (gy - 1) : 0.f;
    int x0 = (int) floorf(fx), y0 = (int) floorf(fy), x1 = x0 + 1, y1 = y0 + 1;
    x0 = x0 < 0 ? 0 : x0 > 47 ? 47 : x0;
    x1 = x1 < 0 ? 0 : x1 > 47 ? 47 : x1;
    y0 = y0 < 0 ? 0 : y0 > 47 ? 47 : y0;
    y1 = y1 < 0 ? 0 : y1 > 47 ? 47 : y1;
    const float dx = fminf(fmaxf(fx - x0, 0.f), 1.f), dy = fminf(fmaxf(fy - y0, 0.f), 1.f);
    float* xr = x + (size_t) t * VIT_H;
    // the table's rows are the file's [token][channel] (flat = c + t*1152): row r starts at r*1152.
    // (the old i*2304 stride read the file transposed — every token got another token's position)
    const float* row = table + (size_t) (y0 * 48 + x0) * VIT_H;
    const float* row_x1 = table + (size_t) (y0 * 48 + x1) * VIT_H;
    const float* row_y1 = table + (size_t) (y1 * 48 + x0) * VIT_H;
    const float* row_y1x1 = table + (size_t) (y1 * 48 + x1) * VIT_H;
    for (int i = threadIdx.x; i < VIT_H; i += blockDim.x) {
        xr[i] += row[i] * (1 - dx) * (1 - dy) + row_x1[i] * dx * (1 - dy) +
                 row_y1[i] * (1 - dx) * dy + row_y1x1[i] * dx * dy;
    }
}

// the vision M-RoPE on this card's heads. ggml's rotate_pairs with GGML_ROPE_TYPE_VISION rotates pairs (i, i+36)
// of each 72-wide head; the angle of pair r (0..35) is pos * 10000^(-2j/36) with j = r % 18 — the theta restarts at
// the section boundary (ggml_mrope_cache_init's indep_sects reset) — and the position is the token's y for r < 18,
// its x for r >= 18 (clip.cpp fills positions[t] = y, [n+t] = x, and the w/e sections never reach ne0 = 72). The
// token's (px, py) comes from the quad order
__global__ void k_vit_rope(float* __restrict__ qk, int gx, int n_pos, int nh) {
    const int mw = gx / 2;
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) n_pos * 36;
         e += (size_t) gridDim.x * blockDim.x) {
        const int t = (int) (e / 36), r = (int) (e % 36);
        const int cell = t / 4, quad = t % 4;
        const int mx = cell % mw, my = cell / mw;
        const int px = 2 * mx + (quad & 1), py = 2 * my + (quad >> 1);
        // r is the pair index (0..35): ggml's rope_vision rotates the pair (r, r+36) exactly once —
        // pairs 0..17 with the y position, 18..35 with x, the exponent restarting at 18 (the old
        // j = r % 18 made threads r and r + 18 race on the same pair, leaving the rope nondeterministic)
        const float pos = r < 18 ? (float) py : (float) px;
        const float theta = pos * powf(10000.f, -2.f * (r % 18) / 36.f);
        const float cs = cosf(theta), sn = sinf(theta);
        const int j = r;
        float* row = qk + (size_t) t * (nh * 216);
        for (int hh = 0; hh < nh; ++hh) {   // a head's Q at +0, K at +72 (V at +144 needs no rope)
            float* qh = row + hh * 216;
            float* kh = qh + 72;
            const float q0 = qh[j], q1 = qh[j + 36], k0 = kh[j], k1 = kh[j + 36];
            qh[j] = q0 * cs - q1 * sn;
            qh[j + 36] = q0 * sn + q1 * cs;
            kh[j] = k0 * cs - k1 * sn;
            kh[j + 36] = k0 * sn + k1 * cs;
        }
    }
}

// one head's softmax attention over every position, streaming (online softmax: no score scratch, so the buffers
// cost nothing per picture size). Q, K, V of local head h at qk[t][h*216 + {0, 72, 144}]; the context into
// att[t][h*72]. Scale 1/sqrt(72) as ggml's soft_max_ext applies it (before the max)
__global__ void k_vit_attn(const float* __restrict__ qk, float* __restrict__ att, int n_pos, int nh) {
    const int q = blockIdx.x, h = blockIdx.y;
    if (q >= n_pos) return;
    const int stride = nh * 216;
    const float* Q = qk + (size_t) q * stride + (size_t) h * 216;
    __shared__ float sc[128];   // one tile's scores
    __shared__ float acc[VIT_DH];
    __shared__ float run[2];    // the running max and sum
    if (threadIdx.x == 0) {
        run[0] = -INFINITY;
        run[1] = 0.f;
    }
    for (int i = threadIdx.x; i < VIT_DH; i += blockDim.x) acc[i] = 0.f;
    __syncthreads();
    for (int k0 = 0; k0 < n_pos; k0 += 128) {
        const int k = k0 + threadIdx.x;
        float s = -INFINITY;
        if (k < n_pos) {
            const float* K = qk + (size_t) k * stride + (size_t) h * 216 + 72;
            s = 0.f;
            for (int i = 0; i < VIT_DH; ++i) s += Q[i] * K[i];
            s *= 0.11785113019775793f;
        }
        sc[threadIdx.x] = s;
        __syncthreads();
        if (threadIdx.x == 0) {   // the tile folded into the running (max, sum), the accumulator rescaled
            float m = run[0];
            for (int j = 0; j < 128 && k0 + j < n_pos; ++j) m = fmaxf(m, sc[j]);
            const float scale = run[0] == -INFINITY ? 0.f : __expf(run[0] - m);
            for (int i = 0; i < VIT_DH; ++i) acc[i] *= scale;
            run[1] *= scale;
            float ts = 0.f;
            for (int j = 0; j < 128 && k0 + j < n_pos; ++j) {
                sc[j] = __expf(sc[j] - m);
                ts += sc[j];
            }
            run[0] = m;
            run[1] += ts;
        }
        __syncthreads();
        if (threadIdx.x < VIT_DH) {   // the tile's keys folded into the accumulator, one lane a channel
            const int i = threadIdx.x;
            for (int j = 0; j < 128 && k0 + j < n_pos; ++j)
                if (sc[j] != 0.f) {
                    const float* V = qk + (size_t) (k0 + j) * stride + (size_t) h * 216 + 144;
                    acc[i] += sc[j] * V[i];
                }
        }
        __syncthreads();
    }
    if (threadIdx.x < VIT_DH) att[(size_t) q * (nh * VIT_DH) + (size_t) h * VIT_DH + threadIdx.x] = acc[threadIdx.x] / run[1];
}

// the small elementwise steps: a row-bias add, a residual add of a summed partial plus its bias, GELU in place,
// the activations staged f32 -> BF16 for a GEMM
__global__ void k_vit_bias_row(float* __restrict__ x, const float* __restrict__ bias, size_t n, int width) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x)
        x[e] += bias[e % width];
}
__global__ void k_vit_add_bias(float* __restrict__ x, const float* __restrict__ a, const float* __restrict__ bias,
                               size_t n, int width) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x)
        x[e] += a[e] + bias[e % width];
}
// x = a + bias (the merger's output: the tower's final rows, no residual under them)
__global__ void k_vit_out_bias(float* __restrict__ x, const float* __restrict__ a, const float* __restrict__ bias,
                               size_t n, int width) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x)
        x[e] = a[e] + bias[e % width];
}
__global__ void k_vit_gelu(float* __restrict__ x, size_t n) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x)
        x[e] = vit_gelu(x[e]);
}
// one ring ADD: out = in + addend (the plan's PACK/SEND/SIGNAL/WAIT steps are copies and flags)
__global__ void k_vit_ring_add(const float* __restrict__ in, const float* __restrict__ add, size_t n,
                               float* __restrict__ out) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x)
        out[e] = in[e] + add[e];
}
// the fused QKV's bias: row layout [head][Q 72 | K 72 | V 72] takes the file's whole [3456] bias at the matching
// offsets (Q at h*72, K at 1152 + h*72, V at 2304 + h*72, h the global head)
__global__ void k_vit_qkv_bias(float* __restrict__ qk, const float* __restrict__ bias, int h0, int nh, size_t n) {
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x) {
        const int o = (int) (e % (size_t) (nh * 216));
        const int hh = o / 216, r = o % 216;
        const int h = h0 + hh;
        const int off = r < 72 ? h * 72 + r : r < 144 ? VIT_H + h * 72 + r - 72 : 2 * VIT_H + h * 72 + r - 144;
        qk[e] += bias[off];
    }
}

}  // namespace
}  // namespace fnx

namespace fnx {

namespace {

// ---- the loader's upload helper
struct VUp {
    VitCard& c;
    template <typename T> T* put(const void* host, size_t bytes, const char* name) {
        void* d = nullptr;
        vck(cudaMalloc(&d, bytes), name);
        if (host) vck(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice), name);
        c.allocs.push_back(d);
        c.bytes += bytes;
        return (T*) d;
    }
    void* raw(size_t bytes, const char* name) {
        void* d = nullptr;
        vck(cudaMalloc(&d, bytes), name);
        c.allocs.push_back(d);
        c.bytes += bytes;
        return d;
    }
    void f32_check(const Tensor& t) {
        if (t.type != T_F32) throw std::runtime_error("vit: " + t.name + " is not F32");
    }
    // a 2-D BF16/F16/F32 tensor's rows [r0, r1) as f32 (BF16/F16 widened on the host: the projector comes either way)
    float* f32_rows(const Tensor& t, int r0, int r1) {
        if (t.type != T_BF16 && t.type != T_F16 && t.type != T_F32) throw std::runtime_error("vit: " + t.name + " is not BF16/F16/F32");
        const int w = (int) t.ne[0];
        std::vector<float> h((size_t) (r1 - r0) * w);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* src = t.data + (size_t) r * w * (t.type == T_F32 ? 4 : 2);
            if (t.type == T_BF16)
                for (int i = 0; i < w; ++i) h[(size_t) (r - r0) * w + i] = bf16_to_f32_host(((const uint16_t*) src)[i]);
            else if (t.type == T_F16)
                for (int i = 0; i < w; ++i) h[(size_t) (r - r0) * w + i] = f16_to_f32_host(((const uint16_t*) src)[i]);
            else std::memcpy(&h[(size_t) (r - r0) * w], src, (size_t) w * 4);
        }
        return put<float>(h.data(), h.size() * 4, t.name.c_str());
    }
    // columns [c0, c1) of a 2-D BF16/F16/F32 tensor (row-major rows, each cut to the range)
    float* f32_cols(const Tensor& t, int c0, int c1) {
        if (t.type != T_BF16 && t.type != T_F16 && t.type != T_F32) throw std::runtime_error("vit: " + t.name + " is not BF16/F16/F32");
        const int w = (int) t.ne[0], rows = (int) t.ne[1], cw = c1 - c0;
        const size_t ib = t.type == T_F32 ? 4 : 2;
        std::vector<float> h((size_t) rows * cw);
        for (int r = 0; r < rows; ++r) {
            const uint8_t* src = t.data + ((size_t) r * w + c0) * ib;
            for (int i = 0; i < cw; ++i)
                h[(size_t) r * cw + i] = t.type == T_BF16 ? bf16_to_f32_host(((const uint16_t*) src)[i])
                                      : t.type == T_F16 ? f16_to_f32_host(((const uint16_t*) src)[i])
                                      : ((const float*) src)[i];
        }
        return put<float>(h.data(), h.size() * 4, t.name.c_str());
    }
};

// the geometry, checked once (card 0), every card's slice of it loaded after
void vit_check(const Gguf& g, int& n_out) {
    if (g.i("clip.vision.embedding_length") != VIT_H || g.i("clip.vision.block_count") != VIT_NL ||
        g.i("clip.vision.attention.head_count") != VIT_HEADS || g.i("clip.vision.feed_forward_length") != VIT_FF ||
        g.i("clip.vision.patch_size") != VIT_PATCH || g.i("clip.vision.spatial_merge_size") != VIT_MERGE)
        throw std::runtime_error("vit: the mmproj's geometry is not Qwen3.8-Flash-Next's vision tower's (1152 wide, "
                                 "27 layers, 16 heads, FF 4304, patch 16, merge 2)");
    const std::string& pt = g.s("clip.projector_type");
    if (pt != "qwen3vl_merger") throw std::runtime_error("vit: the mmproj's projector is " + pt + ", not qwen3vl_merger");
    if (std::fabs(g.num("clip.vision.attention.layer_norm_epsilon") - VIT_LN_EPS) > 1e-8)
        throw std::runtime_error("vit: the mmproj's LayerNorm eps is not 1e-6");
    if (g.t("v.position_embd.weight").ne[1] != 2304)
        throw std::runtime_error("vit: the position table is not 48x48");
    n_out = (int) g.i("clip.vision.projection_dim");
    if (n_out != N)
        throw std::runtime_error("vit: the mmproj projects to " + std::to_string(n_out) + ", the text model's width is " +
                                 std::to_string(N));
    if (g.t("mm.2.weight").ne[1] != n_out || g.t("mm.2.bias").ne[0] != n_out)
        throw std::runtime_error("vit: the merger's second matrix is not " + std::to_string(n_out) + " wide");
}

void vit_load_card(const Gguf& g, VitCard& c, int max_pos, int n_out) {
    VUp up{c};
    const int H = VIT_H, FF = VIT_FF;
    // the two convs summed into one [768][1152] f32 matrix: llama.cpp runs both on the same input and adds the
    // outputs, and a conv being linear, the summed weights give the same sum
    {
        const Tensor& w0 = g.t("v.patch_embd.weight");
        const Tensor& w1 = g.t("v.patch_embd.weight.1");
        const int64_t n = w0.ne[0] * w0.ne[1] * w0.ne[2] * w0.ne[3];
        if (w0.type != w1.type || w0.ne[0] != VIT_PATCH || w0.ne[1] != VIT_PATCH || w0.ne[2] != 3 || w0.ne[3] != H ||
            w1.ne[3] != H || (w0.type != T_BF16 && w0.type != T_F16 && w0.type != T_F32))
            throw std::runtime_error("vit: the patch convs are not two [16, 16, 3, 1152] BF16/F16/F32 tensors");
        auto val = [&](const Tensor& t, size_t i) -> float {
            const uint8_t* p = (const uint8_t*) t.data;
            if (t.type == T_BF16) { uint16_t v; std::memcpy(&v, p + i * 2, 2); return bf16_to_f32_host(v); }
            if (t.type == T_F16) { uint16_t v; std::memcpy(&v, p + i * 2, 2); return f16_to_f32_host(v); }
            float v; std::memcpy(&v, p + i * 4, 4); return v;
        };
        std::vector<float> sum((size_t) n);
        for (size_t i = 0; i < (size_t) n; ++i) sum[i] = val(w0, i) + val(w1, i);
        c.patch = up.put<float>(sum.data(), sum.size() * 4, "v.patch_embd (the convs summed)");
    }
    c.patch_bias = up.put<float>(g.t("v.patch_embd.bias").data, (size_t) H * 4, "v.patch_embd.bias");
    up.f32_check(g.t("v.patch_embd.bias"));
    c.pos_table = up.put<float>(g.t("v.position_embd.weight").data, (size_t) H * 2304 * 4, "v.position_embd.weight");
    up.f32_check(g.t("v.position_embd.weight"));
    // the per-layer f32 stacks (whole on every card): the LayerNorms and the biases the card's slices read
    {
        std::vector<float> ln1w((size_t) VIT_NL * H), ln1b(ln1w.size()), ln2w(ln1w.size()), ln2b(ln1w.size());
        std::vector<float> qkvb((size_t) VIT_NL * 3 * H), aob(ln1w.size()), ffb((size_t) VIT_NL * FF), dnb(ln1w.size());
        for (int l = 0; l < VIT_NL; ++l) {
            const std::string b = "v.blk." + std::to_string(l) + ".";
            for (int i = 0; i < H; ++i) {
                ln1w[(size_t) l * H + i] = f32_of(g.t(b + "ln1.weight"), i);
                ln1b[(size_t) l * H + i] = f32_of(g.t(b + "ln1.bias"), i);
                ln2w[(size_t) l * H + i] = f32_of(g.t(b + "ln2.weight"), i);
                ln2b[(size_t) l * H + i] = f32_of(g.t(b + "ln2.bias"), i);
                aob[(size_t) l * H + i] = f32_of(g.t(b + "attn_out.bias"), i);
                dnb[(size_t) l * H + i] = f32_of(g.t(b + "ffn_down.bias"), i);
            }
            for (int i = 0; i < 3 * H; ++i) qkvb[(size_t) l * 3 * H + i] = f32_of(g.t(b + "attn_qkv.bias"), i);
            for (int i = 0; i < FF; ++i) ffb[(size_t) l * FF + i] = f32_of(g.t(b + "ffn_up.bias"), i);
        }
        c.ln1w = up.put<float>(ln1w.data(), ln1w.size() * 4, "ln1.weight");
        c.ln1b = up.put<float>(ln1b.data(), ln1b.size() * 4, "ln1.bias");
        c.ln2w = up.put<float>(ln2w.data(), ln2w.size() * 4, "ln2.weight");
        c.ln2b = up.put<float>(ln2b.data(), ln2b.size() * 4, "ln2.bias");
        c.qkvb = up.put<float>(qkvb.data(), qkvb.size() * 4, "attn_qkv.bias");
        c.aob = up.put<float>(aob.data(), aob.size() * 4, "attn_out.bias");
        c.ffb = up.put<float>(ffb.data(), ffb.size() * 4, "ffn_up.bias");
        c.dnb = up.put<float>(dnb.data(), dnb.size() * 4, "ffn_down.bias");
    }
    c.post_lnw = up.put<float>(g.t("v.post_ln.weight").data, (size_t) H * 4, "v.post_ln.weight");
    c.post_lnb = up.put<float>(g.t("v.post_ln.bias").data, (size_t) H * 4, "v.post_ln.bias");
    up.f32_check(g.t("v.post_ln.weight"));
    up.f32_check(g.t("v.post_ln.bias"));
    // the fused QKV, the card's heads' rows of every layer stacked [27][nh*216][1152]: the file's rows are all Q
    // (0..1152), then all K, then all V, a head's triple at h*72 + {0, 1152, 2304}
    {
        const int rws = c.nh * 216;
        std::vector<float> hbuf((size_t) VIT_NL * rws * H);
        for (int l = 0; l < VIT_NL; ++l) {
            const Tensor& t = g.t("v.blk." + std::to_string(l) + ".attn_qkv.weight");
            if ((int) t.ne[0] != H || (int) t.ne[1] != 3 * H) throw std::runtime_error("vit: " + t.name + " is not [1152][3456]");
            if (t.type != T_BF16 && t.type != T_F16) throw std::runtime_error("vit: " + t.name + " is neither BF16 nor F16");
            const size_t ib = 2;
            for (int hh = 0; hh < c.nh; ++hh) {
                const int h = c.h0 + hh;
                for (int part = 0; part < 3; ++part)
                    for (int i = 0; i < 72; ++i) {
                        const int r = part * H + h * 72 + i;
                        const uint8_t* src = t.data + (size_t) r * H * ib;
                        float* dst = &hbuf[((size_t) l * rws + hh * 216 + part * 72 + i) * H];
                        if (t.type == T_BF16)
                            for (int k = 0; k < H; ++k) dst[k] = bf16_to_f32_host(((const uint16_t*) src)[k]);
                        else
                            for (int k = 0; k < H; ++k) dst[k] = f16_to_f32_host(((const uint16_t*) src)[k]);
                    }
            }
        }
        c.qkv = up.put<float>(hbuf.data(), hbuf.size() * 4, "attn_qkv.weight");
    }
    // attn_out: this card's heads' columns of every layer's [1152][1152], stacked [27][1152][nh*72]
    {
        const int cw = c.nh * 72;
        std::vector<float> hbuf((size_t) VIT_NL * H * cw);
        for (int l = 0; l < VIT_NL; ++l) {
            const Tensor& t = g.t("v.blk." + std::to_string(l) + ".attn_out.weight");
            if ((int) t.ne[0] != H || (int) t.ne[1] != H || (t.type != T_BF16 && t.type != T_F16))
                throw std::runtime_error("vit: " + t.name + " is not [1152][1152] BF16/F16");
            for (int r = 0; r < H; ++r) {
                const uint8_t* src = t.data + ((size_t) r * H + c.h0 * 72) * 2;
                float* dst = &hbuf[((size_t) l * H + r) * cw];
                if (t.type == T_BF16)
                    for (int k = 0; k < cw; ++k) dst[k] = bf16_to_f32_host(((const uint16_t*) src)[k]);
                else
                    for (int k = 0; k < cw; ++k) dst[k] = f16_to_f32_host(((const uint16_t*) src)[k]);
            }
        }
        c.attn_out = up.put<float>(hbuf.data(), hbuf.size() * 4, "attn_out.weight");
    }
    // ffn_up: the card's rows of the 4304, stacked [27][ffw][1152]; ffn_down: its columns, [27][1152][ffw]
    {
        const int rws = c.ff1 - c.ff0;
        std::vector<float> hbuf((size_t) VIT_NL * rws * H);
        for (int l = 0; l < VIT_NL; ++l) {
            const Tensor& t = g.t("v.blk." + std::to_string(l) + ".ffn_up.weight");
            if ((int) t.ne[0] != H || (int) t.ne[1] != FF || (t.type != T_BF16 && t.type != T_F16))
                throw std::runtime_error("vit: " + t.name + " is not [1152][4304] BF16/F16");
            for (int i = 0; i < rws; ++i) {
                const uint8_t* src = t.data + (size_t) (c.ff0 + i) * H * 2;
                float* dst = &hbuf[((size_t) l * rws + i) * H];
                if (t.type == T_BF16)
                    for (int k = 0; k < H; ++k) dst[k] = bf16_to_f32_host(((const uint16_t*) src)[k]);
                else
                    for (int k = 0; k < H; ++k) dst[k] = f16_to_f32_host(((const uint16_t*) src)[k]);
            }
        }
        c.ffn_up = up.put<float>(hbuf.data(), hbuf.size() * 4, "ffn_up.weight");
    }
    {
        const int cw = c.ff1 - c.ff0;
        std::vector<float> hbuf((size_t) VIT_NL * H * cw);
        for (int l = 0; l < VIT_NL; ++l) {
            const Tensor& t = g.t("v.blk." + std::to_string(l) + ".ffn_down.weight");
            if ((int) t.ne[0] != FF || (int) t.ne[1] != H || (t.type != T_BF16 && t.type != T_F16))
                throw std::runtime_error("vit: " + t.name + " is not [4304][1152] BF16/F16");
            for (int r = 0; r < H; ++r) {
                const uint8_t* src = t.data + ((size_t) r * FF + c.ff0) * 2;
                float* dst = &hbuf[((size_t) l * H + r) * cw];
                if (t.type == T_BF16)
                    for (int k = 0; k < cw; ++k) dst[k] = bf16_to_f32_host(((const uint16_t*) src)[k]);
                else
                    for (int k = 0; k < cw; ++k) dst[k] = f16_to_f32_host(((const uint16_t*) src)[k]);
            }
        }
        c.ffn_down = up.put<float>(hbuf.data(), hbuf.size() * 4, "ffn_down.weight");
    }
    // the merger: mm.0 the card's rows of 4608, mm.2 its columns of the output width
    c.mm0 = up.f32_rows(g.t("mm.0.weight"), c.mu0, c.mu1);
    c.mm2 = up.f32_cols(g.t("mm.2.weight"), c.mu0, c.mu1);
    {
        std::vector<float> b0((size_t) (c.mu1 - c.mu0));
        for (int i = 0; i < c.mu1 - c.mu0; ++i) b0[(size_t) i] = f32_of(g.t("mm.0.bias"), c.mu0 + i);
        c.mm0b = up.put<float>(b0.data(), b0.size() * 4, "mm.0.bias");
        c.mm2b = up.put<float>(g.t("mm.2.bias").data, (size_t) n_out * 4, "mm.2.bias");
    }
    // the forward's buffers: the ring's by-value sizing — a part of the residual sum (max_pos/NCARD rows of 1152)
    // is the widest thing that goes through a slot; the merger's (max_pos/4/NCARD rows of 2560) is 4/9 of that
    {
        const size_t mp = (size_t) max_pos, part = (mp + NCARD - 1) / NCARD * VIT_H, whole = mp * VIT_H;
        c.img = (float*) up.raw(3 * mp * VIT_PATCH * VIT_PATCH * 4, "img");
        c.x = (float*) up.raw(whole * 4, "x");
        c.tmp = (float*) up.raw(whole * 4, "tmp");
        c.cols = (float*) up.raw(mp * 768 * 4, "cols");
        c.qkvh = (float*) up.raw(mp * (size_t) c.nh * 216 * 4, "qkvh");
        c.att = (float*) up.raw(mp * (size_t) c.nh * 72 * 4, "att");
        c.ffn = (float*) up.raw(mp * (size_t) (c.ff1 - c.ff0) * 4, "ffn");
        c.mm0v = (float*) up.raw(mp / 4 * (size_t) (c.mu1 - c.mu0) * 4, "mm0v");
        c.addend = (float*) up.raw(whole * 4, "addend");
        for (int k = 0; k < 2; ++k) {
            c.send[k] = (float*) up.raw(part * 4, "ring send");
            c.ag[k] = (float*) up.raw(whole * 4, "ring ag");
            for (int h = 0; h < NCARD - 1; ++h) c.in[h][k] = (float*) up.raw(part * 4, "ring in");
        }
        c.flags = (unsigned*) up.raw(2 * pf_ring_steps(NCARD) * sizeof(unsigned), "ring flags");
        vck(cudaMemset(c.flags, 0, 2 * pf_ring_steps(NCARD) * sizeof(unsigned)), "ring flags");
    }
}

}  // namespace

// ---- the sums' ring, fp32 (the same plan the prompt path's pf_ring_dir runs, its f16 pack/add/out steps in fp32).
// The caller has written this card's partial of x's rows into c.addend (the whole row range, its values there); the
// plan's parts are rows of x, part p = rows [p*C/n, (p+1)*C/n). A sum's parity (two in flight: the attention's and
// the FFN's of one layer) picks the buffer set, and `seq` — bumped per sum — is the flag value, so a stale WAIT
// passes only on this sum's own signal
namespace {

constexpr int VIT_RING_STEPS = pf_ring_steps(NCARD);

void vit_ring_dir(VitCard& c, int g, float* x, int C, int width, unsigned seq, cudaStream_t s) {
    const int set = (int) (seq & 1u);
    auto r0 = [&](int k) { return k * C / NCARD; };
    auto rows = [&](int k) { return r0(k + 1) - r0(k); };
    CUstream cs = (CUstream) s;
    auto signal = [&](int step) {
        vcud(cuStreamWriteValue32(cs, (CUdeviceptr) (c.nflags + set * VIT_RING_STEPS + step), seq, 0), "ring flag");
    };
    auto wait = [&](int step) {
        vcud(cuStreamWaitValue32(cs, (CUdeviceptr) (c.flags + set * VIT_RING_STEPS + step), seq, CU_STREAM_WAIT_VALUE_GEQ),
             "ring wait");
    };
    auto copy = [&](float* dst, const float* src, int nrows) {
        if (nrows > 0) vck(cudaMemcpyPeerAsync(dst, c.next_dev, src, c.dev, (size_t) nrows * width * 4, s), "ring copy");
    };
    for (const PfRingOp& o : c.plan) {
        switch (o.k) {
            case PfRingOp::PACK:
                if (rows(o.part) > 0)
                    vck(cudaMemcpyAsync(c.send[set], c.addend + (size_t) r0(o.part) * width,
                                        (size_t) rows(o.part) * width * 4, cudaMemcpyDeviceToDevice, s),
                        "ring pack");
                break;
            case PfRingOp::ADD: {
                const int nr = rows(o.part);
                if (nr == 0) break;
                float* out = o.fin ? c.ag[set] + (size_t) r0(o.part) * width : c.send[set];
                k_vit_ring_add<<<vgrid((size_t) nr * width), 256, 0, s>>>(
                    c.in[o.hop][set], c.addend + (size_t) r0(o.part) * width, (size_t) nr * width, out);
                break;
            }
            case PfRingOp::SEND: copy(c.nin[o.hop][set], c.send[set], rows(o.part)); break;
            case PfRingOp::PASS: copy(c.nag[set] + (size_t) r0(o.part) * width, c.ag[set] + (size_t) r0(o.part) * width, rows(o.part)); break;
            case PfRingOp::SIGNAL: signal(o.step); break;
            case PfRingOp::WAIT: wait(o.step); break;
            case PfRingOp::OUT:
                vck(cudaMemcpyAsync(x, c.ag[set], (size_t) C * width * 4, cudaMemcpyDeviceToDevice, s), "ring out");
                break;
            default: throw std::runtime_error("vit: a ring step of unknown kind " + std::to_string((int) o.k));
        }
    }
}

// one sum: every card's partial (in its own addend) into x, on every card. The cards' host threads each enqueue
// their own direction; the flags pace them (a card's WAIT blocks its stream until the previous card's copies land)
struct VitRun {
    cudaStream_t st[NCARD] = {};
    cublasHandle_t blas[NCARD] = {};
    unsigned seq = 0;
    bool ready = false;
    void init(const Vit& v) {
        if (ready) return;
        for (int ci = 0; ci < NCARD; ++ci) {
            vck(cudaSetDevice(v.cards[ci].dev), "set device");
            vck(cudaStreamCreate(&st[ci]), "stream");
            vckb(cublasCreate(&blas[ci]), "create");
            vckb(cublasSetStream(blas[ci], st[ci]), "stream");
            vckb(cublasSetMathMode(blas[ci], CUBLAS_TENSOR_OP_MATH), "math mode");
        }
        ready = true;
    }
    ~VitRun() {
        if (!ready) return;
        for (int ci = 0; ci < NCARD; ++ci) {
            if (blas[ci]) cublasDestroy(blas[ci]);
            if (st[ci]) cudaStreamDestroy(st[ci]);
        }
    }
};
VitRun& vit_run() {
    static VitRun r;
    return r;
}

// out[t*ldo + r] = sum_i W[r][i] * x[t*ldi + i]: W f32 [rows][cols] (the contraction over cols), fp32 all the way —
// sm_70 (V100) has no BF16 GEMM (cublas status 13), so the weights were widened once at load, which is also more
// exact than the planned per-call activation widen (the oracle's own CPU path is fp32 throughout)
void vit_gemm(cublasHandle_t h, const float* W, int rows, int cols, const float* x, int C, int ldi, float* out,
              int ldo, cudaStream_t s) {
    if (ldi != cols) throw std::runtime_error("vit: a gemm's input rows are strided (the conv's 768-wide columns are the only one)");
    const float one = 1.f, zero = 0.f;
    vckb(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, cols, x, ldi, &zero, out, ldo), "gemm");
}

// one card's compute of a layer's attention half (up to its partial in addend): LN -> QKV+bias -> rope -> attention
// -> attn_out over its heads' columns. The ring sum and the residual run after every card's this-half is enqueued
void vit_attn_half(VitRun& R, int ci, VitCard& c, int l, int n_pos, int gx, int gy, cudaStream_t s) {
    const int H = VIT_H, nh = c.nh;
    k_vit_ln<<<n_pos, 128, 0, s>>>(c.x, c.ln1w + (size_t) l * H, c.ln1b + (size_t) l * H, c.tmp, H, VIT_LN_EPS);
    if (const char* dp = std::getenv("VIT_DEBUG_DUMP3"); dp && l == 0 && ci == 0) {
        vck(cudaStreamSynchronize(s), "dbg sync");
        std::vector<float> h((size_t) n_pos * H);
        vck(cudaMemcpy(h.data(), c.tmp, h.size() * 4, cudaMemcpyDeviceToHost), "dbg ln1");
        std::FILE* f = std::fopen((std::string(dp) + "-ln1.f32").c_str(), "wb");
        if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
    }
    vit_gemm(R.blas[ci], c.qkv + (size_t) l * nh * 216 * H, nh * 216, H, c.tmp, n_pos, H, c.qkvh, nh * 216, s);
    k_vit_qkv_bias<<<vgrid((size_t) n_pos * nh * 216), 256, 0, s>>>(c.qkvh, c.qkvb + (size_t) l * 3 * H, c.h0, nh,
                                                                    (size_t) n_pos * nh * 216);
    if (const char* dp = std::getenv("VIT_DEBUG_DUMP3"); dp && l == 0 && ci == 0) {
        vck(cudaStreamSynchronize(s), "dbg sync");
        std::vector<float> h((size_t) n_pos * nh * 216);
        vck(cudaMemcpy(h.data(), c.qkvh, h.size() * 4, cudaMemcpyDeviceToHost), "dbg qkv");
        std::FILE* f = std::fopen((std::string(dp) + "-qkv.f32").c_str(), "wb");
        if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
    }
    k_vit_rope<<<vgrid((size_t) n_pos * 36), 256, 0, s>>>(c.qkvh, gx, n_pos, nh);
    if (const char* dp = std::getenv("VIT_DEBUG_DUMP3"); dp && l == 0 && ci == 0) {
        vck(cudaStreamSynchronize(s), "dbg sync");
        std::vector<float> h((size_t) n_pos * nh * 216);
        vck(cudaMemcpy(h.data(), c.qkvh, h.size() * 4, cudaMemcpyDeviceToHost), "dbg qkv rope");
        std::FILE* f = std::fopen((std::string(dp) + "-qkv-rope.f32").c_str(), "wb");
        if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
    }
    k_vit_attn<<<dim3(n_pos, nh), 128, 0, s>>>(c.qkvh, c.att, n_pos, nh);
    if (const char* dp = std::getenv("VIT_DEBUG_DUMP3"); dp && l == 0 && ci == 0) {
        vck(cudaStreamSynchronize(s), "dbg sync");
        std::vector<float> h((size_t) n_pos * nh * 72);
        vck(cudaMemcpy(h.data(), c.att, h.size() * 4, cudaMemcpyDeviceToHost), "dbg att");
        std::FILE* f = std::fopen((std::string(dp) + "-att.f32").c_str(), "wb");
        if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
    }
    vit_gemm(R.blas[ci], c.attn_out + (size_t) l * H * nh * 72, H, nh * 72, c.att, n_pos, nh * 72, c.addend, H, s);
}

// one card's FFN half: LN -> up (its rows) + bias + GELU -> down (its columns) into addend
void vit_ffn_half(VitRun& R, int ci, VitCard& c, int l, int n_pos, cudaStream_t s) {
    const int H = VIT_H, fw = c.ff1 - c.ff0;
    k_vit_ln<<<n_pos, 128, 0, s>>>(c.x, c.ln2w + (size_t) l * H, c.ln2b + (size_t) l * H, c.tmp, H, VIT_LN_EPS);
    vit_gemm(R.blas[ci], c.ffn_up + (size_t) l * (size_t) fw * H, fw, H, c.tmp, n_pos, H, c.ffn, fw, s);
    k_vit_bias_row<<<vgrid((size_t) n_pos * fw), 256, 0, s>>>(c.ffn, c.ffb + (size_t) l * VIT_FF + c.ff0,
                                                              (size_t) n_pos * fw, fw);
    k_vit_gelu<<<vgrid((size_t) n_pos * fw), 256, 0, s>>>(c.ffn, (size_t) n_pos * fw);
    vit_gemm(R.blas[ci], c.ffn_down + (size_t) l * (size_t) H * fw, H, fw, c.ffn, n_pos, fw, c.addend, H, s);
}

}  // namespace

// ---- the forward. One host thread drives every card's stream (the ops are async; the sums' flags pace the cards
// across the streams, so the enqueue order does not matter). x is the residual, replicated: every card computes the
// same replicated steps (conv, LNs) and its own slice of the split ones, the ring sums leave the summed rows on
// every card, and card 0's copy of the final embeddings comes down.
std::vector<Vit::Grid> Vit::encode_file(const std::vector<std::string>& paths, const std::string& out_path) {
    VitRun& R = vit_run();
    R.init(*this);
    std::vector<Grid> grids;
    std::vector<float> rows;   // every picture's output rows, concatenated
    for (const std::string& path : paths) {
        const VitImage im = vit_preprocess(path, max_pixels);
        const int gx = im.grid_x(), gy = im.grid_y(), n_pos = gx * gy;
        if (n_pos % 4 != 0 || gx % 2 != 0 || gy % 2 != 0)
            throw std::runtime_error("vit: " + path + "'s grid " + std::to_string(gx) + "x" + std::to_string(gy) +
                                     " is not whole 2x2 cells");
        if (n_pos > max_pos)
            throw std::runtime_error("vit: " + path + " needs " + std::to_string(n_pos) + " patches, the tower holds " +
                                     std::to_string(max_pos) + " (raise --vision-max-tokens)");
        // the pixels up to every card (the replicated steps read them)
        for (int ci = 0; ci < NCARD; ++ci) {
            VitCard& c = cards[ci];
            vck(cudaSetDevice(c.dev), "set device");
            vck(cudaMemcpyAsync(c.img, im.px.data(), im.px.size() * 4, cudaMemcpyHostToDevice, R.st[ci]), "pixels");
        }
        // conv + patch bias + the quad order + the position table, every card on its own stream. Every per-card
        // loop below sets the current device first: a single host thread enqueues for all the cards, and a launch
        // (or a cublas call) on a stream/handle of a card that is not current fails (cublas 13, EXECUTION_FAILED)
        const int H = VIT_H;
        if (const char* dp = std::getenv("VIT_DEBUG_DUMP")) {
            std::FILE* f = std::fopen((std::string(dp) + "-img.f32").c_str(), "wb");
            if (f) { std::fwrite(im.px.data(), 4, im.px.size(), f); std::fclose(f); }
        }
        for (int ci = 0; ci < NCARD; ++ci) {
            VitCard& c = cards[ci];
            vck(cudaSetDevice(c.dev), "set device");
            cudaStream_t s = R.st[ci];
            k_vit_conv_im2col<<<vgrid((size_t) n_pos), 256, 0, s>>>(c.img, c.cols, im.nx, im.ny, gx);
            vit_gemm(R.blas[ci], c.patch, H, 3 * VIT_PATCH * VIT_PATCH, c.cols, n_pos, 3 * VIT_PATCH * VIT_PATCH, c.x,
                     H, s);
            k_vit_bias_row<<<vgrid((size_t) n_pos * H), 256, 0, s>>>(c.x, c.patch_bias, (size_t) n_pos * H, H);
            k_vit_quad_reorder<<<vgrid((size_t) n_pos * H), 256, 0, s>>>(c.x, c.tmp, gx, gy);
            vck(cudaMemcpyAsync(c.x, c.tmp, (size_t) n_pos * H * 4, cudaMemcpyDeviceToDevice, s), "quad copy");
            k_vit_pos_add<<<n_pos, 128, 0, s>>>(c.x, c.pos_table, gx, gy);
            if (const char* dp = std::getenv("VIT_DEBUG_DUMP")) {
                static bool dumped = false;   // the first picture's layer-0 input, once
                if (!dumped) {
                    dumped = true;
                    vck(cudaStreamSynchronize(s), "dbg sync");
                    std::vector<float> h((size_t) n_pos * H);
                    vck(cudaMemcpy(h.data(), c.x, h.size() * 4, cudaMemcpyDeviceToHost), "dbg x0");
                    std::FILE* f = std::fopen((std::string(dp) + "-x0.f32").c_str(), "wb");
                    if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
                }
            }
        }
        // the layers: per layer, every card's attention half, then the sum, then the residual; the FFN half likewise
        for (int l = 0; l < VIT_NL; ++l) {
            const unsigned seq_a = ++R.seq, seq_f = ++R.seq;
            for (int ci = 0; ci < NCARD; ++ci) {
                VitCard& c = cards[ci];
            vck(cudaSetDevice(c.dev), "set device");
                vit_attn_half(R, ci, c, l, n_pos, gx, gy, R.st[ci]);
            }
            for (int ci = 0; ci < NCARD; ++ci) {
                VitCard& c = cards[ci];
            vck(cudaSetDevice(c.dev), "set device");
                vit_ring_dir(c, c.g, c.tmp, n_pos, H, seq_a, R.st[ci]);   // the sum into tmp
                k_vit_add_bias<<<vgrid((size_t) n_pos * H), 256, 0, R.st[ci]>>>(c.x, c.tmp, c.aob + (size_t) l * H,
                                                                                (size_t) n_pos * H, H);
            }
            if (const char* dp = std::getenv("VIT_DEBUG_DUMP2"); dp && l == 0) {   // the ring's insides, once
                for (int ci = 0; ci < NCARD; ++ci) {
                    vck(cudaSetDevice(cards[ci].dev), "set device");
                    vck(cudaStreamSynchronize(R.st[ci]), "dbg sync");
                    std::vector<float> h((size_t) n_pos * H);
                    vck(cudaMemcpy(h.data(), cards[ci].addend, h.size() * 4, cudaMemcpyDeviceToHost), "dbg addend-a");
                    // the addend now holds the ffn half's input? no: the ffn half has not run yet — it is still
                    // the attention partial. name it by the card
                    std::FILE* f = std::fopen((std::string(dp) + "-addend-a" + std::to_string(ci) + ".f32").c_str(), "wb");
                    if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
                    vck(cudaMemcpy(h.data(), cards[ci].tmp, h.size() * 4, cudaMemcpyDeviceToHost), "dbg tmp-a");
                    f = std::fopen((std::string(dp) + "-tmp-a" + std::to_string(ci) + ".f32").c_str(), "wb");
                    if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
                    vck(cudaMemcpy(h.data(), cards[ci].ag[seq_a & 1], h.size() * 4, cudaMemcpyDeviceToHost), "dbg ag-a");
                    f = std::fopen((std::string(dp) + "-ag-a" + std::to_string(ci) + ".f32").c_str(), "wb");
                    if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
                }
            }
            for (int ci = 0; ci < NCARD; ++ci) {
                VitCard& c = cards[ci];
            vck(cudaSetDevice(c.dev), "set device");
                vit_ffn_half(R, ci, c, l, n_pos, R.st[ci]);
            }
            for (int ci = 0; ci < NCARD; ++ci) {
                VitCard& c = cards[ci];
            vck(cudaSetDevice(c.dev), "set device");
                vit_ring_dir(c, c.g, c.tmp, n_pos, H, seq_f, R.st[ci]);
                k_vit_add_bias<<<vgrid((size_t) n_pos * H), 256, 0, R.st[ci]>>>(c.x, c.tmp, c.dnb + (size_t) l * H,
                                                                                (size_t) n_pos * H, H);
            }
            if (const char* dp = std::getenv("VIT_DEBUG_DUMP2"); dp) {
                for (int ci = 0; ci < NCARD; ++ci) {
                    vck(cudaSetDevice(cards[ci].dev), "set device");
                    vck(cudaStreamSynchronize(R.st[ci]), "dbg sync");
                    std::vector<float> h((size_t) n_pos * H);
                    vck(cudaMemcpy(h.data(), cards[ci].addend, h.size() * 4, cudaMemcpyDeviceToHost), "dbg addend-f");
                    std::FILE* f = std::fopen((std::string(dp) + "-addend-f" + std::to_string(ci) + ".f32").c_str(), "wb");
                    if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
                    vck(cudaMemcpy(h.data(), cards[ci].tmp, h.size() * 4, cudaMemcpyDeviceToHost), "dbg tmp-f");
                    f = std::fopen((std::string(dp) + "-tmp-f" + std::to_string(ci) + ".f32").c_str(), "wb");
                    if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
                }
            }
            if (const char* dp = std::getenv("VIT_DEBUG_DUMP"); dp && l == 0) {   // the first layer's output, once
                vck(cudaSetDevice(cards[0].dev), "set device");
                vck(cudaStreamSynchronize(R.st[0]), "dbg sync");
                std::vector<float> h((size_t) n_pos * H);
                vck(cudaMemcpy(h.data(), cards[0].x, h.size() * 4, cudaMemcpyDeviceToHost), "dbg x1");
                std::FILE* f = std::fopen((std::string(dp) + "-x1.f32").c_str(), "wb");
                if (f) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
            }
        }
        // post_ln then the merger: cells = tokens/4, a cell's 4608 input channels = its 4 tokens' rows contiguous
        // (the quad order); mm.0 the card's rows (+bias) -> GELU -> mm.2 its columns -> the sum (+bias) = embeddings
        const int n_cells = n_pos / 4;
        for (int ci = 0; ci < NCARD; ++ci) {
            VitCard& c = cards[ci];
            vck(cudaSetDevice(c.dev), "set device");
            cudaStream_t s = R.st[ci];
            k_vit_ln<<<n_pos, 128, 0, s>>>(c.x, c.post_lnw, c.post_lnb, c.tmp, H, VIT_LN_EPS);
            vit_gemm(R.blas[ci], c.mm0, c.mu1 - c.mu0, VIT_MERGER_IN, c.tmp, n_cells, VIT_MERGER_IN, c.mm0v,
                     c.mu1 - c.mu0, s);
            k_vit_bias_row<<<vgrid((size_t) n_cells * (c.mu1 - c.mu0)), 256, 0, s>>>(c.mm0v, c.mm0b,
                                                                                     (size_t) n_cells * (c.mu1 - c.mu0),
                                                                                     c.mu1 - c.mu0);
            k_vit_gelu<<<vgrid((size_t) n_cells * (c.mu1 - c.mu0)), 256, 0, s>>>(c.mm0v, (size_t) n_cells * (c.mu1 - c.mu0));
            vit_gemm(R.blas[ci], c.mm2, n_out, c.mu1 - c.mu0, c.mm0v, n_cells, c.mu1 - c.mu0, c.addend, n_out, s);
        }
        const unsigned seq_m = ++R.seq;
        for (int ci = 0; ci < NCARD; ++ci) {
            VitCard& c = cards[ci];
            vck(cudaSetDevice(c.dev), "set device");
            vit_ring_dir(c, c.g, c.tmp, n_cells, n_out, seq_m, R.st[ci]);
            // the embeddings = the sum + mm.2's bias — a replacement, not a residual (ggml's build_ffn output): x
            // holds the post_ln input at this point, so it is overwritten, not added to
            k_vit_out_bias<<<vgrid((size_t) n_cells * n_out), 256, 0, R.st[ci]>>>(c.x, c.tmp, c.mm2b,
                                                                                 (size_t) n_cells * n_out, n_out);
        }
        // card 0's embeddings down (every card holds the same sum)
        vck(cudaSetDevice(cards[0].dev), "set device");
        vck(cudaStreamSynchronize(R.st[0]), "the tower's forward");
        rows.resize(rows.size() + (size_t) n_cells * n_out);
        vck(cudaMemcpy(rows.data() + rows.size() - (size_t) n_cells * n_out, cards[0].x,
                       (size_t) n_cells * n_out * 4, cudaMemcpyDeviceToHost), "embeddings down");
        grids.push_back({n_cells, gx / 2, gy / 2});
    }
    // the SVE1 file: every picture's records in order, the engine's GENI (attach_images) reads exactly these
    std::FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) throw std::runtime_error("vit: cannot write " + out_path);
    size_t at = 0;
    for (const Grid& g : grids) {
        const int32_t hdr[5] = {0x31455653, g.n, g.nx, g.ny, n_out};
        const size_t cnt = (size_t) g.n * n_out;
        if (std::fwrite(hdr, sizeof hdr, 1, f) != 1 || std::fwrite(rows.data() + at, 4, cnt, f) != cnt) {
            std::fclose(f);
            throw std::runtime_error("vit: short write to " + out_path);
        }
        at += cnt;
    }
    std::fclose(f);
    return grids;
}

// ---- the constructor / destructor
Vit::Vit(const Model& m, const std::string& mmproj, int max_tokens) {
    if (NCARD != 2 && NCARD != 4)
        throw std::runtime_error("vit: the tower's 16 heads and 4608 merger rows split evenly over 2 or 4 cards, not " +
                                 std::to_string(NCARD));
    Gguf g(mmproj);
    int n_out0 = 0;
    vit_check(g, n_out0);
    max_pixels = max_tokens > 0 ? max_tokens * VIT_PATCH * VIT_PATCH * VIT_MERGE * VIT_MERGE : 1024 * 32 * 32;
    max_pos = max_tokens > 0 ? max_tokens * 4 : 4096;   // a merged token is 4 patches
    cards.resize(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        cards[c].dev = m.cards[c].dev;
        cards[c].g = c;
        cards[c].h0 = c * VIT_HEADS / NCARD;
        cards[c].nh = VIT_HEADS / NCARD;
        cards[c].ff0 = c * VIT_FF / NCARD;
        cards[c].ff1 = (c + 1) * VIT_FF / NCARD;
        cards[c].mu0 = c * VIT_MERGER_IN / NCARD;
        cards[c].mu1 = (c + 1) * VIT_MERGER_IN / NCARD;
    }
    std::vector<std::thread> th;
    std::vector<std::string> errs(NCARD);
    for (int c = 0; c < NCARD; ++c)
        th.emplace_back([&, c] {
            try {
                vck(cudaSetDevice(cards[c].dev), "set device");
                vit_load_card(g, cards[c], max_pos, n_out0);
            } catch (const std::exception& e) {
                errs[c] = e.what();
            }
        });
    for (auto& t : th) t.join();
    for (int c = 0; c < NCARD; ++c)
        if (!errs[c].empty()) {
            for (VitCard& cc : cards) {
                cudaSetDevice(cc.dev);
                for (void* p : cc.allocs) cudaFree(p);
            }
            cards.clear();
            throw std::runtime_error("vit card " + std::to_string(c) + ": " + errs[c]);
        }
    n_out = n_out0;
    // the ring wiring, as prefill.cu wires PfRing: this card's plan, the next card's receive slots and flags
    for (int c = 0; c < NCARD; ++c) {
        VitCard& v = cards[c];
        const VitCard& n = cards[(c + 1) % NCARD];
        for (int k = 0; k < 2; ++k) {
            for (int h = 0; h < NCARD - 1; ++h) v.nin[h][k] = n.in[h][k];
            v.nag[k] = n.ag[k];
        }
        v.nflags = n.flags;
        v.next_dev = n.dev;
        v.plan = pf_ring_plan(NCARD, c);
    }
    std::fprintf(stderr, "vit: the tower is up on %d cards: %d patches at most (%d merged tokens), %.0f MiB a card\n",
                 NCARD, max_pos, max_pos / 4, cards[0].bytes / 1048576.0);
}

Vit::~Vit() {
    for (VitCard& c : cards) {
        cudaSetDevice(c.dev);
        for (void* p : c.allocs) cudaFree(p);
    }
}

}  // namespace fnx
