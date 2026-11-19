// The host half of the picture path (vit_host.hpp): stb_image decode and llama.cpp's qwen3vl preprocessing, carried
// over as is (mtmd-image.cpp's resize_pillow and calc_size_preserved_ratio, clip.cpp's normalize). The fixed-point
// bicubic is bit-exact by construction; the only float steps are the pixel /255 and the (v - 0.5) / 0.5, both
// single-op f32 as in clip-impl.h's from_u8 / normalize. Plain C++, no CUDA: the same code runs in tools/vit_test
// against the oracle.
#include "vit_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vit_def.hpp"

namespace {

// llama.cpp reads with stb_image through mtmd's helper; the front end has already turned anything exotic into PNG
// and upright, so the four formats stb reads are the whole surface. stb is header-only and namespace-free: keep it
// inside this translation unit's include.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PIC
#define STBI_NO_PSD
#define STBI_NO_PNM
#include "stb_image.h"

// ggml's clip uses these to keep 8-bit resampling deterministic; the numbers are Pillow's
constexpr int PRECISION_BITS = 32 - 8 - 2;   // 22: 8-bit pixels, 2 bits of headroom
constexpr double BICUBIC_A = -0.5;           // Pillow's a; PyTorch/ggml use -0.75 — not this one

double resample_filter(double x) {   // the bicubic kernel, |x| < 2
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((BICUBIC_A + 2.0) * x - (BICUBIC_A + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * BICUBIC_A;
    return 0.0;
}

// one dimension's fixed-point filter: per output pixel the input range and the weights (mtmd-image.cpp,
// precompute_weights). inSize -> outSize; bounds[xx] = {xmin, xcnt}; weights [xx*ksize + x] in 22-bit fixed point
int precompute_weights(int inSize, int outSize, std::vector<int>& bounds, std::vector<int32_t>& weights) {
    const double filter_support = 2.0;   // bicubic
    double filterscale = (double) inSize / outSize;
    if (filterscale < 1.0) filterscale = 1.0;
    const double support = filter_support * filterscale;
    const int ksize = (int) std::ceil(support) * 2 + 1;
    bounds.assign((size_t) outSize * 2, 0);
    std::vector<double> pre((size_t) outSize * ksize);
    for (int xx = 0; xx < outSize; ++xx) {
        const double center = (xx + 0.5) * ((double) inSize / outSize);
        const double ss = 1.0 / filterscale;
        int xmin = (int) (center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int) (center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double w = resample_filter((x + xmin - center + 0.5) * ss);
            pre[(size_t) xx * ksize + x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; ++x)
            if (ww != 0.0) pre[(size_t) xx * ksize + x] /= ww;
        for (int x = xmax; x < ksize; ++x) pre[(size_t) xx * ksize + x] = 0;
        bounds[(size_t) xx * 2] = xmin;
        bounds[(size_t) xx * 2 + 1] = xmax;
    }
    const double fxp = std::ldexp(1.0, PRECISION_BITS);
    weights.resize((size_t) outSize * ksize);
    for (size_t i = 0; i < weights.size(); ++i) {
        // Pillow adds +/- 0.5 then truncates toward zero; std::round would round twice
        const double rounded = pre[i] * fxp + (pre[i] < 0 ? -0.5 : 0.5);
        weights[i] = (int32_t) rounded;
    }
    return ksize;
}

uint8_t clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : (uint8_t) v; }

// the two resampling passes (mtmd-image.cpp): horizontal first, then vertical over whole rows
std::vector<uint8_t> resample_h(const uint8_t* src, int in_nx, int in_ny, int out_nx, int ksize,
                                const std::vector<int>& bounds, const std::vector<int32_t>& weights) {
    std::vector<uint8_t> out((size_t) out_nx * in_ny * 3);
    for (int yy = 0; yy < in_ny; ++yy) {
        const uint8_t* srow = src + (size_t) yy * in_nx * 3;
        uint8_t* drow = out.data() + (size_t) yy * out_nx * 3;
        for (int xx = 0; xx < out_nx; ++xx) {
            const int xmin = bounds[(size_t) xx * 2], xcnt = bounds[(size_t) xx * 2 + 1];
            const int32_t* k = &weights[(size_t) xx * ksize];
            const uint8_t* p = srow + (size_t) xmin * 3;
            int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
            for (int x = 0; x < xcnt; ++x) {
                s0 += p[0] * k[x];
                s1 += p[1] * k[x];
                s2 += p[2] * k[x];
                p += 3;
            }
            drow[(size_t) xx * 3] = clip8(s0 >> PRECISION_BITS);
            drow[(size_t) xx * 3 + 1] = clip8(s1 >> PRECISION_BITS);
            drow[(size_t) xx * 3 + 2] = clip8(s2 >> PRECISION_BITS);
        }
    }
    return out;
}

std::vector<uint8_t> resample_v(const uint8_t* src, int in_nx, int out_ny, int ksize, const std::vector<int>& bounds,
                                const std::vector<int32_t>& weights) {
    const size_t row = (size_t) in_nx * 3;
    std::vector<uint8_t> out(row * out_ny);
    std::vector<int32_t> acc(row);
    for (int yy = 0; yy < out_ny; ++yy) {
        const int ymin = bounds[(size_t) yy * 2], ycnt = bounds[(size_t) yy * 2 + 1];
        const int32_t* k = &weights[(size_t) yy * ksize];
        std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));
        for (int y = 0; y < ycnt; ++y) {
            const uint8_t* srow = src + (size_t) (ymin + y) * row;
            const int32_t w = k[y];
            for (size_t i = 0; i < row; ++i) acc[i] += srow[i] * w;
        }
        uint8_t* drow = out.data() + (size_t) yy * row;
        for (size_t i = 0; i < row; ++i) drow[i] = clip8(acc[i] >> PRECISION_BITS);
    }
    return out;
}

// llama.cpp's smart_resize (img_tool::calc_size_preserved_ratio): align to align_size with round-half-even, then the
// min/max pixel clamps (the max one shrinks with floor, the min one grows with ceil)
void calc_size(int width, int height, int align, int min_pixels, int max_pixels, int& w_bar, int& h_bar) {
    auto round_by = [&](float x) { return (int) std::nearbyint(x / align) * align; };   // Python round(): half-even
    auto ceil_by = [&](float x) { return (int) std::ceil(x / align) * align; };
    auto floor_by = [&](float x) { return (int) std::floor(x / align) * align; };
    w_bar = std::max(align, round_by((float) width));
    h_bar = std::max(align, round_by((float) height));
    if (max_pixels > 0 && (int64_t) h_bar * w_bar > max_pixels) {
        const float beta = std::sqrt((float) ((int64_t) height * width) / max_pixels);
        h_bar = std::max(align, floor_by(height / beta));
        w_bar = std::max(align, floor_by(width / beta));
    } else if (min_pixels > 0 && (int64_t) h_bar * w_bar < min_pixels) {
        const float beta = std::sqrt((float) min_pixels / ((int64_t) height * width));
        h_bar = std::max(align, ceil_by(height * beta));
        w_bar = std::max(align, ceil_by(width * beta));
    }
}

}  // namespace

namespace fnx {

VitImage vit_preprocess(const std::string& path, int max_pixels) {
    int w = 0, h = 0, comp = 0;
    // stb decodes to 3 or 4 channels on request; mtmd's helper asks for 3 (RGB), any alpha dropped on the floor
    unsigned char* rgb = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!rgb) throw std::runtime_error("vit: cannot read " + path + " (" + (stbi_failure_reason() ? stbi_failure_reason() : "?") + ")");
    if (w <= 0 || h <= 0) {
        stbi_image_free(rgb);
        throw std::runtime_error("vit: " + path + " decodes to nothing");
    }
    // the tower's caps: align 32 (patch 16 x merge 2), min 8 image tokens, max max_pixels (llama.cpp's default caps:
    // 8 and 4096 tokens = 8192 and 4194304 pixels; the count is the merged token count, so the pixel caps are
    // token_count * 256)
    const int align = VIT_PATCH * VIT_MERGE;   // 32
    int min_pixels = 8 * align * align, mp = max_pixels;
    if (mp <= 0) mp = 4096 * align * align;
    int tw = 0, th = 0;
    calc_size(w, h, align, min_pixels, mp, tw, th);
    if (tw < align || th < align || (int64_t) tw * th <= 0) {
        stbi_image_free(rgb);
        throw std::runtime_error("vit: " + path + " resizes to nothing (" + std::to_string(tw) + "x" + std::to_string(th) + ")");
    }
    std::vector<uint8_t> px;
    if (tw != w || th != h) {   // PAD_NONE: a direct resize, no letterbox
        std::vector<int> wb, hb;
        std::vector<int32_t> ww, hw;
        const int kx = precompute_weights(w, tw, wb, ww);
        const int ky = precompute_weights(h, th, hb, hw);
        px = resample_v(resample_h(rgb, w, h, tw, kx, wb, ww).data(), tw, th, ky, hb, hw);
    } else {
        px.assign(rgb, rgb + (size_t) w * h * 3);
    }
    stbi_image_free(rgb);
    // channel-planar, /255 then (v - 0.5) / 0.5 (clip-impl.h's from_u8 + normalize with mean = std = 0.5)
    VitImage im;
    im.nx = tw;
    im.ny = th;
    const size_t n = (size_t) tw * th;
    im.px.resize(3 * n);
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            const float v = (float) px[i * 3 + c] / 255.0f;
            im.px[(size_t) c * n + i] = (v - 0.5f) / 0.5f;
        }
    }
    return im;
}

}  // namespace fnx
