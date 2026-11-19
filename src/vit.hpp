// The picture path's vision tower (the mmproj of Qwen3.8-Flash-Next, qwen3vl_merger) split across the cards the way
// the text model is, in-process: no llama.cpp runtime, no second process, no second CUDA context. One picture at a
// time (a request's images encode one after another, before its prompt is read), so the forward is a plain
// prefill-shaped batch over the picture's patches: no windows, no MAXT, no graphs. The mmproj is mapped read-only
// (Gguf), its weights uploaded BF16 exactly as the file holds them (the oracle ran on the same bytes), the residual
// stream, the attention and the sums fp32.
//
// The geometry is fixed (checked at load): hidden 1152, 27 layers, 16 heads of 72, FF 4304, GELU (tanh), LayerNorm
// with bias, fused QKV 3456 (Q rows 0..1152, K 1152..2304, V 2304..3456), patch 16, merge 2. The split over the
// cards (every width divides evenly at 2 or 4 cards):
//   replicated: patch conv (both summed), position table, LayerNorms, biases, the residual stream
//   QKV        : rows by head (a card takes heads [c*16/n, (c+1)*16/n) — its Q, K and V rows alike)
//   attn_out   : columns by head; the cards' partials summed (ring, fp32)
//   ffn_up     : rows (the card's slice of 4304); ffn_down: columns, summed (ring, fp32)
//   mm.0       : rows of 4608; mm.2: columns, summed (ring, fp32)
// Attention is per-head independent, so heads on different cards never exchange K/V: the only cross-card traffic is
// the sums — 2 a layer + 1 for the merger — on the same copy-engine ring plan the prompt path uses
// (pf_ring_plan.hpp), with this tower's own fp32 buffers and flags (the text engine's rings are untouched).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model.hpp"
#include "pf_ring_plan.hpp"
#include "vit_def.hpp"

namespace fnx {

// one card's slice of the tower (vit.cu's loader fills it; the buffers live as long as the Vit)
struct VitCard {
    int dev = 0, g = 0;                // the device, and this card's ring position
    int h0 = 0, nh = 0;                // its heads [h0, h0 + nh) of the 16
    int ff0 = 0, ff1 = 0;              // its slice of the ViT FF width
    int mu0 = 0, mu1 = 0;              // its rows of the merger's first matrix (of 4608)
    // weights: the wide ones f32 [rows][cols] with the layers stacked ([layer][local rows]); the small ones f32.
    // V100 (sm_70) has no BF16 GEMM: the weights are widened once at load (exact for BF16/F16) and every GEMM is
    // cublasSgemm, fp32 all the way — strictly more exact than the planned bf16 widen-per-call
    float *qkv = nullptr, *attn_out = nullptr, *ffn_up = nullptr, *ffn_down = nullptr, *mm0 = nullptr, *mm2 = nullptr;
    float* patch = nullptr;            // the two convs summed, [768][1152]
    float* patch_bias = nullptr;       // v.patch_embd.bias, [1152]
    float* pos_table = nullptr;        // [1152][48*48], column x + y*48
    float *ln1w = nullptr, *ln1b = nullptr, *ln2w = nullptr, *ln2b = nullptr;   // [27][1152]
    float *qkvb = nullptr;             // [27][3456], whole (the card reads its heads' rows)
    float *aob = nullptr, *dnb = nullptr;                                       // [27][1152]
    float *ffb = nullptr;              // [27][4304], whole (the card reads its slice)
    float *post_lnw = nullptr, *post_lnb = nullptr;                             // [1152]
    float *mm0b = nullptr, *mm2b = nullptr;                                      // mm.0's slice [mu1-mu0], mm.2's [n_out]
    // the forward's buffers (fp32 unless said; one set per card, sized for the worst picture)
    float* img = nullptr;              // the picture's pixels, [3][ny][nx] as uploaded
    float *x = nullptr, *tmp = nullptr;   // the residual stream and the LN/staging row-mate, [max_pos][1152]
    float* cols = nullptr;             // the conv's im2col columns, [max_pos][768]
    float* qkvh = nullptr;             // [max_pos][nh*216]: a head's Q, K, V rows contiguous
    float* att = nullptr;              // [max_pos][nh*72]
    float* ffn = nullptr;              // [max_pos][ff1-ff0]
    float* mm0v = nullptr;             // [max_pos/4][mu1-mu0]
    // the sums' ring (fp32): this card's partial of the rows being summed, the plan's buffers (two sets by parity,
    // a receive slot a hop), and the next card's receive slots and flags (wired after every card exists, as PfRing)
    float* addend = nullptr;           // [max_pos][1152] (the merger's is [max_pos/4][2560], smaller in values)
    float* send[2] = {};               // a part's rows, [part values]
    float* in[NCARD > 1 ? NCARD - 1 : 1][2] = {};   // this card's receive slots, a hop's
    float* ag[2] = {};                 // the gathered sum, [max_pos][1152]
    unsigned* flags = nullptr;         // [2 sets][pf_ring_steps(NCARD)], set by the previous card's stream
    float* nin[NCARD > 1 ? NCARD - 1 : 1][2] = {};
    float* nag[2] = {};
    unsigned* nflags = nullptr;
    int next_dev = 0;
    std::vector<PfRingOp> plan;
    std::vector<void*> allocs;         // freed with the tower
    size_t bytes = 0;
};

// the tower over the cards. Built after the Model (it checks the projector's output width against fnx::N and takes
// the devices from the Model's cards); encode_file never runs while a decode window or a prompt read is on the
// cards (the engine's command loop serializes it, as it does the requests).
struct Vit {
    std::vector<VitCard> cards;
    int n_out = 0;       // the projector's output width (must be fnx::N)
    int max_pos = 0;     // the largest patch count the buffers hold
    int max_pixels = 0;  // the preprocessing cap the tower accepts (a bigger picture is an error, not a silent clip)
    // mmproj: the projector file; max_tokens: the merged-token cap the buffers are sized for (default 1024; the
    // buffers cost ~0.4 GB a card at that, 4x at the mtmd default 4096)
    Vit(const Model& m, const std::string& mmproj, int max_tokens);
    ~Vit();
    Vit(const Vit&) = delete;
    Vit& operator=(const Vit&) = delete;
    // pictures -> one SVE1 file (the records the engine's GENI and the front's cache read): every path's rows in
    // order. Returns each picture's (n, nx, ny) — n merged tokens, the merged grid nx*ny.
    struct Grid { int n, nx, ny; };
    std::vector<Grid> encode_file(const std::vector<std::string>& paths, const std::string& out);
};

}  // namespace fnx
