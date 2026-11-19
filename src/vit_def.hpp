// The mmproj's fixed geometry (the v.* tensors of the qwen3vl_merger projector) — the one header both halves of the
// picture path include (the host's preprocessing sizes the grids by it, the tower's shapes are checked against it).
// Plain constants, no dependencies: includable from plain C++ and CUDA alike.
#pragma once

namespace fnx {

constexpr int VIT_H = 1152, VIT_NL = 27, VIT_HEADS = 16, VIT_DH = 72, VIT_FF = 4304, VIT_PATCH = 16, VIT_MERGE = 2;
constexpr int VIT_QKV = 3 * VIT_H;                            // 3456
constexpr int VIT_MERGER_IN = VIT_H * VIT_MERGE * VIT_MERGE;  // 4608
constexpr float VIT_LN_EPS = 1e-6f;
static_assert(VIT_HEADS * VIT_DH == VIT_H, "the ViT's head size and hidden width disagree");

}  // namespace fnx
