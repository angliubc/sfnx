// The picture path's host half: decode a JPEG/PNG/BMP/GIF (stb_image, as llama.cpp's mtmd helper decodes), then the
// exact preprocessing llama.cpp's qwen3vl does before the tower — calc_size_preserved_ratio (align 32, round
// half-even, min/max pixels) and Pillow's separable bicubic resize (a = -0.5, 22-bit fixed point, horizontal pass
// first) — then channel-planar normalize (v - 0.5) / 0.5. Every step is llama.cpp's arithmetic carried over as is:
// the oracle (ref/img/sphinx.sve) was produced by that code, and any deviation moves every embedding. The pixels go
// to the cards as f32, the layout inp_raw has in clip.cpp: [3][ny][nx], row y then column x inside a channel.
// Plain C++ (no CUDA), so a CPU test can check it against the oracle without a GPU (tools/vit_test).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vit_def.hpp"

namespace fnx {

// a decoded picture in the tower's input layout: channel-planar f32, already resized and normalized
struct VitImage {
    int nx = 0, ny = 0;                   // the resized size (a multiple of 32, the patch*merge grid)
    std::vector<float> px;                // [3][ny][nx]
    int grid_x() const { return nx / VIT_PATCH; }
    int grid_y() const { return ny / VIT_PATCH; }
};

// decode (upright: the front end normalizes EXIF) and preprocess one picture to the tower's input. Throws
// std::runtime_error naming the picture for anything it cannot read; max_pixels is the preprocessing cap.
VitImage vit_preprocess(const std::string& path, int max_pixels);

}  // namespace fnx
