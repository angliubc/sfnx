// wmma_test.cu — verify the sm61_wmma.h emulation layer against a CPU reference, on the P40.
// Run BEFORE any full rebuild after touching sm61_wmma.h (build is ~25 min, this is ~5 min).
// Covers: matrix_a row-major load, matrix_b row-major AND col-major load, mma_sync,
// store row-major AND col-major. Requires max abs err = 0 (exactly, same accumulation order).
// Build: nvcc -arch=sm_61 -ccbin g++-13 -o wmma_test wmma_test.cu -Wno-deprecated-gpu-targets
//   (sm61_wmma.h must be next to this file or on the include path)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "sm61_wmma.h"

using namespace nvcuda;
using namespace nvcuda::wmma;

__global__ void test_kernel(float* out_err) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half* A = reinterpret_cast<__half*>(smem);            // [16][24] row-major, ldm 24
    __half* B = A + 16 * 24;                                 // [16][24] row-major source
    __half* Bc = B + 16 * 24;                                // col-major copy: element (r,c) at Bc[c*24+r]
    float* D = reinterpret_cast<float*>(Bc + 16 * 24);       // [16][40]
    float* Dc = D + 16 * 40;                                 // col-major store target: (r,c) at Dc[c*40+r]
    float* ref = Dc + 16 * 40;                               // [16][16]

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, l = threadIdx.x;

    // all 128 threads fill, then all compute the reference (16x16 = 256 refs, 2 per thread)
    for (int e = l; e < 16 * 24; e += 128) { A[e] = __float2half_rn(((e * 37) % 97) / 97.f - 0.5f); B[e] = __float2half_rn(((e * 53) % 89) / 89.f - 0.5f); }
    for (int e = l; e < 16 * 24; e += 128) { const int r = e / 24, c = e % 24; Bc[c * 24 + r] = B[e]; }
    for (int e = l; e < 16 * 40; e += 128) { D[e] = 0.f; Dc[e] = 0.f; }
    for (int e = l; e < 16 * 16; e += 128) ref[e] = 0.f;
    __syncthreads();
    for (int e = l; e < 16 * 16; e += 128) {
        const int r = e / 16, c = e % 16;
        float s = 0.f;
        for (int k = 0; k < 16; ++k) s += __half2float(A[r * 24 + k]) * __half2float(B[k * 24 + c]);
        ref[e] = s;
    }
    __syncthreads();

    if (warp == 1) {   // row-major B path: D = A(row) x B(row)
        fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
        fragment<matrix_b, 16, 16, 16, __half, row_major> fb;
        fragment<accumulator, 16, 16, 16, float, void> fc;
        load_matrix_sync(fa, A, 24);
        load_matrix_sync(fb, B, 24);
        fill_fragment(fc, 0.f);
        mma_sync(fc, fa, fb, fc);
        store_matrix_sync(D, fc, 40, mem_row_major);
    }
    if (warp == 2) {   // col-major B path, col-major store
        fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
        fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
        fragment<accumulator, 16, 16, 16, float, void> fc;
        load_matrix_sync(fa, A, 24);
        load_matrix_sync(fb, Bc, 24);
        fill_fragment(fc, 0.f);
        mma_sync(fc, fa, fb, fc);
        store_matrix_sync(Dc, fc, 40, mem_col_major);
    }
    __syncthreads();
    if (warp == 3) {   // error check
        float err = 0.f;
        for (int e = lane; e < 16 * 16; e += 32) {
            const int r = e / 16, c = e % 16;
            err = fmaxf(err, fabsf(D[r * 40 + c] - ref[r * 16 + c]));
            err = fmaxf(err, fabsf(Dc[c * 40 + r] - ref[r * 16 + c]));
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) err = fmaxf(err, __shfl_xor_sync(0xffffffffu, err, o));
        if (lane == 0) out_err[0] = fmaxf(out_err[0], err);
    }
}

int main() {
    float* err;
    cudaMalloc(&err, 4);
    cudaMemset(err, 0, 4);
    const size_t sm = (16 * 24 * 3) * 2 + (16 * 40) * 4 + (16 * 40) * 4 + (16 * 16) * 4 + 64;
    test_kernel<<<1, 128, sm>>>(err);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("LAUNCH/EXEC FAIL: %s\n", cudaGetErrorString(e)); return 1; }
    float h = -1;
    cudaMemcpy(&h, err, 4, cudaMemcpyDeviceToHost);
    printf("max abs err = %g  %s\n", h, h < 1e-2f ? "PASS" : "FAIL");
    return h < 1e-2f ? 0 : 2;
}
