// p2p_test.cu — pre-deployment topology check: enumerate cudaDeviceCanAccessPeer for ALL GPU pairs.
// sFNX requires full-mesh P2P; pick the largest all-can=1 group and set FNX_NCARD to its size.
// P40 boxes often split into two 4-card NUMA groups with no cross-group P2P.
// Build: nvcc -arch=sm_61 -ccbin g++-13 -o p2p_test p2p_test.cu -Wno-deprecated-gpu-targets
#include <cstdio>
int main() {
    int n; cudaGetDeviceCount(&n);
    printf("devices=%d\n", n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int can = -1;
            cudaError_t e = cudaDeviceCanAccessPeer(&can, i, j);
            printf("%d->%d: %s can=%d\n", i, j, cudaGetErrorString(e), can);
        }
    return 0;
}
