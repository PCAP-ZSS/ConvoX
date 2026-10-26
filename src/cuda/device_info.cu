#include <cstdio>
#include <cuda_runtime.h>

#include "cuda/device_info.h"

void convox_print_cuda_info() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("CUDA:     no device found\n");
        return;
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    std::printf("GPU:      %s (cc %d.%d, %zu MB VRAM, %zu KB shared/block)\n", p.name,
                p.major, p.minor, p.totalGlobalMem >> 20, p.sharedMemPerBlock >> 10);
    std::printf("CUDA:     runtime %d.%d\n", rt / 1000, (rt % 1000) / 10);
}
