#include <cstdio>
#include <cuda_runtime.h>

#include "cuda/device_info.h"

std::string convox_cuda_device_name() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return "";
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    return p.name;
}

std::string convox_cuda_version_string() {
    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    return std::to_string(rt / 1000) + "." + std::to_string((rt % 1000) / 10);
}

void convox_print_cuda_info() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("CUDA:     no device found\n");
        return;
    }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    std::printf("GPU:      %s (cc %d.%d, %zu MB VRAM, %zu KB shared/block)\n", p.name, p.major,
                p.minor, p.totalGlobalMem >> 20, p.sharedMemPerBlock >> 10);
    std::printf("CUDA:     runtime %s\n", convox_cuda_version_string().c_str());
}
