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

void convox_cuda_peaks(double* gbps, double* gflops) {
    *gbps = *gflops = 0;
    int dev = 0, mem_khz = 0, bus_bits = 0, sms = 0, core_khz = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return;
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&core_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    int cores_per_sm = 0;  // fp32 lanes per SM
    switch (major * 10 + minor) {
        case 60: cores_per_sm = 64; break;
        case 61: case 62: cores_per_sm = 128; break;
        case 70: case 72: case 75: cores_per_sm = 64; break;
        case 80: cores_per_sm = 64; break;
        case 86: case 87: case 89: cores_per_sm = 128; break;
        default: cores_per_sm = 128; break;
    }
    *gbps = 2.0 * mem_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // DDR: two transfers per clock
    *gflops = static_cast<double>(sms) * cores_per_sm * core_khz * 1e3 * 2.0 / 1e9;
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
