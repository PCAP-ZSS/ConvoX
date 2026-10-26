#pragma once

#include "common/config.h"
#include "common/image.h"
#include "common/timing.h"

// True if a CUDA device is present and usable.
bool cuda_available();

// True if `impl` is one of cuda-naive | cuda-const | cuda-tiled.
bool is_cuda_impl(const std::string& impl);

// GPU 2D convolution; same arithmetic and border handling as convolve_serial.
//   cuda-naive  filter and image both read from global memory
//   cuda-const  filter in __constant__ memory, image from global memory
//   cuda-tiled  filter in __constant__ memory, image staged in a shared-memory tile with halo
// cfg.block_x x cfg.block_y is the thread-block shape (and the output tile for cuda-tiled).
// Returns kernel / H2D / D2H times from CUDA events and the total from a host clock that spans
// transfers and kernel (device allocation is excluded). Throws std::runtime_error on CUDA errors
// or an invalid launch configuration.
Timing convolve_cuda(const Image& in, const Filter& f, const Config& cfg, Image& out);

struct CudaLaunchInfo {
    int smem_bytes = 0;      // dynamic shared memory per block
    double occupancy = -1;   // theoretical, resident threads / max threads per SM
};
CudaLaunchInfo cuda_launch_info(const Config& cfg, int k);
