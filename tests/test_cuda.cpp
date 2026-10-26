// Every CUDA variant must match the serial reference, for awkward image sizes and all block shapes.
#include <cstdio>
#include <random>
#include <stdexcept>

#include "common/filter.h"
#include "common/runner.h"
#include "cuda/conv_cuda.h"
#include "serial/serial.h"

int main() {
    if (!cuda_available()) {
        std::printf("no CUDA device, skipping\n");
        return 0;
    }
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    struct Size { int w, h; };
    int failures = 0, runs = 0;

    for (Size s : {Size{257, 131}, Size{33, 17}, Size{8, 8}, Size{1, 1}}) {
        Image img;
        img.w = s.w;
        img.h = s.h;
        img.px.resize(static_cast<size_t>(s.w) * s.h);
        for (float& v : img.px) v = d(rng);

        for (FilterType ft : {FilterType::Gaussian, FilterType::Sobel})
            for (int k : {3, 7, 15}) {
                Filter f = make_filter(ft, k);
                Image ref;
                convolve_serial(img, f, ref);
                for (const char* impl : {"cuda-naive", "cuda-const", "cuda-tiled"})
                    for (auto bs : {std::pair<int, int>{8, 8}, {16, 16}, {32, 32}, {32, 8}, {7, 5}}) {
                        Config c;
                        c.impl = impl;
                        c.block_x = bs.first;
                        c.block_y = bs.second;
                        Image out;
                        Timing t = convolve_cuda(img, f, c, out);
                        Validation v = validate(ref, out);
                        ++runs;
                        bool timing_ok = t.kernel_ms >= 0 && t.h2d_ms >= 0 && t.d2h_ms >= 0 && t.total_ms >= t.kernel_ms;
                        if (!v.pass || !timing_ok) {
                            std::printf("FAIL: %s %dx%d K=%d block=%dx%d err=%g timing_ok=%d\n", impl, s.w, s.h, k,
                                        bs.first, bs.second, v.max_abs_err, timing_ok);
                            ++failures;
                        }
                    }
            }
    }

    // Invalid launch shapes are rejected with a clear error.
    Image img;
    img.w = img.h = 4;
    img.px.assign(16, 1.0f);
    Config bad;
    bad.impl = "cuda-naive";
    bad.block_x = 64;
    bad.block_y = 32;  // 2048 threads
    bool threw = false;
    try {
        Image o;
        convolve_cuda(img, make_filter(FilterType::Box, 3), bad, o);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        std::printf("FAIL: oversized block accepted\n");
        ++failures;
    }

    // Launch info: tiled K=15 at 32x32 needs 46*46*4 bytes of shared memory.
    Config t;
    t.impl = "cuda-tiled";
    t.block_x = t.block_y = 32;
    CudaLaunchInfo li = cuda_launch_info(t, 15);
    if (li.smem_bytes != 46 * 46 * 4 || li.occupancy <= 0 || li.occupancy > 1) {
        std::printf("FAIL: launch info smem=%d occ=%g\n", li.smem_bytes, li.occupancy);
        ++failures;
    }

    if (failures)
        std::printf("%d of %d cuda run(s) failed\n", failures, runs);
    else
        std::printf("all %d cuda variants match serial\n", runs);
    return failures ? 1 : 0;
}
