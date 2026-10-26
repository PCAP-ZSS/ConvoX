#include "openmp/openmp.h"

#include <omp.h>

#include <stdexcept>

bool omp_schedule_valid(const std::string& name) {
    return name == "static" || name == "dynamic" || name == "guided";
}

// Convolution of one output pixel; identical arithmetic to the serial reference.
static inline float pixel(const float* src, const float* w, int W, int H, int k, int r, int y, int x) {
    float acc = 0.0f;
    for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
            int yy = y + i, xx = x + j;
#ifdef CONVOX_CLAMP_EDGE
            yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
            xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
            acc += src[static_cast<size_t>(yy) * W + xx] * w[(i + r) * k + (j + r)];
#else
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            acc += src[static_cast<size_t>(yy) * W + xx] * w[(i + r) * k + (j + r)];
#endif
        }
    }
    return acc;
}

void convolve_omp(const Image& in, const Filter& f, const Config& cfg, Image& out) {
    if (!omp_schedule_valid(cfg.schedule)) throw std::invalid_argument("bad OpenMP schedule '" + cfg.schedule + "'");
    if (cfg.threads < 1) throw std::invalid_argument("OpenMP thread count must be at least 1");

    const int W = in.w, H = in.h, k = f.k, r = (k - 1) / 2;
    out.w = W;
    out.h = H;
    out.px.assign(static_cast<size_t>(W) * H, 0.0f);

    omp_sched_t kind = cfg.schedule == "static" ? omp_sched_static
                       : cfg.schedule == "dynamic" ? omp_sched_dynamic
                                                   : omp_sched_guided;
    omp_set_schedule(kind, cfg.chunk);
    omp_set_num_threads(cfg.threads);

    const float* src = in.px.data();
    const float* w = f.w.data();
    float* dst = out.px.data();

    if (cfg.collapse == 2) {
#pragma omp parallel for collapse(2) schedule(runtime)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) dst[static_cast<size_t>(y) * W + x] = pixel(src, w, W, H, k, r, y, x);
    } else {
#pragma omp parallel for schedule(runtime)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) dst[static_cast<size_t>(y) * W + x] = pixel(src, w, W, H, k, r, y, x);
    }
}
