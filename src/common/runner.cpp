#include "common/runner.h"

#include <omp.h>

#include <stdexcept>

#include "openmp/openmp.h"
#include "serial/serial.h"

#ifdef CONVOX_HAVE_CUDA
#include "cuda/conv_cuda.h"
#endif

bool impl_available(const std::string& impl) {
    if (impl == "serial" || impl == "omp") return true;
#ifdef CONVOX_HAVE_CUDA
    if (is_cuda_impl(impl)) return cuda_available();
#endif
    return false;
}

Config normalized(Config cfg) {
#ifdef CONVOX_HAVE_CUDA
    if (is_cuda_impl(cfg.impl) && cfg.block_x <= 0 && cfg.block_y <= 0) cfg.block_x = cfg.block_y = 16;
#endif
    if (cfg.impl == "omp") {
        if (cfg.threads <= 0) cfg.threads = omp_get_max_threads();
        if (cfg.schedule.empty()) cfg.schedule = "static";
        if (cfg.collapse <= 0) cfg.collapse = 1;
        if (cfg.collapse > 2) throw std::invalid_argument("--collapse must be 1 or 2");
        if (cfg.schedule == "dynamic" && cfg.chunk <= 0) cfg.chunk = 1;
        if (!omp_schedule_valid(cfg.schedule))
            throw std::invalid_argument("--schedule must be static, dynamic or guided");
    }
    return cfg;
}

Timing run_once(const Config& cfg, const Image& in, const Filter& f, Image& out) {
    if (cfg.impl == "serial") {
        Stopwatch sw;
        convolve_serial(in, f, out);
        double t = sw.ms();
        return {t, 0, 0, t};
    }
    if (cfg.impl == "omp") {
        Stopwatch sw;
        convolve_omp(in, f, cfg, out);
        double t = sw.ms();
        return {t, 0, 0, t};
    }
#ifdef CONVOX_HAVE_CUDA
    if (is_cuda_impl(cfg.impl)) return convolve_cuda(in, f, cfg, out);
#endif
    throw std::runtime_error("implementation '" + cfg.impl + "' is not available in this build");
}

Row execute(const Config& requested, const Image& in, const Filter& f, const Image& ref, int reps,
            Image& out, std::vector<Timing>* reps_out) {
    const Config cfg = normalized(requested);
    BenchResult b = bench([&] { return run_once(cfg, in, f, out); }, reps);
    Row r;
    r.cfg = cfg;
    r.width = in.w;
    r.height = in.h;
    r.k = f.k;
    r.reps = reps;
#ifdef CONVOX_CLAMP_EDGE
    r.border = "clamp";
#else
    r.border = "zero";
#endif
    r.med = b.median;
    r.val = validate(ref, out);
#ifdef CONVOX_HAVE_CUDA
    if (is_cuda_impl(cfg.impl)) {
        CudaLaunchInfo li = cuda_launch_info(cfg, f.k);
        r.smem_bytes = li.smem_bytes;
        r.occupancy = li.occupancy;
    }
#endif
    if (reps_out) *reps_out = b.reps;
    return r;
}
