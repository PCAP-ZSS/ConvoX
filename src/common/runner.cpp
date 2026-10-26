#include "common/runner.h"

#include <stdexcept>

#include "serial/serial.h"

bool impl_available(const std::string& impl) {
    return impl == "serial";
}

Timing run_once(const Config& cfg, const Image& in, const Filter& f, Image& out) {
    if (cfg.impl == "serial") {
        Stopwatch sw;
        convolve_serial(in, f, out);
        double t = sw.ms();
        return {t, 0, 0, t};
    }
    throw std::runtime_error("implementation '" + cfg.impl + "' is not available in this build");
}

Row execute(const Config& cfg, const Image& in, const Filter& f, const Image& ref, int reps,
            Image& out, std::vector<Timing>* reps_out) {
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
    if (reps_out) *reps_out = b.reps;
    return r;
}
