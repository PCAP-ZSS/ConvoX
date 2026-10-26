#include "common/timing.h"

#include <algorithm>

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

BenchResult bench(const std::function<Timing()>& fn, int reps, int warmup) {
    for (int i = 0; i < warmup; ++i) fn();
    BenchResult r;
    for (int i = 0; i < reps; ++i) r.reps.push_back(fn());
    auto field = [&](double Timing::*m) {
        std::vector<double> v;
        for (const Timing& t : r.reps) v.push_back(t.*m);
        return median_of(v);
    };
    r.median = {field(&Timing::kernel_ms), field(&Timing::h2d_ms), field(&Timing::d2h_ms),
                field(&Timing::total_ms)};
    return r;
}
