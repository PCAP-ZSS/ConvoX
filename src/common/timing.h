#pragma once

#include <chrono>
#include <functional>
#include <vector>

// Times in milliseconds. CPU runs report kernel_ms == total_ms and zero transfers.
struct Timing {
    double kernel_ms = 0, h2d_ms = 0, d2h_ms = 0, total_ms = 0;
};

class Stopwatch {
public:
    Stopwatch() : t0_(Clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0_;
};

struct BenchResult {
    Timing median;             // field-wise median over the measured runs
    std::vector<Timing> reps;  // every measured run (warm-up excluded)
};

// Run `fn` `warmup` times (discarded) then `reps` times, collecting each Timing.
BenchResult bench(const std::function<Timing()>& fn, int reps = 10, int warmup = 1);
