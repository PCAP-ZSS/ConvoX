// Sweep file parsing and matrix expansion. Run from the repository root.
#include <cstdio>
#include <fstream>
#include <set>
#include <stdexcept>

#include "common/sweep.h"

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    SweepConfig s = load_sweep_config("benchmarks/sweep.conf");
    check(s.images.size() == 4, "4 images");
    check(s.k.size() == 7 && s.k.front() == 3 && s.k.back() == 15, "K = 3..15 in 7 steps");
    check(s.reps == 10, "10 reps");

    // Per (image, K): serial + 7 thread counts + 9 extra schedule/collapse combos (static,1 at 8
    // threads is already in the thread sweep) = 17 CPU configs, plus 3 variants x 3 blocks of CUDA.
    auto cpu = expand_configs(s, false);
    auto all = expand_configs(s, true);
    check(cpu.size() == 17, "17 CPU configs per (image, K)");
    check(all.size() == 26, "26 configs with CUDA");
    check(all.front().impl == "serial", "serial runs first");

    std::set<std::string> impls;
    for (const auto& c : all) impls.insert(c.impl);
    check(impls == std::set<std::string>({"serial", "omp", "cuda-naive", "cuda-const", "cuda-tiled"}), "all implementations present");

    bool chunks_ok = false;
    for (const auto& c : all)
        if (c.impl == "omp" && c.schedule == "dynamic" && c.chunk == 8 && c.collapse == 2 && c.threads == 8) chunks_ok = true;
    check(chunks_ok, "dynamic:8 with collapse(2) at 8 threads is present");

    bool threw = false;
    try { load_sweep_config("does/not/exist.conf"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "missing file rejected");

    std::ofstream("/tmp/convox_bad.conf") << "bogus.key = 1\n";
    threw = false;
    try { load_sweep_config("/tmp/convox_bad.conf"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unknown key rejected");

    std::printf(failures ? "%d check(s) failed\n" : "all sweep checks passed\n", failures);
    return failures ? 1 : 0;
}
