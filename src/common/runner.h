#pragma once

#include <string>
#include <vector>

#include "common/config.h"
#include "common/filter.h"
#include "common/image.h"
#include "common/timing.h"
#include "common/validator.h"

// One finished configuration: what was run, its median timing and its validation result.
struct Row {
    std::string run_id, sweep_id, timestamp;
    std::string image_id, filter_type;
    Config cfg;
    int width = 0, height = 0, k = 0, reps = 0;
    std::string border;
    int mpi_ranks = 0;
    int smem_bytes = 0;
    double occupancy = -1, dram_gbps = -1;  // negative = not measured
    Timing med;
    Validation val;
};

// True when `impl` names a variant that is compiled into this binary.
bool impl_available(const std::string& impl);

// Fill in per-implementation defaults (omp: all threads, static schedule, row parallelism) and
// reject invalid settings with std::invalid_argument.
Config normalized(Config cfg);

// Run one convolution of `in` with the implementation named in `cfg`. Throws
// std::runtime_error if the implementation is unavailable.
Timing run_once(const Config& cfg, const Image& in, const Filter& f, Image& out);

// Warm-up + `reps` timed runs of `cfg`, validated against `ref`. The last output stays in `out`.
// Fills the timing, validation and size fields of the returned Row; the caller sets the
// identity fields (run_id, image_id, filter_type, ...).
Row execute(const Config& cfg, const Image& in, const Filter& f, const Image& ref, int reps,
            Image& out, std::vector<Timing>* reps_out = nullptr);
