#pragma once

#include <string>
#include <vector>

#include "common/config.h"

struct SweepConfig {
    std::string image_dir = "data/images";
    std::vector<std::string> images;
    std::string filter = "gaussian";
    std::vector<int> k;
    int reps = 10;
    std::vector<int> omp_threads;
    int omp_sched_threads = 8;
    std::vector<std::string> omp_schedule;  // "static", "dynamic:8", ...
    std::vector<int> omp_collapse;
    std::vector<std::string> cuda_variants;  // naive | const | tiled
    std::vector<std::string> cuda_block;     // "16x16"
};

// Parse a sweep file (see benchmarks/sweep.conf). Throws std::invalid_argument on bad input.
SweepConfig load_sweep_config(const std::string& path);

// The configurations to run for one (image, K) pair: serial, the OpenMP thread sweep, the OpenMP
// schedule x collapse sweep, and every CUDA variant at every block shape. No duplicates.
// CUDA entries are included only when `with_cuda` is true.
std::vector<Config> expand_configs(const SweepConfig& s, bool with_cuda);

struct SweepOptions {
    std::string out_csv = "results/raw/results.csv";
    bool dry_run = false;
};

// Run every configuration for every image and K, appending to the CSV files and writing
// sweep_manifest.json next to them. Returns 0 if all runs validated, 1 otherwise.
int run_sweep(const SweepConfig& s, const SweepOptions& opt);
