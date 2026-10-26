#pragma once

#include <string>

// One benchmark configuration. Fields that do not apply to an implementation stay 0 / empty.
//   impl: serial | omp | cuda-naive | cuda-const | cuda-tiled
struct Config {
    std::string impl = "serial";
    int threads = 0;
    std::string schedule;  // static | dynamic | guided (omp)
    int chunk = 0;         // schedule chunk size, 0 = default
    int collapse = 0;      // 1 = rows, 2 = collapse(2) (omp)
    int block_x = 0, block_y = 0;  // CUDA block dimensions
};
