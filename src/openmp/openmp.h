#pragma once

#include "common/config.h"
#include "common/image.h"

// True if `name` is a valid OpenMP schedule: static | dynamic | guided.
bool omp_schedule_valid(const std::string& name);

// OpenMP 2D convolution, same arithmetic and border handling as convolve_serial.
//   cfg.threads   thread count (> 0)
//   cfg.schedule  static | dynamic | guided, applied through schedule(runtime)
//   cfg.chunk     chunk size, 0 = runtime default
//   cfg.collapse  1 = parallel over rows, 2 = collapse(2) over rows and columns
// The filter loops stay serial inside each thread.
void convolve_omp(const Image& in, const Filter& f, const Config& cfg, Image& out);
