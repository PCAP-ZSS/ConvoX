#pragma once

#include <string>

void convox_print_cuda_info();
std::string convox_cuda_device_name();     // empty if no device
std::string convox_cuda_version_string();  // e.g. "12.6"

// Theoretical peaks of device 0, for the roofline plot: DRAM bandwidth (GB/s) and fp32
// throughput (GFLOP/s, counting an FMA as 2 FLOPs). Both 0 if unknown.
void convox_cuda_peaks(double* gbps, double* gflops);
