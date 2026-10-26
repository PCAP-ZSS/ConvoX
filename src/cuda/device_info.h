#pragma once

#include <string>

void convox_print_cuda_info();
std::string convox_cuda_device_name();     // empty if no device
std::string convox_cuda_version_string();  // e.g. "12.6"
