#pragma once

#include <string>

#include "common/filter.h"

// Time of the non-parallel parts of a convolution job, in milliseconds (median of `reps`).
// These make up the serial fraction f that feeds Amdahl's law:
//   load   decode the image file to a float buffer
//   filter build the K x K filter
//   alloc  allocate and first-touch the output buffer
//   save   encode and write the result image
struct SerialProfile {
    double load_ms = 0, filter_ms = 0, alloc_ms = 0, save_ms = 0;
    int width = 0, height = 0;
};

SerialProfile profile_serial_parts(const std::string& image_path, FilterType type, int k, int reps = 3);

// Append one row to serial_profile.csv (header written for a new file).
void append_profile(const std::string& path, const std::string& image_id, const std::string& filter,
                    int k, const SerialProfile& p);
