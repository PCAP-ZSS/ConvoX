#pragma once

#include <string>

#include "common/image.h"

enum class FilterType { Gaussian, Sobel, Box };

constexpr int MAX_K = 15;

// Parse "gaussian" | "sobel" | "box". Throws std::invalid_argument.
FilterType parse_filter_type(const std::string& name);

// Build a K x K filter. K must be odd and in [3, MAX_K]; throws std::invalid_argument otherwise.
//   Gaussian: sigma = 0.3 * ((K - 1) / 2 - 1) + 0.8, weights sum to 1.
//   Box:      every weight is 1 / K^2.
//   Sobel:    horizontal-gradient kernel (smoothing column x derivative row), extended to
//             any odd K, scaled so the positive weights sum to 1. Weights sum to 0.
Filter make_filter(FilterType type, int k);
