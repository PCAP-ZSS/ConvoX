#pragma once

#include <string>
#include <vector>

// Single-channel float image, row-major, values nominally in [0, 1].
struct Image {
    int w = 0, h = 0;
    std::vector<float> px;
};

// K x K filter, K odd, row-major.
struct Filter {
    int k = 0;
    std::vector<float> w;
};

// Load any stb-supported file as grayscale floats in [0, 1]. Throws std::runtime_error.
Image load_image(const std::string& path);

// Write as 8-bit grayscale (PNG or JPG by extension). Values are clamped to [0, 1].
void save_image(const std::string& path, const Image& img);
