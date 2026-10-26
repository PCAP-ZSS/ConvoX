// Usage: test_image <input.png> <output.png>
// Loads an image, checks dimensions and value range, and writes it back.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "common/image.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: test_image <in> <out>\n");
        return 2;
    }
    Image img = load_image(argv[1]);
    auto [lo, hi] = std::minmax_element(img.px.begin(), img.px.end());
    std::printf("loaded %dx%d, %zu px, range [%.4f, %.4f]\n", img.w, img.h, img.px.size(), *lo, *hi);
    if (img.px.size() != static_cast<size_t>(img.w) * img.h || *lo < 0.0f || *hi > 1.0f) {
        std::fprintf(stderr, "FAIL: bad size or range\n");
        return 1;
    }
    save_image(argv[2], img);
    Image back = load_image(argv[2]);
    float maxd = 0;
    for (size_t i = 0; i < img.px.size(); ++i) maxd = std::max(maxd, std::abs(img.px[i] - back.px[i]));
    std::printf("round-trip max diff %.6f (8-bit quantisation expected <= 0.004)\n", maxd);
    return maxd <= 0.004f ? 0 : 1;
}
