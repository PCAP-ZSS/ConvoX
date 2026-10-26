// Checks the filter generator for every type and every supported K.
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "common/filter.h"

static int failures = 0;

static void check(bool ok, const char* what, int k) {
    if (!ok) {
        std::printf("FAIL: %s (K=%d)\n", what, k);
        ++failures;
    }
}

static double sum(const Filter& f) {
    double s = 0;
    for (float v : f.w) s += v;
    return s;
}

int main() {
    for (int k = 3; k <= MAX_K; k += 2) {
        Filter g = make_filter(FilterType::Gaussian, k);
        Filter b = make_filter(FilterType::Box, k);
        Filter s = make_filter(FilterType::Sobel, k);
        check(g.w.size() == size_t(k) * k, "gaussian size", k);
        check(std::fabs(sum(g) - 1.0) < 1e-6, "gaussian sums to 1", k);
        check(std::fabs(sum(b) - 1.0) < 1e-6, "box sums to 1", k);
        check(std::fabs(sum(s)) < 1e-6, "sobel sums to 0", k);
        bool sym = true, anti = true;
        for (int y = 0; y < k; ++y)
            for (int x = 0; x < k; ++x) {
                sym &= std::fabs(g.w[y * k + x] - g.w[(k - 1 - y) * k + (k - 1 - x)]) < 1e-9;
                anti &= std::fabs(s.w[y * k + x] + s.w[y * k + (k - 1 - x)]) < 1e-9;
            }
        check(sym, "gaussian is symmetric", k);
        check(anti, "sobel is antisymmetric in x", k);
        check(g.w[(k / 2) * k + k / 2] >= g.w[0], "gaussian peaks at the centre", k);
    }

    // K=3 Sobel must be the classic kernel scaled by 1/4.
    Filter s3 = make_filter(FilterType::Sobel, 3);
    const float expect[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    for (int i = 0; i < 9; ++i) check(std::fabs(s3.w[i] - expect[i] / 4.0f) < 1e-6, "sobel 3x3 values", 3);

    for (int bad : {1, 2, 4, 16, 17, -3}) {
        bool threw = false;
        try { make_filter(FilterType::Box, bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid K rejected", bad);
    }
    bool threw = false;
    try { parse_filter_type("laplace"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unknown filter name rejected", 0);

    std::printf(failures ? "%d check(s) failed\n" : "all filter checks passed\n", failures);
    return failures ? 1 : 0;
}
