// Hand-checkable cases for the serial reference. Assumes the default zero-padding build.
#include <cmath>
#include <cstdio>

#include "common/filter.h"
#include "serial/serial.h"

static int failures = 0;

static void expect(float got, float want, const char* what) {
    if (std::fabs(got - want) > 1e-5f) {
        std::printf("FAIL: %s: got %f, want %f\n", what, got, want);
        ++failures;
    }
}

int main() {
#ifdef CONVOX_CLAMP_EDGE
    std::printf("test_serial checks zero-padding; build without CONVOX_CLAMP_EDGE\n");
    return 2;
#endif
    // 3x3 image 1..9 with an all-ones 3x3 filter.
    Image img;
    img.w = img.h = 3;
    img.px = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Filter ones;
    ones.k = 3;
    ones.w.assign(9, 1.0f);
    Image out;
    convolve_serial(img, ones, out);
    expect(out.px[4], 45, "centre sums all nine pixels");
    expect(out.px[0], 12, "corner (1+2+4+5)");
    expect(out.px[1], 21, "top edge (1..6)");

    // Delta filter returns the input, on a non-square image.
    Image rect;
    rect.w = 7;
    rect.h = 4;
    for (int i = 0; i < 28; ++i) rect.px.push_back(i * 0.1f);
    Filter delta;
    delta.k = 5;
    delta.w.assign(25, 0.0f);
    delta.w[12] = 1.0f;
    convolve_serial(rect, delta, out);
    for (int i = 0; i < 28; ++i) expect(out.px[i], rect.px[i], "delta filter is identity");
    if (out.w != 7 || out.h != 4) {
        std::printf("FAIL: output size\n");
        ++failures;
    }

    // Image smaller than the filter must not crash; centre sees every pixel.
    Image tiny;
    tiny.w = tiny.h = 2;
    tiny.px = {1, 2, 3, 4};
    convolve_serial(tiny, make_filter(FilterType::Box, 15), out);
    expect(out.px[0], 10.0f / 225.0f, "image smaller than filter");

    // Box filter on a constant image: interior keeps the constant.
    Image flat;
    flat.w = flat.h = 20;
    flat.px.assign(400, 0.6f);
    convolve_serial(flat, make_filter(FilterType::Box, 5), out);
    expect(out.px[10 * 20 + 10], 0.6f, "box on constant image (interior)");

    std::printf(failures ? "%d check(s) failed\n" : "all serial checks passed\n", failures);
    return failures ? 1 : 0;
}
