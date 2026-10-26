#include <cmath>
#include <cstdio>
#include <limits>

#include "common/validator.h"

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

static Image make(int w, int h, float v) {
    Image img;
    img.w = w;
    img.h = h;
    img.px.assign(static_cast<size_t>(w) * h, v);
    return img;
}

int main() {
    Image a = make(4, 3, 0.5f);

    Validation same = validate(a, a);
    check(same.pass && same.max_abs_err == 0.0 && same.rms_err == 0.0, "identical images pass with zero error");

    Image small = a;
    small.px[5] += 5e-5f;
    Validation ok = validate(a, small);
    check(ok.pass, "error below 1e-4 passes");
    check(std::fabs(ok.max_abs_err - 5e-5) < 1e-7, "max-abs error is reported");
    check(std::fabs(ok.rms_err - 5e-5 / std::sqrt(12.0)) < 1e-7, "RMS error is reported");

    Image big = a;
    big.px[0] += 2e-4f;
    check(!validate(a, big).pass, "error above 1e-4 fails");

    Image wrong = make(3, 4, 0.5f);
    Validation sz = validate(a, wrong);
    check(!sz.pass && std::isinf(sz.max_abs_err), "size mismatch fails");

    Image nan = a;
    nan.px[2] = std::numeric_limits<float>::quiet_NaN();
    check(!validate(a, nan).pass, "NaN fails");

    std::printf(failures ? "%d check(s) failed\n" : "all validator checks passed\n", failures);
    return failures ? 1 : 0;
}
