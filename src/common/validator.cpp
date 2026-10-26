#include "common/validator.h"

#include <cmath>
#include <limits>

Validation validate(const Image& ref, const Image& test, double tol) {
    Validation v;
    if (ref.w != test.w || ref.h != test.h || ref.px.size() != test.px.size()) {
        v.max_abs_err = v.rms_err = std::numeric_limits<double>::infinity();
        return v;
    }
    double sq = 0.0;
    for (size_t i = 0; i < ref.px.size(); ++i) {
        double d = std::fabs(static_cast<double>(ref.px[i]) - test.px[i]);
        if (std::isnan(d)) {
            v.max_abs_err = v.rms_err = std::numeric_limits<double>::infinity();
            return v;
        }
        if (d > v.max_abs_err) v.max_abs_err = d;
        sq += d * d;
    }
    v.rms_err = ref.px.empty() ? 0.0 : std::sqrt(sq / static_cast<double>(ref.px.size()));
    v.pass = v.max_abs_err < tol;
    return v;
}
