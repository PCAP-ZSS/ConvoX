#pragma once

#include "common/image.h"

constexpr double VALIDATION_TOL = 1e-4;

struct Validation {
    double max_abs_err = 0.0;
    double rms_err = 0.0;
    bool pass = false;  // max_abs_err < VALIDATION_TOL and the sizes match
};

// Compare `test` element-wise against `ref`. A size mismatch or a NaN fails with
// max_abs_err = infinity.
Validation validate(const Image& ref, const Image& test, double tol = VALIDATION_TOL);
