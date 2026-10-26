#pragma once

#include "common/image.h"

// Reference 2D convolution (the correctness oracle and speedup baseline).
//   O[y][x] = sum_{i,j=-R..R} I[y+i][x+j] * F[i+R][j+R],  R = (K-1)/2
// Pixels outside the image read as 0 (zero-padding), or as the nearest edge pixel when the
// project is built with CONVOX_CLAMP_EDGE. `out` is resized to match `in`.
void convolve_serial(const Image& in, const Filter& f, Image& out);
