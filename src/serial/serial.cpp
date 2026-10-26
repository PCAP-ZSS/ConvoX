#include "serial/serial.h"

void convolve_serial(const Image& in, const Filter& f, Image& out) {
    const int w = in.w, h = in.h, k = f.k, r = (k - 1) / 2;
    out.w = w;
    out.h = h;
    out.px.assign(static_cast<size_t>(w) * h, 0.0f);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int i = -r; i <= r; ++i) {
                for (int j = -r; j <= r; ++j) {
                    int yy = y + i, xx = x + j;
#ifdef CONVOX_CLAMP_EDGE
                    yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
                    xx = xx < 0 ? 0 : (xx >= w ? w - 1 : xx);
                    acc += in.px[static_cast<size_t>(yy) * w + xx] * f.w[(i + r) * k + (j + r)];
#else
                    if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                    acc += in.px[static_cast<size_t>(yy) * w + xx] * f.w[(i + r) * k + (j + r)];
#endif
                }
            }
            out.px[static_cast<size_t>(y) * w + x] = acc;
        }
    }
}
