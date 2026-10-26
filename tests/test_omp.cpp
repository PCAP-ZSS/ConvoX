// Every OpenMP variant must match the serial reference.
#include <cstdio>
#include <random>
#include <stdexcept>

#include "common/filter.h"
#include "common/runner.h"
#include "openmp/openmp.h"
#include "serial/serial.h"

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    Image img;
    img.w = 211;  // deliberately not a multiple of anything
    img.h = 137;
    img.px.resize(static_cast<size_t>(img.w) * img.h);
    for (float& v : img.px) v = d(rng);

    int failures = 0, runs = 0;
    for (FilterType ft : {FilterType::Gaussian, FilterType::Sobel}) {
        for (int k : {3, 7, 15}) {
            Filter f = make_filter(ft, k);
            Image ref;
            convolve_serial(img, f, ref);
            for (int threads : {1, 2, 5, 16})
                for (const char* sched : {"static", "dynamic", "guided"})
                    for (int chunk : {0, 1, 32})
                        for (int collapse : {1, 2}) {
                            Config c;
                            c.impl = "omp";
                            c.threads = threads;
                            c.schedule = sched;
                            c.chunk = chunk;
                            c.collapse = collapse;
                            Image out;
                            convolve_omp(img, f, c, out);
                            Validation v = validate(ref, out);
                            ++runs;
                            if (!v.pass) {
                                std::printf("FAIL: K=%d threads=%d %s chunk=%d collapse=%d err=%g\n", k, threads,
                                            sched, chunk, collapse, v.max_abs_err);
                                ++failures;
                            }
                        }
        }
    }

    // Defaults are filled in and bad settings rejected.
    Config c;
    c.impl = "omp";
    Config n = normalized(c);
    if (n.threads < 1 || n.schedule != "static" || n.collapse != 1) {
        std::printf("FAIL: omp defaults\n");
        ++failures;
    }
    bool threw = false;
    try {
        c.schedule = "bogus";
        normalized(c);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        std::printf("FAIL: bad schedule accepted\n");
        ++failures;
    }

    if (failures)
        std::printf("%d of %d omp run(s) failed\n", failures, runs);
    else
        std::printf("all %d omp variants match serial\n", runs);
    return failures ? 1 : 0;
}
