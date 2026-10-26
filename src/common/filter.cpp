#include "common/filter.h"

#include <cmath>
#include <stdexcept>
#include <vector>

FilterType parse_filter_type(const std::string& name) {
    if (name == "gaussian") return FilterType::Gaussian;
    if (name == "sobel") return FilterType::Sobel;
    if (name == "box") return FilterType::Box;
    throw std::invalid_argument("unknown filter '" + name + "' (use gaussian, sobel or box)");
}

static std::vector<double> pascal_row(int n) {  // n + 1 binomial coefficients
    std::vector<double> r(n + 1, 1.0);
    for (int i = 1; i < n; ++i) r[i] = r[i - 1] * (n - i + 1) / i;
    return r;
}

static std::vector<double> convolve_1d(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) out[i + j] += a[i] * b[j];
    return out;
}

Filter make_filter(FilterType type, int k) {
    if (k < 3 || k > MAX_K || k % 2 == 0)
        throw std::invalid_argument("K must be odd and between 3 and " + std::to_string(MAX_K) +
                                    " (got " + std::to_string(k) + ")");
    const int r = (k - 1) / 2;
    std::vector<double> w(static_cast<size_t>(k) * k);

    if (type == FilterType::Box) {
        for (auto& v : w) v = 1.0 / (k * k);
    } else if (type == FilterType::Gaussian) {
        const double sigma = 0.3 * (r - 1) + 0.8;
        double sum = 0;
        for (int y = -r; y <= r; ++y)
            for (int x = -r; x <= r; ++x) {
                double v = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
                w[(y + r) * k + (x + r)] = v;
                sum += v;
            }
        for (auto& v : w) v /= sum;
    } else {
        std::vector<double> smooth = pascal_row(k - 1);
        std::vector<double> deriv = convolve_1d(pascal_row(k - 3), {-1.0, 0.0, 1.0});
        double pos = 0;
        for (int y = 0; y < k; ++y)
            for (int x = 0; x < k; ++x) {
                double v = smooth[y] * deriv[x];
                w[y * k + x] = v;
                if (v > 0) pos += v;
            }
        for (auto& v : w) v /= pos;
    }

    Filter f;
    f.k = k;
    f.w.assign(w.begin(), w.end());
    return f;
}
