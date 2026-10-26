#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common/image.h"

#include <algorithm>
#include <stdexcept>

#include "stb_image.h"
#include "stb_image_write.h"

Image load_image(const std::string& path) {
    int w = 0, h = 0, n = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 1);  // force grayscale
    if (!data) throw std::runtime_error("cannot read image '" + path + "': " + stbi_failure_reason());
    Image img;
    img.w = w;
    img.h = h;
    img.px.resize(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < img.px.size(); ++i) img.px[i] = data[i] / 255.0f;
    stbi_image_free(data);
    return img;
}

void save_image(const std::string& path, const Image& img) {
    std::vector<unsigned char> out(img.px.size());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<unsigned char>(std::clamp(img.px[i], 0.0f, 1.0f) * 255.0f + 0.5f);
    auto ends_with = [&](const char* s) {
        std::string e(s);
        return path.size() >= e.size() && path.compare(path.size() - e.size(), e.size(), e) == 0;
    };
    int ok = ends_with(".jpg") || ends_with(".jpeg")
                 ? stbi_write_jpg(path.c_str(), img.w, img.h, 1, out.data(), 95)
                 : stbi_write_png(path.c_str(), img.w, img.h, 1, out.data(), img.w);
    if (!ok) throw std::runtime_error("cannot write image '" + path + "'");
}
