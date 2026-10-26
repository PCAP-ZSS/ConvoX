#include "common/profile.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/image.h"
#include "common/timing.h"

namespace fs = std::filesystem;

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

SerialProfile profile_serial_parts(const std::string& image_path, FilterType type, int k, int reps) {
    std::vector<double> load, filt, alloc, save;
    SerialProfile p;
    const fs::path tmp = fs::temp_directory_path() / "convox_profile_out.png";
    for (int i = 0; i < reps; ++i) {
        Stopwatch sw;
        Image img = load_image(image_path);
        load.push_back(sw.ms());
        p.width = img.w;
        p.height = img.h;

        sw = Stopwatch();
        Filter f = make_filter(type, k);
        filt.push_back(sw.ms());

        sw = Stopwatch();
        std::vector<float> out;
        out.assign(img.px.size(), 0.0f);
        alloc.push_back(sw.ms());

        sw = Stopwatch();
        save_image(tmp.string(), img);
        save.push_back(sw.ms());
    }
    fs::remove(tmp);
    p.load_ms = median(load);
    p.filter_ms = median(filt);
    p.alloc_ms = median(alloc);
    p.save_ms = median(save);
    return p;
}

void append_profile(const std::string& path, const std::string& image_id, const std::string& filter,
                    int k, const SerialProfile& p) {
    fs::path fp(path);
    if (fp.has_parent_path()) fs::create_directories(fp.parent_path());
    bool fresh = !fs::exists(fp) || fs::file_size(fp) == 0;
    std::ofstream f(path, std::ios::app);
    if (fresh) f << "image_id,filter_type,k,width,height,load_ms,filter_ms,alloc_ms,save_ms\n";
    f << image_id << ',' << filter << ',' << k << ',' << p.width << ',' << p.height << ',' << p.load_ms
      << ',' << p.filter_ms << ',' << p.alloc_ms << ',' << p.save_ms << '\n';
}
