#include "common/sweep.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "common/csv.h"
#include "common/env.h"
#include "common/profile.h"
#include "common/runner.h"
#include "serial/serial.h"

#ifdef CONVOX_HAVE_CUDA
#include "cuda/device_info.h"
#endif

namespace fs = std::filesystem;

static std::vector<std::string> words(const std::string& s) {
    std::istringstream is(s);
    std::vector<std::string> out;
    for (std::string w; is >> w;) out.push_back(w);
    return out;
}

static std::vector<int> ints(const std::string& key, const std::vector<std::string>& v) {
    std::vector<int> out;
    for (const auto& w : v) {
        try {
            out.push_back(std::stoi(w));
        } catch (...) {
            throw std::invalid_argument("sweep: '" + key + "' expects integers (got '" + w + "')");
        }
    }
    return out;
}

SweepConfig load_sweep_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::invalid_argument("cannot read sweep file '" + path + "'");
    std::map<std::string, std::vector<std::string>> kv;
    std::string line;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            if (!words(line).empty()) throw std::invalid_argument("sweep: expected 'key = values' in '" + line + "'");
            continue;
        }
        auto key = words(line.substr(0, eq));
        if (key.size() != 1) throw std::invalid_argument("sweep: bad key in '" + line + "'");
        kv[key[0]] = words(line.substr(eq + 1));
    }

    static const std::set<std::string> known = {
        "image_dir", "images", "filter", "k", "reps", "omp.threads", "omp.sched_threads",
        "omp.schedule", "omp.collapse", "cuda.variants", "cuda.block"};
    for (const auto& [key, _] : kv)
        if (!known.count(key)) throw std::invalid_argument("sweep: unknown key '" + key + "'");

    SweepConfig s;
    auto one = [&](const std::string& key, std::string& dst) {
        if (kv.count(key) && kv[key].size() == 1) dst = kv[key][0];
    };
    one("image_dir", s.image_dir);
    one("filter", s.filter);
    if (kv.count("images")) s.images = kv["images"];
    if (kv.count("k")) s.k = ints("k", kv["k"]);
    if (kv.count("reps")) s.reps = ints("reps", kv["reps"]).at(0);
    if (kv.count("omp.threads")) s.omp_threads = ints("omp.threads", kv["omp.threads"]);
    if (kv.count("omp.sched_threads")) s.omp_sched_threads = ints("omp.sched_threads", kv["omp.sched_threads"]).at(0);
    if (kv.count("omp.schedule")) s.omp_schedule = kv["omp.schedule"];
    if (kv.count("omp.collapse")) s.omp_collapse = ints("omp.collapse", kv["omp.collapse"]);
    if (kv.count("cuda.variants")) s.cuda_variants = kv["cuda.variants"];
    if (kv.count("cuda.block")) s.cuda_block = kv["cuda.block"];

    if (s.images.empty()) throw std::invalid_argument("sweep: 'images' is empty");
    if (s.k.empty()) throw std::invalid_argument("sweep: 'k' is empty");
    if (s.reps < 1) throw std::invalid_argument("sweep: reps must be at least 1");
    return s;
}

static Config omp_config(int threads, const std::string& sched, int collapse) {
    Config c;
    c.impl = "omp";
    c.threads = threads;
    size_t colon = sched.find(':');
    c.schedule = sched.substr(0, colon);
    c.chunk = colon == std::string::npos ? 0 : std::stoi(sched.substr(colon + 1));
    c.collapse = collapse;
    return c;
}

std::vector<Config> expand_configs(const SweepConfig& s, bool with_cuda) {
    std::vector<Config> out;
    std::set<std::tuple<std::string, int, std::string, int, int, int, int>> seen;
    auto add = [&](const Config& raw) {
        Config c = normalized(raw);
        if (seen.insert({c.impl, c.threads, c.schedule, c.chunk, c.collapse, c.block_x, c.block_y}).second)
            out.push_back(c);
    };

    Config serial;
    serial.impl = "serial";
    add(serial);
    for (int t : s.omp_threads) add(omp_config(t, "static", 1));
    if (!s.omp_schedule.empty() && !s.omp_collapse.empty())
        for (const auto& sched : s.omp_schedule)
            for (int col : s.omp_collapse) add(omp_config(s.omp_sched_threads, sched, col));
    if (with_cuda)
        for (const auto& v : s.cuda_variants)
            for (const auto& b : s.cuda_block) {
                size_t x = b.find('x');
                if (x == std::string::npos) throw std::invalid_argument("sweep: bad block '" + b + "'");
                Config c;
                c.impl = "cuda-" + v;
                c.block_x = std::stoi(b.substr(0, x));
                c.block_y = std::stoi(b.substr(x + 1));
                add(c);
            }
    return out;
}

static void write_manifest(const fs::path& path, const std::string& sweep_id, const SweepConfig& s) {
    const Env& e = env();
    double peak_gbps = 0, peak_gflops = 0;
#ifdef CONVOX_HAVE_CUDA
    convox_cuda_peaks(&peak_gbps, &peak_gflops);
#endif
    std::ofstream f(path);
    f << "{\n  \"sweep_id\": \"" << sweep_id << "\",\n  \"git_commit\": \"" << e.git_commit
      << "\",\n  \"cpu\": \"" << e.cpu_model << "\",\n  \"gpu\": \"" << e.gpu_model
      << "\",\n  \"compiler\": \"" << e.compiler << "\",\n  \"cuda_version\": \"" << e.cuda_version
      << "\",\n  \"peak_gpu_gbps\": " << peak_gbps << ",\n  \"peak_gpu_gflops\": " << peak_gflops
      << ",\n  \"filter\": \"" << s.filter << "\",\n  \"reps\": " << s.reps << ",\n  \"images\": [";
    for (size_t i = 0; i < s.images.size(); ++i) f << (i ? ", " : "") << '"' << s.images[i] << '"';
    f << "],\n  \"k\": [";
    for (size_t i = 0; i < s.k.size(); ++i) f << (i ? ", " : "") << s.k[i];
    f << "]\n}\n";
}

// Identity of one configuration inside a results file, matching how append_result writes it.
static std::string config_key(const std::string& impl, const std::string& image, int k, const Config& c) {
    auto opt = [](int v) { return v > 0 ? std::to_string(v) : std::string(); };
    return impl + "|" + image + "|" + std::to_string(k) + "|" + opt(c.threads) + "|" + c.schedule + "|" +
           opt(c.chunk) + "|" + opt(c.collapse) + "|" + opt(c.block_x) + "|" + opt(c.block_y);
}

// Read the configurations already recorded in a results file, and the sweep_id of its last row.
static std::set<std::string> read_done(const std::string& path, std::string* last_sweep_id) {
    std::set<std::string> done;
    std::ifstream f(path);
    std::string line;
    if (!std::getline(f, line)) return done;  // header
    while (std::getline(f, line)) {
        std::vector<std::string> v;
        std::stringstream ss(line);
        for (std::string cell; v.size() < 17 && std::getline(ss, cell, ',');) v.push_back(cell);
        if (v.size() < 17) continue;
        // columns: 0 run_id 1 sweep_id 3 impl 4 image_id 8 k 10 threads 11 schedule 12 chunk
        //          13 collapse 14 block_x 15 block_y
        *last_sweep_id = v[1];
        done.insert(v[3] + "|" + v[4] + "|" + v[8] + "|" + v[10] + "|" + v[11] + "|" + v[12] + "|" + v[13] + "|" +
                    v[14] + "|" + v[15]);
    }
    return done;
}

int run_sweep(const SweepConfig& s, const SweepOptions& opt) {
    bool with_cuda = false;
#ifdef CONVOX_HAVE_CUDA
    with_cuda = impl_available("cuda-naive");
#endif
    const std::vector<Config> configs = expand_configs(s, with_cuda);
    const size_t total = configs.size() * s.images.size() * s.k.size();
    std::printf("sweep: %zu images x %zu K x %zu configs = %zu runs%s\n", s.images.size(), s.k.size(),
                configs.size(), total, with_cuda ? "" : " (no CUDA device: CUDA variants skipped)");
    if (opt.dry_run) return 0;

    std::string sweep_id = iso_timestamp_now();
    const fs::path out_dir = fs::path(opt.out_csv).parent_path();
    const std::string reps_csv = (out_dir / "reps.csv").string();
    fs::create_directories(out_dir);

    std::set<std::string> already;
    if (opt.resume && fs::exists(opt.out_csv)) {
        std::string prev;
        already = read_done(opt.out_csv, &prev);
        if (!prev.empty()) sweep_id = prev;  // keep one sweep_id so the analysis sees a single sweep
        std::printf("resume: %zu configurations already recorded in sweep %s\n", already.size(), sweep_id.c_str());
    } else {
        write_manifest(out_dir / "sweep_manifest.json", sweep_id, s);
    }

    size_t pending_total = 0;
    for (const auto& image_id : s.images)
        for (int k : s.k)
            for (const Config& cfg : configs)
                if (!already.count(config_key(cfg.impl, image_id, k, cfg))) ++pending_total;

    size_t done = 0, failed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& image_id : s.images) {
        const std::string image_path = (fs::path(s.image_dir) / (image_id + ".png")).string();
        Image img;
        bool loaded = false;
        for (int k : s.k) {
            std::vector<Config> todo;
            for (const Config& cfg : configs)
                if (!already.count(config_key(cfg.impl, image_id, k, cfg))) todo.push_back(cfg);
            if (todo.empty()) continue;
            if (!loaded) {
                img = load_image(image_path);
                loaded = true;
            }
            const Filter filter = make_filter(parse_filter_type(s.filter), k);
            Image ref;
            convolve_serial(img, filter, ref);
            if (todo.size() == configs.size())  // first visit of this (image, K)
                append_profile((out_dir / "serial_profile.csv").string(), image_id, s.filter, k,
                               profile_serial_parts(image_path, parse_filter_type(s.filter), k));
            for (const Config& cfg : todo) {
                Image out;
                std::vector<Timing> reps;
                Row row = execute(cfg, img, filter, ref, s.reps, out, &reps);
                row.sweep_id = sweep_id;
                row.run_id = make_run_id();
                row.timestamp = iso_timestamp_now();
                row.image_id = image_id;
                row.filter_type = s.filter;
                append_result(opt.out_csv, row);
                append_reps(reps_csv, row.run_id, reps);
                ++done;
                if (!row.val.pass) ++failed;
                double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                double eta = elapsed / done * (pending_total - done);
                std::string sched = row.cfg.schedule.empty() ? "-" : row.cfg.schedule;
                if (row.cfg.chunk > 0) sched += ":" + std::to_string(row.cfg.chunk);
                std::printf("[%4zu / %zu] %-10s %-8s K=%-2d thr=%-2d sch=%-10s col=%d blk=%dx%d  %9.3f ms  %s  ~%.0f min left\n",
                            done, pending_total, row.cfg.impl.c_str(), image_id.c_str(), k, row.cfg.threads,
                            sched.c_str(), row.cfg.collapse, row.cfg.block_x, row.cfg.block_y, row.med.total_ms,
                            row.val.pass ? "PASS" : "FAIL", eta / 60.0);
                std::fflush(stdout);
            }
        }
    }
    std::printf("sweep done: %zu runs, %zu failed validation\n", done, failed);
    return failed ? 1 : 0;
}
