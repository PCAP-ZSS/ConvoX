#include "common/csv.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "common/env.h"

namespace fs = std::filesystem;

static std::string quote(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string o = "\"";
    for (char c : s) o += (c == '"') ? "\"\"" : std::string(1, c);
    return o + "\"";
}

static std::string num(double v) {
    std::ostringstream o;
    o.precision(9);
    o << v;
    return o.str();
}

static std::string opt_int(int v) { return v > 0 ? std::to_string(v) : ""; }
static std::string opt_num(double v) { return v >= 0 ? num(v) : ""; }

static std::ofstream open_append(const std::string& path, const std::string& header) {
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    bool fresh = !fs::exists(p) || fs::file_size(p) == 0;
    std::ofstream f(path, std::ios::app);
    if (!f) throw std::runtime_error("cannot open '" + path + "' for writing");
    if (fresh) f << header << '\n';
    return f;
}

std::string iso_timestamp_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string make_run_id() {
    static int counter = 0;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    return std::to_string(ms) + "-" + std::to_string(++counter);
}

static const char* RESULT_HEADER =
    "run_id,sweep_id,timestamp,impl,image_id,width,height,filter_type,k,border,threads,schedule,"
    "chunk,collapse,block_x,block_y,mpi_ranks,reps,kernel_ms,h2d_ms,d2h_ms,total_ms,max_abs_err,"
    "rms_err,pass,smem_bytes,occupancy,dram_gbps,cpu_model,gpu_model,compiler,cuda_version,"
    "git_commit";

void append_result(const std::string& path, const Row& r) {
    std::ofstream f = open_append(path, RESULT_HEADER);
    const Env& e = env();
    const bool gpu = r.cfg.impl.find("cuda") != std::string::npos;
    f << quote(r.run_id) << ',' << quote(r.sweep_id) << ',' << r.timestamp << ',' << r.cfg.impl << ','
      << quote(r.image_id) << ',' << r.width << ',' << r.height << ',' << r.filter_type << ',' << r.k
      << ',' << r.border << ',' << opt_int(r.cfg.threads) << ',' << r.cfg.schedule << ','
      << opt_int(r.cfg.chunk) << ',' << opt_int(r.cfg.collapse) << ',' << opt_int(r.cfg.block_x)
      << ',' << opt_int(r.cfg.block_y) << ',' << opt_int(r.mpi_ranks) << ',' << r.reps << ','
      << num(r.med.kernel_ms) << ',' << (gpu ? num(r.med.h2d_ms) : "") << ','
      << (gpu ? num(r.med.d2h_ms) : "") << ',' << num(r.med.total_ms) << ',' << num(r.val.max_abs_err)
      << ',' << num(r.val.rms_err) << ',' << (r.val.pass ? 1 : 0) << ',' << opt_int(r.smem_bytes)
      << ',' << opt_num(r.occupancy) << ',' << opt_num(r.dram_gbps) << ',' << quote(e.cpu_model) << ','
      << quote(e.gpu_model) << ',' << quote(e.compiler) << ',' << quote(e.cuda_version) << ','
      << quote(e.git_commit) << '\n';
}

void append_reps(const std::string& path, const std::string& run_id, const std::vector<Timing>& reps) {
    std::ofstream f = open_append(path, "run_id,rep,kernel_ms,h2d_ms,d2h_ms,total_ms");
    for (size_t i = 0; i < reps.size(); ++i)
        f << quote(run_id) << ',' << i + 1 << ',' << num(reps[i].kernel_ms) << ',' << num(reps[i].h2d_ms)
          << ',' << num(reps[i].d2h_ms) << ',' << num(reps[i].total_ms) << '\n';
}
