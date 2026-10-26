#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>

#include "common/csv.h"
#include "common/env.h"
#include "common/runner.h"
#include "serial/serial.h"

#ifdef CONVOX_HAVE_CUDA
#include "cuda/device_info.h"
#endif

// Exit codes: 0 ok, 1 validation failure, 2 bad arguments, 3 missing device or implementation.
enum { EXIT_VALIDATION = 1, EXIT_BAD_ARGS = 2, EXIT_UNAVAILABLE = 3 };

static const char* USAGE =
    "usage:\n"
    "  convox info\n"
    "  convox run --impl serial --image PATH [--k 5] [--filter gaussian|sobel|box]\n"
    "             [--reps 10] [--out results/raw/results.csv] [--no-csv] [--save-image PATH]\n"
    "             [--threads N --schedule static|dynamic|guided --chunk N --collapse 1|2]\n"
    "             [--block 16x16]\n";

using Args = std::map<std::string, std::string>;

static Args parse_args(int argc, char** argv, int first) {
    Args a;
    for (int i = first; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "-i") key = "--image";
        if (key == "-k") key = "--k";
        if (key.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument '" + key + "'");
        key = key.substr(2);
        if (key == "no-csv") {
            a[key] = "1";
        } else {
            if (i + 1 >= argc) throw std::invalid_argument("--" + key + " needs a value");
            a[key] = argv[++i];
        }
    }
    return a;
}

static std::string get(const Args& a, const std::string& k, const std::string& def = "") {
    auto it = a.find(k);
    return it == a.end() ? def : it->second;
}

static int get_int(const Args& a, const std::string& k, int def) {
    auto it = a.find(k);
    if (it == a.end()) return def;
    try {
        return std::stoi(it->second);
    } catch (...) {
        throw std::invalid_argument("--" + k + " expects an integer (got '" + it->second + "')");
    }
}

static int cmd_info() {
    const Env& e = env();
    std::printf("ConvoX (%s)\n", e.git_commit.c_str());
    std::printf("CPU:      %s\n", e.cpu_model.c_str());
    std::printf("Compiler: %s\n", e.compiler.c_str());
    std::printf("OpenMP:   %d max threads, %d procs\n", omp_get_max_threads(), omp_get_num_procs());
#ifdef CONVOX_HAVE_CUDA
    convox_print_cuda_info();
#else
    std::printf("CUDA:     not built\n");
#endif
    return 0;
}

static int cmd_run(const Args& a) {
    Config cfg;
    cfg.impl = get(a, "impl", "serial");
    cfg.threads = get_int(a, "threads", 0);
    cfg.schedule = get(a, "schedule");
    cfg.chunk = get_int(a, "chunk", 0);
    cfg.collapse = get_int(a, "collapse", 0);
    std::string block = get(a, "block");
    if (!block.empty()) {
        size_t x = block.find('x');
        if (x == std::string::npos) throw std::invalid_argument("--block expects e.g. 16x16");
        cfg.block_x = std::stoi(block.substr(0, x));
        cfg.block_y = std::stoi(block.substr(x + 1));
    }
    const std::string image_path = get(a, "image");
    if (image_path.empty()) throw std::invalid_argument("--image is required");
    const int k = get_int(a, "k", 5);
    const int reps = get_int(a, "reps", 10);
    if (reps < 1) throw std::invalid_argument("--reps must be at least 1");
    const std::string filter_name = get(a, "filter", "gaussian");
    const Filter filter = make_filter(parse_filter_type(filter_name), k);

    if (!impl_available(cfg.impl)) {
        std::fprintf(stderr, "error: implementation '%s' is not available in this build\n",
                     cfg.impl.c_str());
        return EXIT_UNAVAILABLE;
    }

    const Image img = load_image(image_path);
    Image ref;
    convolve_serial(img, filter, ref);

    Image out;
    std::vector<Timing> rep_times;
    Row row = execute(cfg, img, filter, ref, reps, out, &rep_times);
    row.run_id = make_run_id();
    row.timestamp = iso_timestamp_now();
    row.image_id = std::filesystem::path(image_path).stem().string();
    row.filter_type = filter_name;

    std::printf("%-11s %-10s K=%-2d  kernel %10.3f ms  total %10.3f ms  err %.2e  %s\n",
                cfg.impl.c_str(), row.image_id.c_str(), k, row.med.kernel_ms, row.med.total_ms,
                row.val.max_abs_err, row.val.pass ? "PASS" : "FAIL");

    if (!a.count("no-csv")) {
        const std::string out_csv = get(a, "out", "results/raw/results.csv");
        append_result(out_csv, row);
        const auto reps_csv = std::filesystem::path(out_csv).parent_path() / "reps.csv";
        append_reps(reps_csv.string(), row.run_id, rep_times);
    }
    if (a.count("save-image")) save_image(get(a, "save-image"), out);
    return row.val.pass ? 0 : EXIT_VALIDATION;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fputs(USAGE, stderr);
        return EXIT_BAD_ARGS;
    }
    const std::string cmd = argv[1];
    try {
        if (cmd == "info") return cmd_info();
        if (cmd == "run") return cmd_run(parse_args(argc, argv, 2));
        std::fputs(USAGE, stderr);
        return EXIT_BAD_ARGS;
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_BAD_ARGS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_BAD_ARGS;
    }
}
