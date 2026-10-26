// Distributed-memory convolution: rank 0 scatters horizontal strips (with R halo rows on each
// side) of the image, every rank convolves its strip with the serial / OpenMP / CUDA kernel,
// and the interior rows are gathered back at rank 0, validated against the serial reference.
//
//   mpirun -np 4 ./build/convox_mpi -i data/images/img01_4k.png -k 7 --local omp --threads 4
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/csv.h"
#include "common/runner.h"
#include "serial/serial.h"

using Args = std::map<std::string, std::string>;

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "-i") key = "--image";
        if (key == "-k") key = "--k";
        if (key.rfind("--", 0) != 0 || (i + 1 >= argc && key != "--no-csv"))
            throw std::invalid_argument("bad argument '" + std::string(argv[i]) + "'");
        key = key.substr(2);
        if (key == "no-csv") a[key] = "1";
        else a[key] = argv[++i];
    }
    return a;
}

static std::string get(const Args& a, const std::string& k, const std::string& d = "") {
    auto it = a.find(k);
    return it == a.end() ? d : it->second;
}

static void die(int rank, const std::string& msg, int code = 2) {
    if (rank == 0) std::fprintf(stderr, "error: %s\n", msg.c_str());
    MPI_Abort(MPI_COMM_WORLD, code);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        Args a = parse_args(argc, argv);
        const std::string image_path = get(a, "image");
        if (image_path.empty()) throw std::invalid_argument("--image is required");
        const int k = std::stoi(get(a, "k", "5"));
        const int reps = std::stoi(get(a, "reps", "10"));
        const std::string filter_name = get(a, "filter", "gaussian");
        const Filter filter = make_filter(parse_filter_type(filter_name), k);
        const int R = (k - 1) / 2;

        Config local;
        local.impl = get(a, "local", "omp");
        if (local.impl == "cuda") local.impl = "cuda-tiled";
        if (local.impl == "omp") {
            unsigned hw = std::thread::hardware_concurrency();
            local.threads = std::stoi(get(a, "threads", std::to_string(std::max(1u, hw / size))));
        }
        const std::string block = get(a, "block");
        if (!block.empty()) {
            local.block_x = std::stoi(block.substr(0, block.find('x')));
            local.block_y = std::stoi(block.substr(block.find('x') + 1));
        }
        local = normalized(local);
        if (!impl_available(local.impl)) throw std::runtime_error("local implementation '" + local.impl + "' is not available");

        // Rank 0 loads the image and computes the serial reference; everyone learns the size.
        Image img, ref;
        int dims[2] = {0, 0};
        if (rank == 0) {
            img = load_image(image_path);
            dims[0] = img.w;
            dims[1] = img.h;
            convolve_serial(img, filter, ref);
        }
        MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
        const int W = dims[0], H = dims[1];
        if (size > H) throw std::runtime_error("more ranks than image rows");

        // Row split: the first (H % size) ranks get one extra row.
        std::vector<int> rows(size), first(size), scount(size), sdispl(size), gcount(size), gdispl(size);
        for (int r = 0, y = 0; r < size; ++r) {
            rows[r] = H / size + (r < H % size ? 1 : 0);
            first[r] = y;
            y += rows[r];
            scount[r] = (rows[r] + 2 * R) * W;
            sdispl[r] = first[r] * W;  // strips overlap in the padded image: that overlap is the halo
            gcount[r] = rows[r] * W;
            gdispl[r] = first[r] * W;
        }

        // Padded image: R rows above and below, zero (or replicated edge) to match the border mode.
        std::vector<float> padded;
        if (rank == 0) {
            padded.assign(static_cast<size_t>(H + 2 * R) * W, 0.0f);
            for (int y = 0; y < H; ++y)
                std::copy(img.px.begin() + static_cast<size_t>(y) * W, img.px.begin() + static_cast<size_t>(y + 1) * W,
                          padded.begin() + static_cast<size_t>(y + R) * W);
#ifdef CONVOX_CLAMP_EDGE
            for (int i = 0; i < R; ++i) {
                std::copy(img.px.begin(), img.px.begin() + W, padded.begin() + static_cast<size_t>(i) * W);
                std::copy(img.px.begin() + static_cast<size_t>(H - 1) * W, img.px.end(),
                          padded.begin() + static_cast<size_t>(H + R + i) * W);
            }
#endif
        }

        const int my_rows = rows[rank];
        Image strip, strip_out;
        strip.w = W;
        strip.h = my_rows + 2 * R;
        strip.px.resize(static_cast<size_t>(strip.h) * W);
        Image result;
        if (rank == 0) {
            result.w = W;
            result.h = H;
            result.px.resize(static_cast<size_t>(H) * W);
        }

        auto one_run = [&]() {
            MPI_Barrier(MPI_COMM_WORLD);
            const double t0 = MPI_Wtime();
            MPI_Scatterv(padded.data(), scount.data(), sdispl.data(), MPI_FLOAT, strip.px.data(),
                         static_cast<int>(strip.px.size()), MPI_FLOAT, 0, MPI_COMM_WORLD);
            Timing lt = run_once(local, strip, filter, strip_out);
            MPI_Gatherv(strip_out.px.data() + static_cast<size_t>(R) * W, my_rows * W, MPI_FLOAT, result.px.data(),
                        gcount.data(), gdispl.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
            const double mine = (MPI_Wtime() - t0) * 1e3;
            double loc[4] = {lt.kernel_ms, lt.h2d_ms, lt.d2h_ms, mine}, mx[4] = {0, 0, 0, 0};
            MPI_Reduce(loc, mx, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            return Timing{mx[0], mx[1], mx[2], mx[3]};
        };

        BenchResult b = bench(one_run, reps);

        if (rank == 0) {
            Row row;
            row.cfg = local;
            row.cfg.impl = "mpi-" + (local.impl.rfind("cuda", 0) == 0 ? std::string("cuda") : local.impl);
            row.mpi_ranks = size;
            row.width = W;
            row.height = H;
            row.k = k;
            row.reps = reps;
#ifdef CONVOX_CLAMP_EDGE
            row.border = "clamp";
#else
            row.border = "zero";
#endif
            row.med = b.median;
            row.val = validate(ref, result);
            row.run_id = make_run_id();
            row.timestamp = iso_timestamp_now();
            row.image_id = std::filesystem::path(image_path).stem().string();
            row.filter_type = filter_name;
            std::printf("%-9s %-10s K=%-2d ranks=%-2d local=%-10s compute %9.3f ms  total %9.3f ms  err %.2e  %s\n",
                        row.cfg.impl.c_str(), row.image_id.c_str(), k, size, local.impl.c_str(), row.med.kernel_ms,
                        row.med.total_ms, row.val.max_abs_err, row.val.pass ? "PASS" : "FAIL");
            if (!a.count("no-csv")) {
                const std::string out = get(a, "out", "results/raw/results.csv");
                append_result(out, row);
                append_reps((std::filesystem::path(out).parent_path() / "reps.csv").string(), row.run_id, b.reps);
            }
            MPI_Finalize();
            return row.val.pass ? 0 : 1;
        }
    } catch (const std::exception& e) {
        die(rank, e.what());
    }
    MPI_Finalize();
    return 0;
}
