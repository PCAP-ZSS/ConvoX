// Checks the timing harness (median, warm-up handling) and the CSV writer.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "common/csv.h"
#include "common/timing.h"

static int failures = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    // Warm-up call is discarded; the 10 measured values are 1..10, median 5.5.
    int calls = 0;
    BenchResult b = bench([&] {
        ++calls;
        double v = calls - 1;  // call 1 (warm-up) returns 0, then 1..10
        return Timing{v, 0, 0, v};
    });
    check(calls == 11, "1 warm-up + 10 measured calls");
    check(b.reps.size() == 10, "warm-up excluded from reps");
    check(b.median.kernel_ms == 5.5, "median of 1..10 is 5.5");

    // Odd count median is the middle element, outliers are ignored.
    int c2 = 0;
    const double vals[] = {100, 3, 1, 2, 50};
    BenchResult b2 = bench([&] { double v = vals[c2++ % 5]; return Timing{v, 0, 0, v}; }, 5, 0);
    check(b2.median.total_ms == 3, "median ignores outliers");

    // CSV: header written once, rows appended.
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "convox_test_csv";
    fs::remove_all(dir);
    Row r;
    r.run_id = make_run_id();
    r.timestamp = iso_timestamp_now();
    r.image_id = "img";
    r.filter_type = "gaussian";
    r.cfg.impl = "serial";
    r.width = 4;
    r.height = 4;
    r.k = 3;
    r.reps = 10;
    r.border = "zero";
    r.val.pass = true;
    std::string path = (dir / "results.csv").string();
    append_result(path, r);
    append_result(path, r);
    std::ifstream f(path);
    std::string line;
    int lines = 0, header_cols = 0, row_cols = 0;
    while (std::getline(f, line)) {
        int cols = 1;
        bool inq = false;
        for (char ch : line) {
            if (ch == '"') inq = !inq;
            if (ch == ',' && !inq) ++cols;
        }
        if (lines == 0) header_cols = cols; else row_cols = cols;
        ++lines;
    }
    check(lines == 3, "header + 2 rows");
    check(header_cols == 33, "33 columns in header");
    check(row_cols == header_cols, "rows match header width");
    append_reps((dir / "reps.csv").string(), r.run_id, b.reps);
    check(fs::file_size(dir / "reps.csv") > 0, "reps.csv written");
    fs::remove_all(dir);

    std::printf(failures ? "%d check(s) failed\n" : "all timing/csv checks passed\n", failures);
    return failures ? 1 : 0;
}
