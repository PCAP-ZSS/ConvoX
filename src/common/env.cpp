#include "common/env.h"

#include <fstream>

#ifdef CONVOX_HAVE_CUDA
#include "cuda/device_info.h"
#endif

#ifndef CONVOX_GIT_COMMIT
#define CONVOX_GIT_COMMIT "unknown"
#endif

static std::string read_cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0) {
            size_t p = line.find(':');
            if (p != std::string::npos) return line.substr(p + 2);
        }
    }
    return "unknown";
}

const Env& env() {
    static const Env e = [] {
        Env x;
        x.cpu_model = read_cpu_model();
        x.compiler = std::string("gcc ") + __VERSION__;
#ifdef CONVOX_HAVE_CUDA
        x.gpu_model = convox_cuda_device_name();
        x.cuda_version = convox_cuda_version_string();
#endif
        x.git_commit = CONVOX_GIT_COMMIT;
        return x;
    }();
    return e;
}
