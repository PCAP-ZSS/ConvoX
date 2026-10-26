#include <cstdio>
#include <cstring>
#include <omp.h>

#ifdef CONVOX_HAVE_CUDA
#include "cuda/device_info.h"
#endif

static int cmd_info() {
    std::printf("ConvoX\n");
    std::printf("Compiler: GCC/Clang %s\n", __VERSION__);
    std::printf("OpenMP:   %d max threads, %d procs\n", omp_get_max_threads(),
                omp_get_num_procs());
#ifdef CONVOX_HAVE_CUDA
    convox_print_cuda_info();
#else
    std::printf("CUDA:     not built\n");
#endif
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "info") == 0) return cmd_info();
    std::fprintf(stderr, "usage: convox info\n");
    return 2;
}
