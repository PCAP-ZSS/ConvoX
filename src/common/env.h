#pragma once

#include <string>

// Description of the machine and toolchain, recorded with every result row.
struct Env {
    std::string cpu_model, gpu_model, compiler, cuda_version, git_commit;
};

const Env& env();
