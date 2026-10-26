#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "common/filter.h"
#include "cuda/conv_cuda.h"

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess)                                                              \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + \
                                     " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")"); \
    } while (0)

__constant__ float c_filter[MAX_K * MAX_K];

// Read a pixel with the project's border rule.
__device__ __forceinline__ float fetch(const float* in, int W, int H, int y, int x) {
#ifdef CONVOX_CLAMP_EDGE
    y = y < 0 ? 0 : (y >= H ? H - 1 : y);
    x = x < 0 ? 0 : (x >= W ? W - 1 : x);
    return in[static_cast<size_t>(y) * W + x];
#else
    if (y < 0 || y >= H || x < 0 || x >= W) return 0.0f;
    return in[static_cast<size_t>(y) * W + x];
#endif
}

// Stage 3: everything from global memory.
__global__ void k_naive(const float* in, const float* filt, float* out, int W, int H, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int R = K / 2;
    float acc = 0.0f;
    for (int i = -R; i <= R; ++i)
        for (int j = -R; j <= R; ++j) acc += fetch(in, W, H, y + i, x + j) * filt[(i + R) * K + (j + R)];
    out[static_cast<size_t>(y) * W + x] = acc;
}

// Stage 4a: filter coefficients from constant memory (warp-uniform broadcast reads).
__global__ void k_const(const float* in, float* out, int W, int H, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int R = K / 2;
    float acc = 0.0f;
    for (int i = -R; i <= R; ++i)
        for (int j = -R; j <= R; ++j) acc += fetch(in, W, H, y + i, x + j) * c_filter[(i + R) * K + (j + R)];
    out[static_cast<size_t>(y) * W + x] = acc;
}

// Stage 4b: constant-memory filter plus a shared-memory tile of (BX+K-1) x (BY+K-1) inputs.
__global__ void k_tiled(const float* in, float* out, int W, int H, int K) {
    extern __shared__ float tile[];
    const int R = K / 2;
    const int TW = blockDim.x + K - 1, TH = blockDim.y + K - 1;
    const int x0 = blockIdx.x * blockDim.x - R, y0 = blockIdx.y * blockDim.y - R;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    for (int idx = tid; idx < TW * TH; idx += nthreads)
        tile[idx] = fetch(in, W, H, y0 + idx / TW, x0 + idx % TW);
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j) acc += tile[(threadIdx.y + i) * TW + threadIdx.x + j] * c_filter[i * K + j];
    out[static_cast<size_t>(y) * W + x] = acc;
}

bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

bool is_cuda_impl(const std::string& impl) {
    return impl == "cuda-naive" || impl == "cuda-const" || impl == "cuda-tiled";
}

namespace {

struct DevBuf {
    float* p = nullptr;
    explicit DevBuf(size_t n) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(float))); }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

struct Events {
    cudaEvent_t e[4];
    Events() {
        for (auto& ev : e) CUDA_CHECK(cudaEventCreate(&ev));
    }
    ~Events() {
        for (auto ev : e) cudaEventDestroy(ev);
    }
    float ms(int a, int b) const {
        float t = 0;
        cudaEventElapsedTime(&t, e[a], e[b]);
        return t;
    }
};

void check_launch(const Config& cfg, int k) {
    if (cfg.block_x < 1 || cfg.block_y < 1 || cfg.block_x * cfg.block_y > 1024)
        throw std::runtime_error("invalid CUDA block " + std::to_string(cfg.block_x) + "x" +
                                 std::to_string(cfg.block_y) + " (need 1..1024 threads)");
    if (cfg.impl == "cuda-tiled") {
        size_t smem = static_cast<size_t>(cfg.block_x + k - 1) * (cfg.block_y + k - 1) * sizeof(float);
        int dev = 0, limit = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlock, dev));
        if (smem > static_cast<size_t>(limit))
            throw std::runtime_error("tile needs " + std::to_string(smem) + " B shared memory, limit is " +
                                     std::to_string(limit) + " B");
    }
}

}  // namespace

Timing convolve_cuda(const Image& in, const Filter& f, const Config& cfg, Image& out) {
    if (!is_cuda_impl(cfg.impl)) throw std::runtime_error("not a CUDA implementation: " + cfg.impl);
    check_launch(cfg, f.k);

    const int W = in.w, H = in.h, K = f.k;
    const size_t n = static_cast<size_t>(W) * H;
    out.w = W;
    out.h = H;
    out.px.resize(n);

    DevBuf d_in(n), d_out(n);
    float* d_filt = nullptr;
    if (cfg.impl == "cuda-naive") CUDA_CHECK(cudaMalloc(&d_filt, f.w.size() * sizeof(float)));

    dim3 block(cfg.block_x, cfg.block_y);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    const size_t smem = cfg.impl == "cuda-tiled"
                            ? static_cast<size_t>(block.x + K - 1) * (block.y + K - 1) * sizeof(float)
                            : 0;

    Events ev;
    CUDA_CHECK(cudaDeviceSynchronize());  // nothing pending when the host clock starts
    Stopwatch host;
    CUDA_CHECK(cudaEventRecord(ev.e[0]));
    CUDA_CHECK(cudaMemcpy(d_in.p, in.px.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    if (cfg.impl == "cuda-naive")
        CUDA_CHECK(cudaMemcpy(d_filt, f.w.data(), f.w.size() * sizeof(float), cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemcpyToSymbol(c_filter, f.w.data(), f.w.size() * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(ev.e[1]));

    if (cfg.impl == "cuda-naive")
        k_naive<<<grid, block>>>(d_in.p, d_filt, d_out.p, W, H, K);
    else if (cfg.impl == "cuda-const")
        k_const<<<grid, block>>>(d_in.p, d_out.p, W, H, K);
    else
        k_tiled<<<grid, block, smem>>>(d_in.p, d_out.p, W, H, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev.e[2]));

    CUDA_CHECK(cudaMemcpy(out.px.data(), d_out.p, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev.e[3]));
    CUDA_CHECK(cudaDeviceSynchronize());
    double total = host.ms();

    if (d_filt) cudaFree(d_filt);
    Timing t;
    t.h2d_ms = ev.ms(0, 1);
    t.kernel_ms = ev.ms(1, 2);
    t.d2h_ms = ev.ms(2, 3);
    t.total_ms = total;
    return t;
}

CudaLaunchInfo cuda_launch_info(const Config& cfg, int k) {
    CudaLaunchInfo info;
    if (!is_cuda_impl(cfg.impl) || !cuda_available()) return info;
    check_launch(cfg, k);
    info.smem_bytes = cfg.impl == "cuda-tiled"
                          ? (cfg.block_x + k - 1) * (cfg.block_y + k - 1) * static_cast<int>(sizeof(float))
                          : 0;
    const int threads = cfg.block_x * cfg.block_y;
    int blocks = 0;
    if (cfg.impl == "cuda-naive")
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k_naive, threads, 0));
    else if (cfg.impl == "cuda-const")
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k_const, threads, 0));
    else
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k_tiled, threads, info.smem_bytes));
    int dev = 0, max_threads = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    info.occupancy = static_cast<double>(blocks * threads) / max_threads;
    return info;
}
