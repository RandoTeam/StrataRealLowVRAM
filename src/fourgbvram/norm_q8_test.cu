// src/fourgbvram/norm_q8_test.cu - kernels::native_add_rms_norm_q8_1() against the launches it stands for: axpy(),
// rms_norm() and native_quantize_q8_1(), with and without the sum.  The residual, the normed rows and the Q8_1
// blocks must be the same bytes.
//
// usage: fourgbvram_norm_q8_test   (exit 0: every byte the same)
#include "strata/kernels/native_mmvq.hpp"
#include "strata/fourgbvram/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata;

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<unsigned char> bytes(const void* d, size_t n) {
    std::vector<unsigned char> v(n);
    cudaMemcpy(v.data(), d, n, cudaMemcpyDeviceToHost);
    return v;
}

int check(int n, int rows, bool with_add, float spread, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, spread);
    std::vector<float> r((size_t) n * rows), add(r.size()), w(n);
    for (float& x : r) x = nd(rng);
    for (float& x : add) x = nd(rng);
    for (float& x : w) x = 1.f + 0.1f * nd(rng);
    const size_t qb = kernels::native_q8_1_bytes(n) * rows, fb = r.size() * sizeof(float);
    float *r1 = upload(r), *r2 = upload(r), *a = upload(add), *dw = upload(w), *y1 = nullptr, *y2 = nullptr;
    void *q1 = nullptr, *q2 = nullptr;
    cudaMalloc(&y1, fb), cudaMalloc(&y2, fb), cudaMalloc(&q1, qb), cudaMalloc(&q2, qb);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const float eps = 1e-5f;
    if (with_add) fourgbvram::axpy(a, 1.f, r1, n * rows, true, s);
    fourgbvram::rms_norm(r1, dw, y1, n, rows, eps, s);
    kernels::native_quantize_q8_1(y1, q1, n * rows, 1, s);
    kernels::native_add_rms_norm_q8_1(r2, with_add ? a : nullptr, dw, y2, q2, n, rows, eps, fourgbvram::sum_threads(n), s);
    cudaStreamSynchronize(s);
    const bool same_r = bytes(r1, fb) == bytes(r2, fb), same_y = bytes(y1, fb) == bytes(y2, fb),
               same_q = bytes(q1, qb) == bytes(q2, qb);
    cudaFree(r1), cudaFree(r2), cudaFree(a), cudaFree(dw), cudaFree(y1), cudaFree(y2), cudaFree(q1), cudaFree(q2);
    cudaStreamDestroy(s);
    const bool ok = same_r && same_y && same_q;
    std::printf("%s n %5d rows %4d %s spread %g: residual %s, normed %s, q8_1 %s\n", ok ? "ok  " : "FAIL", n, rows,
                with_add ? "add " : "norm", spread, same_r ? "same" : "DIFF", same_y ? "same" : "DIFF",
                same_q ? "same" : "DIFF");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(11);
    int bad = 0;
    for (int n : {512, 1024, 2048, 2816, 4096, 5120})
        for (int rows : {1, 3, 8, 64})
            for (bool add : {false, true}) bad += check(n, rows, add, 1.f, rng);
    bad += check(2048, 1024, true, 1.f, rng);   // a prompt batch
    bad += check(2048, 8, true, 300.f, rng);    // large activations
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
