// src/fourgbvram/attend_test.cu - attend()'s blocks of rows for LFM2's heads (64 values, groups of 4) against the score,
// softmax and value kernels it takes without the room for them, and against itself: a token alone, with the
// tokens of a call beside it, and with the position on the card.  The last three must be the same bits.
//
// usage: fourgbvram_attend_test   (exit 0: within 1e-5 of the other kernels and the same bits)
#include "strata/fourgbvram/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace {

using namespace strata::fourgbvram;

constexpr int kDim = 64, kHeads = 32, kKvHeads = 8, kChunk = 8, kMaxRows = 32768, kBatch = 1024;

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

std::vector<float> download(const float* d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return v;
}

struct Buffers {
    float *q, *score, *out;
    uint16_t *k, *v;
    long long* at;
    size_t score_n;
};

// rows tokens from pos: the other kernels, the blocks of rows, each token alone, and the blocks with `at`
int check(const Buffers& b, long long pos, int rows) {
    const size_t n = size_t(rows) * kHeads * kDim;
    const float scale = 0.125f;
    attend(b.q, b.k, b.v, pos, rows, b.score, kChunk, b.out, kDim, kHeads, kKvHeads, scale, nullptr);
    const std::vector<float> other = download(b.out, n);
    attend(b.q, b.k, b.v, pos, rows, b.score, kChunk, b.out, kDim, kHeads, kKvHeads, scale, nullptr, 0, 0, false,
           nullptr, 0, b.score_n);
    const std::vector<float> split = download(b.out, n);
    std::vector<float> alone(n);
    for (int t = 0; t < rows; ++t) {
        const size_t o = size_t(t) * kHeads * kDim;
        attend(b.q + o, b.k, b.v, pos + t, 1, b.score, kChunk, b.out, kDim, kHeads, kKvHeads, scale, nullptr, 0, 0,
               false, nullptr, 0, b.score_n);
        cudaMemcpy(alone.data() + o, b.out, kHeads * kDim * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(b.at, &pos, sizeof pos, cudaMemcpyHostToDevice);
    const int room = int(std::min<long long>(pos + rows + 100, kMaxRows));
    attend(b.q, b.k, b.v, 0, rows, b.score, kChunk, b.out, kDim, kHeads, kKvHeads, scale, nullptr, 0, 0, false, b.at,
           room, b.score_n);
    const std::vector<float> card = download(b.out, n);
    float most = 0.f;
    for (size_t i = 0; i < n; ++i) most = std::max(most, std::fabs(split[i] - other[i]));
    const bool same_alone = !std::memcmp(split.data(), alone.data(), n * sizeof(float)),
               same_card = !std::memcmp(split.data(), card.data(), n * sizeof(float)),
               ok = most <= 1e-5f && same_alone && same_card;
    std::printf("%s pos %5lld rows %2d: %.2e from the other kernels, alone %s, on the card %s\n", ok ? "ok  " : "FAIL",
                pos, rows, most, same_alone ? "same" : "DIFF", same_card ? "same" : "DIFF");
    return ok ? 0 : 1;
}

// a prompt batch of rows tokens from pos through attend_batch(), against attend(): the queries and the weights go
// in as halves there, so the results differ in the third or fourth digit
int check_batch(const Buffers& b, long long pos, int rows) {
    const size_t n = size_t(rows) * kHeads * kDim;
    const float scale = 0.125f;
    std::vector<float> ref(n);
    for (int first = 0; first < rows; first += kChunk) {
        const int c = std::min(kChunk, rows - first);
        const size_t o = size_t(first) * kHeads * kDim;
        attend(b.q + o, b.k, b.v, pos + first, c, b.score, kChunk, b.out, kDim, kHeads, kKvHeads, scale, nullptr);
        cudaMemcpy(ref.data() + o, b.out, size_t(c) * kHeads * kDim * sizeof(float), cudaMemcpyDeviceToHost);
    }
    attend_batch(nullptr, b.q, b.k, b.v, pos, rows, b.score, b.score_n, nullptr, b.out, kDim, kHeads, kKvHeads, scale,
                 nullptr);
    const std::vector<float> batch = download(b.out, n);
    float most = 0.f, big = 0.f;
    for (size_t i = 0; i < n; ++i) {
        most = std::max(most, std::fabs(batch[i] - ref[i]));
        big = std::max(big, std::fabs(ref[i]));
    }
    const bool ok = most <= 4e-3f;
    std::printf("%s batch pos %5lld rows %4d: %.2e from attend() (results up to %.2f)\n", ok ? "ok  " : "FAIL", pos,
                rows, most, big);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q(size_t(kBatch) * kHeads * kDim);
    for (float& x : q) x = 2.f * nd(rng);   // spread scores, so the softmax has a few large weights
    std::vector<uint16_t> k(size_t(kMaxRows) * kKvHeads * kDim), v(k.size());
    for (uint16_t& x : k) x = __half_as_ushort(__float2half(nd(rng)));
    for (uint16_t& x : v) x = __half_as_ushort(__float2half(nd(rng)));
    Buffers b{};
    b.q = upload(q), b.k = upload(k), b.v = upload(v);
    b.score_n = std::max(size_t(kChunk) * kHeads * kMaxRows, attend_split_floats(kChunk, kHeads, kMaxRows, kDim));
    cudaMalloc(&b.score, b.score_n * sizeof(float));
    cudaMalloc(&b.out, q.size() * sizeof(float));
    cudaMalloc(&b.at, sizeof(long long));
    int bad = 0;
    for (long long pos : {0LL, 5LL, 63LL, 64LL, 1000LL, 4095LL, 30000LL})
        for (int rows : {1, 2, 4, 8}) bad += check(b, pos, rows);
    bad += check(b, 2000, 13);                  // two launches of a chunk
    bad += check(b, kMaxRows - 16, 16);         // to the end of the cache
    const std::pair<long long, int> batches[] = {{0, 37}, {0, kBatch}, {1000, 517}, {30000, 700}};
    for (const auto& [pos, rows] : batches) bad += check_batch(b, pos, rows);
    cudaFree(b.q), cudaFree(b.k), cudaFree(b.v), cudaFree(b.score), cudaFree(b.out), cudaFree(b.at);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
