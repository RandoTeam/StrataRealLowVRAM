// src/fourgbvram/shortconv_test.cu - shortconv() against the same arithmetic on the host: the outputs, the state it
// leaves, and the snapshots of a window's tokens.  Calls of a few tokens walk the channels, longer ones take a
// thread a token (CONV_ROWS), so both are run, and one long call is checked against the same tokens in two calls.
//
// usage: fourgbvram_shortconv_test   (exit 0: everything within a few units in the last place)
#include "strata/fourgbvram/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using namespace strata::fourgbvram;

constexpr int kChannels = 2048;

// out = C * (s0 * w0 + s1 * w1 + bx * w2), bx = B * x; the state moves on by a token.  `snaps` as shortconv's.
void host_conv(const std::vector<float>& bcx, std::vector<float>& state, const std::vector<float>& w,
               std::vector<float>& out, int rows, std::vector<float>* snap, unsigned snaps) {
    int k = 0;
    for (int row = 0; row < rows; ++row) {
        const float* in = bcx.data() + size_t(row) * 3 * kChannels;
        for (int c = 0; c < kChannels; ++c) {
            float* st = state.data() + size_t(c) * 2;
            const float bx = in[c] * in[2 * kChannels + c];
            const float y = st[0] * w[3 * c] + st[1] * w[3 * c + 1] + bx * w[3 * c + 2];
            st[0] = st[1];
            st[1] = bx;
            out[size_t(row) * kChannels + c] = in[kChannels + c] * y;
        }
        if (snap && (snaps >> row & 1)) {
            std::copy(state.begin(), state.end(), snap->begin() + size_t(k) * 2 * kChannels);
            ++k;
        }
    }
}

float worst(const std::vector<float>& a, const std::vector<float>& b) {
    float w = 0.f;
    for (size_t i = 0; i < a.size(); ++i) w = std::fmax(w, std::fabs(a[i] - b[i]) / (1.f + std::fabs(b[i])));
    return w;
}

template <typename T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

// rows tokens in calls of the sizes in `calls` (they add up to rows), against one pass on the host
int check(int rows, const std::vector<int>& calls, unsigned snaps, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> bcx(size_t(rows) * 3 * kChannels), state(2 * kChannels), w(3 * kChannels);
    for (float& x : bcx) x = nd(rng);
    for (float& x : state) x = nd(rng);
    for (float& x : w) x = 0.5f * nd(rng);
    int n_snap = 0;
    for (unsigned b = snaps; b; b &= b - 1) ++n_snap;
    std::vector<float> want(size_t(rows) * kChannels), want_state = state, want_snap(size_t(n_snap) * 2 * kChannels);
    host_conv(bcx, want_state, w, want, rows, &want_snap, snaps);

    float *d_bcx = upload(bcx), *d_state = upload(state), *d_w = upload(w), *d_out = nullptr, *d_snap = nullptr;
    cudaMalloc(&d_out, want.size() * sizeof(float));
    if (n_snap) cudaMalloc(&d_snap, want_snap.size() * sizeof(float));
    int at = 0;
    for (int n : calls) {
        shortconv(d_bcx + size_t(at) * 3 * kChannels, d_state, d_w, d_out + size_t(at) * kChannels, kChannels, n,
                  nullptr, n_snap ? d_snap : nullptr, 2 * size_t(kChannels), n_snap ? snaps : 0u);
        at += n;
    }
    cudaDeviceSynchronize();
    const float e_out = worst(download(d_out, want.size()), want);
    const float e_state = worst(download(d_state, state.size()), want_state);
    const float e_snap = n_snap ? worst(download(d_snap, want_snap.size()), want_snap) : 0.f;
    cudaFree(d_bcx), cudaFree(d_state), cudaFree(d_w), cudaFree(d_out), cudaFree(d_snap);
    const bool ok = e_out < 1e-5f && e_state == 0.f && e_snap == 0.f;
    std::printf("%s rows %4d in %zu call(s), snaps %#x: out %.2g, state %.2g, snapshots %.2g\n", ok ? "ok  " : "FAIL",
                rows, calls.size(), snaps, e_out, e_state, e_snap);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    int bad = 0;
    for (int rows : {1, 2, 3, 15, 16, 17, 64, 1024}) bad += check(rows, {rows}, 0u, rng);
    bad += check(8, {8}, 0xb5u, rng);                  // a window's tokens, some of them leaving their state
    bad += check(4, {4}, 0x1u, rng);
    bad += check(40, {20, 1, 19}, 0u, rng);            // the state carried from call to call, across both paths
    bad += check(300, {256, 44}, 0u, rng);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
