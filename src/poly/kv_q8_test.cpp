// src/poly/kv_q8_test.cpp - the Q8_0 KV cache kernels of the poly engine, store_q8 and store_q8_at, against a host
// reference.  No model file and no card beyond CUDA device 0:
//   1. round trip: for several row counts and n in {32,128,256}, random floats of mixed magnitude are stored and
//      read back; every dequantized value must be within half a block scale (plus a hair) of the input;
//   2. byte layout: for n=256 the device codes and fp16 scales must match the host's own arithmetic byte for byte,
//      which pins codes[row*n + blk*32 + j] and scales[row*(n/32) + blk];
//   3. store_q8_at with a ring: two rows written from a device `at` land at (*at + r) % ring and equal what plain
//      store_q8 writes from the same position.
// Exit 0 prints "poly_kv_q8_test: PASS"; any failed assertion prints the offending index and values and exits 1.
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "poly_kv_q8_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n ? n : 1), "cudaMalloc"); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() { return static_cast<T*>(p); }
};

uint16_t half_bits(__half h) {
    uint16_t b;
    std::memcpy(&b, &h, sizeof(b));
    return b;
}

float half_value(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, sizeof(h));
    return __half2float(h);
}

// A magnitude drawn over five orders (1e-4 .. 1e3) and either sign, so small and large values mix within a block.
float rand_val(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    const float mag = std::pow(10.f, -4.f + 7.f * u(rng));
    return (u(rng) < 0.5f ? -1.f : 1.f) * mag;
}

// 1. Round trip: dequantize with the fp16-rounded scale read back from `scales` and bound the error.
void round_trip(int rows, int n, std::mt19937& rng, int& fails, long& values) {
    const int nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    Dev dx(x.size() * sizeof(float)), dcodes((size_t) rows * n), dscales((size_t) rows * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    strata::poly::store_q8(dx.as<float>(), dcodes.as<int8_t>(), dscales.as<uint16_t>(), n, 0, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> codes((size_t) rows * n);
    std::vector<uint16_t> scales((size_t) rows * nb);
    ck(cudaMemcpy(codes.data(), dcodes.p, codes.size(), cudaMemcpyDeviceToHost), "download codes");
    ck(cudaMemcpy(scales.data(), dscales.p, scales.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");

    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float d = half_value(scales[(size_t) r * nb + b]);
            const float half_scale = std::fabs(d) * 0.5f;
            for (int j = 0; j < 32; ++j) {
                const size_t i = (size_t) r * n + (size_t) b * 32 + j;
                const float got = (float) codes[i] * d;
                // A q8_0 value cannot land closer than half a block scale.  The scale is written fp16 from the
                // full-precision d the codes were rounded with (the kernel's id = 1/d), so decoding with the fp16
                // scale can miss by up to 127 * 2^-11 block scales at the rounding boundary: allow |x| * 5e-4 too.
                const float tol = half_scale + std::fabs(x[i]) * 5e-4f + 1e-3f;
                ++values;
                if (!(std::fabs(got - x[i]) <= tol)) {
                    std::fprintf(stderr,
                                 "poly_kv_q8_test: round-trip: n=%d rows=%d value [%d,%d,%d] x=%.9g got=%.9g "
                                 "d=%.9g err=%.9g tol=%.9g\n",
                                 n, rows, r, b, j, (double) x[i], (double) got, (double) d,
                                 (double) std::fabs(got - x[i]), (double) tol);
                    ++fails;
                }
            }
        }
    }
}

// 2. Byte layout: the host's own d, codes and fp16 scale must equal the device's bytes.
void byte_layout(int rows, int n, std::mt19937& rng, int& fails, long& values) {
    const int nb = n / 32;
    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    std::vector<int8_t> ecodes((size_t) rows * n);
    std::vector<uint16_t> escales((size_t) rows * nb);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float* xr = x.data() + (size_t) r * n + (size_t) b * 32;
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(xr[j]));
            const float d = amax / 127.0f;
            const float id = d > 0.f ? 1.0f / d : 0.0f;
            escales[(size_t) r * nb + b] = half_bits(__float2half(d));
            int8_t* cr = ecodes.data() + (size_t) r * n + (size_t) b * 32;
            for (int j = 0; j < 32; ++j) {
                float q = std::rint(xr[j] * id);
                q = std::fmin(std::fmax(q, -127.0f), 127.0f);
                cr[j] = (int8_t) q;
            }
        }
    }

    Dev dx(x.size() * sizeof(float)), dcodes((size_t) rows * n), dscales((size_t) rows * nb * sizeof(uint16_t));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    strata::poly::store_q8(dx.as<float>(), dcodes.as<int8_t>(), dscales.as<uint16_t>(), n, 0, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> codes((size_t) rows * n);
    std::vector<uint16_t> scales((size_t) rows * nb);
    ck(cudaMemcpy(codes.data(), dcodes.p, codes.size(), cudaMemcpyDeviceToHost), "download codes");
    ck(cudaMemcpy(scales.data(), dscales.p, scales.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "download scales");

    ++values;
    if (std::memcmp(ecodes.data(), codes.data(), codes.size()) != 0) {
        size_t i = 0;
        while (i < codes.size() && ecodes[i] == codes[i]) ++i;
        const int r = (int) (i / n), rem = (int) (i % n);
        std::fprintf(stderr,
                     "poly_kv_q8_test: byte layout: codes differ at row=%d blk=%d j=%d: host=%d device=%d (n=%d "
                     "rows=%d)\n",
                     r, rem / 32, rem % 32, (int) ecodes[i], (int) codes[i], n, rows);
        ++fails;
    }
    ++values;
    if (std::memcmp(escales.data(), scales.data(), scales.size() * sizeof(uint16_t)) != 0) {
        size_t i = 0;
        while (i < scales.size() && escales[i] == scales[i]) ++i;
        const int r = (int) (i / nb), b = (int) (i % nb);
        std::fprintf(stderr,
                     "poly_kv_q8_test: byte layout: scales differ at row=%d blk=%d: host=0x%04x device=0x%04x "
                     "(n=%d rows=%d)\n",
                     r, b, (unsigned) escales[i], (unsigned) scales[i], n, rows);
        ++fails;
    }
}

// 3. store_q8_at with a ring: rows land at (*at + r) % ring and match plain store_q8 from the same position.
void ring_test(std::mt19937& rng, int& fails, long& values) {
    const int n = 128, nb = n / 32, rows = 2, ring = 5;
    const long long at_val = 4;   // rows go to 4 and 0: the second wraps

    std::vector<float> x((size_t) rows * n);
    for (float& v : x) v = rand_val(rng);

    Dev dx(x.size() * sizeof(float)), d_at(sizeof(long long));
    ck(cudaMemcpy(dx.as<float>(), x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "upload x");
    ck(cudaMemcpy(d_at.p, &at_val, sizeof(at_val), cudaMemcpyHostToDevice), "upload at");

    Dev rcodes((size_t) ring * n), rscales((size_t) ring * nb * sizeof(uint16_t));
    strata::poly::store_q8_at(dx.as<float>(), rcodes.as<int8_t>(), rscales.as<uint16_t>(), n, d_at.as<long long>(),
                              rows, ring, nullptr);
    ck(cudaGetLastError(), "store_q8_at launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8_at sync");

    Dev pcodes((size_t) (at_val + rows) * n), pscales((size_t) (at_val + rows) * nb * sizeof(uint16_t));
    strata::poly::store_q8(dx.as<float>(), pcodes.as<int8_t>(), pscales.as<uint16_t>(), n, at_val, rows, nullptr);
    ck(cudaGetLastError(), "store_q8 launch");
    ck(cudaStreamSynchronize(nullptr), "store_q8 sync");

    std::vector<int8_t> rcode((size_t) ring * n), pcode((size_t) (at_val + rows) * n);
    std::vector<uint16_t> rscale((size_t) ring * nb), pscale((size_t) (at_val + rows) * nb);
    ck(cudaMemcpy(rcode.data(), rcodes.p, rcode.size(), cudaMemcpyDeviceToHost), "download ring codes");
    ck(cudaMemcpy(rscale.data(), rscales.p, rscale.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "download ring scales");
    ck(cudaMemcpy(pcode.data(), pcodes.p, pcode.size(), cudaMemcpyDeviceToHost), "download plain codes");
    ck(cudaMemcpy(pscale.data(), pscales.p, pscale.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
       "download plain scales");

    for (int r = 0; r < rows; ++r) {
        const size_t rr = (size_t) ((at_val + r) % ring);
        const size_t pr = (size_t) (at_val + r);
        ++values;
        if (std::memcmp(rcode.data() + rr * n, pcode.data() + pr * n, (size_t) n) != 0) {
            size_t i = 0;
            while (i < (size_t) n && rcode[rr * n + i] == pcode[pr * n + i]) ++i;
            std::fprintf(stderr,
                         "poly_kv_q8_test: ring: row %d at %zu != plain row %zu: code [%zu] ring=%d plain=%d\n",
                         r, rr, pr, i, (int) rcode[rr * n + i], (int) pcode[pr * n + i]);
            ++fails;
        }
        ++values;
        if (std::memcmp(rscale.data() + rr * nb, pscale.data() + pr * nb, (size_t) nb * sizeof(uint16_t)) != 0) {
            size_t i = 0;
            while (i < (size_t) nb && rscale[rr * nb + i] == pscale[pr * nb + i]) ++i;
            std::fprintf(stderr,
                         "poly_kv_q8_test: ring: row %d at %zu != plain row %zu: scale [%zu] ring=0x%04x "
                         "plain=0x%04x\n",
                         r, rr, pr, i, (unsigned) rscale[rr * nb + i], (unsigned) pscale[pr * nb + i]);
            ++fails;
        }
    }
}

}  // namespace

int main() {
    ck(cudaSetDevice(0), "cudaSetDevice");

    std::mt19937 rng(0x51u);
    int fails = 0;
    long values = 0;

    const int ns[3] = {32, 128, 256};
    const int rss[3] = {1, 3, 7};
    // many trials: 20 sweeps of (1+3+7) rows over three n = 660 random rows
    for (int t = 0; t < 20; ++t)
        for (int ri = 0; ri < 3; ++ri)
            for (int ci = 0; ci < 3; ++ci) round_trip(rss[ri], ns[ci], rng, fails, values);

    byte_layout(2, 256, rng, fails, values);
    byte_layout(3, 256, rng, fails, values);
    byte_layout(5, 256, rng, fails, values);

    ring_test(rng, fails, values);

    if (fails) {
        std::fprintf(stderr, "poly_kv_q8_test: FAIL: %d failed assertion(s)\n", fails);
        return 1;
    }
    std::printf("poly_kv_q8_test: PASS (%ld values checked)\n", values);
    return 0;
}
