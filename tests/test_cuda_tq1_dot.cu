// tests/test_cuda_tq1_dot.cu - Native CUDA TQ1_0 (GGML Type 34) verification test.
// Verifies GPU execution of TQ1_0 x Q8_1 dot products, MMVQ, and dequantization.

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

static int g_fails = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        std::printf("[FAIL] Line %d: " msg "\n", __LINE__, ##__VA_ARGS__); \
        ++g_fails; \
    } \
} while (0)

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::printf("[CUDA ERROR] %s: %s\n", #call, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

static const uint8_t k_pow3[6] = {1, 3, 9, 27, 81, 243};

// Reference TQ1_0 quantizer
void quantize_row_tq1_0_ref(const float* x, k::block_tq1_0* y, int64_t n) {
    assert(n % 256 == 0);
    const int64_t nb = n / 256;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < 256; j++) {
            amax = fmaxf(amax, fabsf(x[j]));
        }

        const float d = amax;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = k::f16_from_f32(d);

        // 5 elements per byte, along 32 bytes (160 elements)
        for (size_t j = 0; j < 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n_trit = 0; n_trit < 5; ++n_trit) {
                    int xi = (int) lroundf(x[m + n_trit * 32] * id) + 1;
                    q *= 3;
                    q += (uint8_t) xi;
                }
                q = (uint8_t) (((uint16_t) q * 256 + (243 - 1)) / 243);
                y[i].qs[j + m] = q;
            }
            x += 5 * 32;
        }

        // along 16 bytes (80 elements)
        for (size_t j = 32; j < 48; j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n_trit = 0; n_trit < 5; ++n_trit) {
                    int xi = (int) lroundf(x[m + n_trit * 16] * id) + 1;
                    q *= 3;
                    q += (uint8_t) xi;
                }
                q = (uint8_t) (((uint16_t) q * 256 + (243 - 1)) / 243);
                y[i].qs[j + m] = q;
            }
            x += 5 * 16;
        }

        // 4 elements per byte in qh (16 elements)
        for (size_t j = 0; j < 4; ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                int xi = (int) lroundf(x[j + m * 4] * id) + 1;
                q *= 3;
                q += (uint8_t) xi;
            }
            q *= 3;
            q = (uint8_t) (((uint16_t) q * 256 + (243 - 1)) / 243);
            y[i].qh[j] = q;
        }
        x += 16;
    }
}

// Reference TQ1_0 dequantizer
void dequantize_row_tq1_0_ref(const k::block_tq1_0* x, float* y, int64_t n) {
    assert(n % 256 == 0);
    const int64_t nb = n / 256;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = k::f32_from_f16(x[i].d);

        for (size_t n_trit = 0; n_trit < 5; ++n_trit) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = (uint8_t)(x[i].qs[m] * k_pow3[n_trit]);
                int16_t xi = (int16_t)(((uint16_t) q * 3) >> 8);
                *y++ = (float)(xi - 1) * d;
            }
        }
        for (size_t n_trit = 0; n_trit < 5; ++n_trit) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = (uint8_t)(x[i].qs[32 + m] * k_pow3[n_trit]);
                int16_t xi = (int16_t)(((uint16_t) q * 3) >> 8);
                *y++ = (float)(xi - 1) * d;
            }
        }
        for (size_t n_trit = 0; n_trit < 4; ++n_trit) {
            for (size_t j = 0; j < 4; ++j) {
                uint8_t q = (uint8_t)(x[i].qh[j] * k_pow3[n_trit]);
                int16_t xi = (int16_t)(((uint16_t) q * 3) >> 8);
                *y++ = (float)(xi - 1) * d;
            }
        }
    }
}

// Dequantize Q8_1 activations to double on CPU
std::vector<double> dequant_q8_1(const std::vector<uint8_t>& q, size_t n) {
    std::vector<double> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, &q[b * 36], 2);
        const double d = k::f32_from_f16(dh);
        for (int i = 0; i < 32; ++i) v[b * 32 + i] = d * (int8_t) q[b * 36 + 4 + i];
    }
    return v;
}

void test_dequantize() {
    std::printf("Testing iq_dequant_f32 and iq_dequant_f16 for TQ1_0 on GPU...\n");
    const int n = 1024;
    const int nb = n / 256;
    std::vector<float> orig_x(n);
    for (int i = 0; i < n; ++i) {
        orig_x[i] = (float)((i % 3) - 1) * 0.15f;
    }

    std::vector<k::block_tq1_0> h_tq(nb);
    quantize_row_tq1_0_ref(orig_x.data(), h_tq.data(), n);

    std::vector<float> ref_deq(n);
    dequantize_row_tq1_0_ref(h_tq.data(), ref_deq.data(), n);

    k::block_tq1_0* d_tq = nullptr;
    float* d_out_f32 = nullptr;
    uint16_t* d_out_f16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tq, nb * sizeof(k::block_tq1_0)));
    CUDA_CHECK(cudaMalloc(&d_out_f32, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_f16, n * sizeof(uint16_t)));

    CUDA_CHECK(cudaMemcpy(d_tq, h_tq.data(), nb * sizeof(k::block_tq1_0), cudaMemcpyHostToDevice));

    // Test f32 dequant
    k::iq_dequant_f32(k::GGML_TYPE_TQ1_0, d_tq, n, d_out_f32, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_deq_f32(n);
    CUDA_CHECK(cudaMemcpy(gpu_deq_f32.data(), d_out_f32, n * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff_f32 = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(ref_deq[i] - gpu_deq_f32[i]);
        if (diff > max_diff_f32) max_diff_f32 = diff;
    }
    CHECK(max_diff_f32 < 1e-6f, "iq_dequant_f32 max diff: %e", max_diff_f32);
    std::printf("  iq_dequant_f32 diff: %e PASS\n", max_diff_f32);

    // Test f16 dequant
    k::iq_dequant_f16(k::GGML_TYPE_TQ1_0, d_tq, n, d_out_f16, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> gpu_deq_f16(n);
    CUDA_CHECK(cudaMemcpy(gpu_deq_f16.data(), d_out_f16, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    float max_diff_f16 = 0.0f;
    for (int i = 0; i < n; ++i) {
        float f16_val = k::f32_from_f16(gpu_deq_f16[i]);
        float diff = std::fabs(ref_deq[i] - f16_val);
        if (diff > max_diff_f16) max_diff_f16 = diff;
    }
    CHECK(max_diff_f16 < 1e-4f, "iq_dequant_f16 max diff: %e", max_diff_f16);
    std::printf("  iq_dequant_f16 diff: %e PASS\n", max_diff_f16);

    cudaFree(d_tq);
    cudaFree(d_out_f32);
    cudaFree(d_out_f16);
}

void test_mmvq() {
    std::printf("Testing iq_mmvq for TQ1_0 on GPU (single and multi-column)...\n");
    std::mt19937 rng(42);
    const int n_in = 2560; // Hidden size (multiple of 256)
    const int n_out = 64;  // Number of output rows
    const int nb = n_in / 256;

    std::vector<float> w_float((size_t) n_out * n_in);
    for (auto& val : w_float) {
        val = (float)((rand() % 3) - 1) * 0.05f;
    }

    std::vector<k::block_tq1_0> w_tq((size_t) n_out * nb);
    for (int r = 0; r < n_out; ++r) {
        quantize_row_tq1_0_ref(&w_float[(size_t) r * n_in], &w_tq[(size_t) r * nb], n_in);
    }

    k::block_tq1_0* d_w = nullptr;
    const size_t w_bytes = w_tq.size() * sizeof(k::block_tq1_0);
    CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
    CUDA_CHECK(cudaMemcpy(d_w, w_tq.data(), w_bytes, cudaMemcpyHostToDevice));

    // Dequantize weights on CPU for reference
    std::vector<float> w_deq((size_t) n_out * n_in);
    for (int r = 0; r < n_out; ++r) {
        dequantize_row_tq1_0_ref(&w_tq[(size_t) r * nb], &w_deq[(size_t) r * n_in], n_in);
    }

    const int max_cols = 8;
    std::vector<float> x_float((size_t) max_cols * n_in);
    for (auto& val : x_float) {
        val = ((float)(rand() % 200 - 100) / 100.0f) * 1.5f;
    }

    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, x_float.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x_float.data(), x_float.size() * sizeof(float), cudaMemcpyHostToDevice));

    const size_t xq_bytes = (size_t) max_cols * (n_in / 32) * 36;
    uint8_t* d_xq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xq, xq_bytes));
    k::quantize_q8_1_rows(d_x, max_cols, n_in, d_xq, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> h_xq(xq_bytes);
    CUDA_CHECK(cudaMemcpy(h_xq.data(), d_xq, xq_bytes, cudaMemcpyDeviceToHost));
    const auto x_deq = dequant_q8_1(h_xq, (size_t) max_cols * n_in);

    float* d_y_old = nullptr;
    float* d_y_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_y_old, (size_t) max_cols * n_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_new, (size_t) max_cols * n_out * sizeof(float)));

    std::vector<float> h_y_old((size_t) max_cols * n_out);
    std::vector<float> h_y_new((size_t) max_cols * n_out);

    const int test_cols[] = {1, 2, 4, 8};
    for (int nc : test_cols) {
        CUDA_CHECK(cudaMemset(d_y_old, 0, (size_t) max_cols * n_out * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_y_new, 0, (size_t) max_cols * n_out * sizeof(float)));

        // Run old kernel (single-column unrolled)
        k::iq_set_old_kernels(true);
        k::iq_mmvq(k::GGML_TYPE_TQ1_0, d_w, d_xq, d_y_old, n_in, n_out, nc, nullptr);

        // Run new kernel (multi-column Split<34>)
        k::iq_set_old_kernels(false);
        k::iq_mmvq(k::GGML_TYPE_TQ1_0, d_w, d_xq, d_y_new, n_in, n_out, nc, nullptr);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_y_old.data(), d_y_old, (size_t) nc * n_out * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y_new.data(), d_y_new, (size_t) nc * n_out * sizeof(float), cudaMemcpyDeviceToHost));

        // 1. Bitwise comparison between old and new kernels
        size_t bitwise_diff = 0;
        for (size_t i = 0; i < (size_t) nc * n_out; ++i) {
            if (std::memcmp(&h_y_old[i], &h_y_new[i], sizeof(float)) != 0) {
                bitwise_diff++;
            }
        }
        CHECK(bitwise_diff == 0, "ncols=%d: Old and new kernels differ by %zu floats!", nc, bitwise_diff);

        // 2. Parity against CPU double reference
        double num = 0, den = 0;
        for (int c = 0; c < nc; ++c) {
            for (int r = 0; r < n_out; ++r) {
                double ref = 0;
                for (int i = 0; i < n_in; ++i) {
                    ref += (double) w_deq[(size_t) r * n_in + i] * x_deq[(size_t) c * n_in + i];
                }
                float gpu = h_y_new[(size_t) c * n_out + r];
                num += std::fabs(gpu - ref);
                den += std::fabs(ref);
            }
        }
        double rel_err = num / (den + 1e-30);
        CHECK(rel_err < 1e-4, "ncols=%d: Relative error vs CPU ref: %e", nc, rel_err);
        std::printf("  ncols=%d: bitwise match with old=%s, ref rel err=%.2e PASS\n",
                    nc, bitwise_diff == 0 ? "YES" : "NO", rel_err);
    }

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_xq);
    cudaFree(d_y_old);
    cudaFree(d_y_new);
}

void test_expert_layout() {
    std::printf("Testing NativeExpertLayout for TQ1_0 (34)...\n");
    const int64_t n_embd = 2560;
    const int64_t n_ff_512 = 512;
    const int64_t n_ff_640 = 640;

    // TQ1_0 gate/up and TQ1_0 down (requires n_ff divisible by 256)
    CHECK(k::native_expert_supported(34, 34, n_embd, n_ff_512), "native_expert_supported(34, 34, 2560, 512)");

    // TQ1_0 gate/up and IQ4_NL down (supports n_ff divisible by 32, e.g. 640)
    CHECK(k::native_expert_supported(34, 20, n_embd, n_ff_640), "native_expert_supported(34, 20, 2560, 640)");

    auto L = k::native_expert_layout(34, 34, n_embd, n_ff_512);
    CHECK(L.gu_type == 34, "gu_type == 34");
    CHECK(L.d_type == 34, "d_type == 34");
    CHECK(L.gu_row == (size_t)(n_embd / 256) * 54, "gu_row calculation");
    CHECK(L.d_row == (size_t)(n_ff_512 / 256) * 54, "d_row calculation");
    CHECK(L.up_off == (size_t) n_ff_512 * L.gu_row, "up_off");
    CHECK(L.down_off == 2 * L.up_off, "down_off");
    CHECK(L.bytes == L.down_off + (size_t) n_embd * L.d_row, "total bytes");

    std::printf("  gu_row=%zu, d_row=%zu, bytes=%zu PASS\n", L.gu_row, L.d_row, L.bytes);
}

void test_grouped_experts() {
    std::printf("Testing native_expert_grouped with TQ1_0 gate/up and down on GPU...\n");
    const int64_t H = 2560;
    const int64_t FF = 512;
    auto L = k::native_expert_layout(34, 34, H, FF);

    const int n_groups = 4;
    const int cap_groups = 6;
    const int n_tokens = 8;
    const int n_entries = 8;
    const int cap_entries = 10;

    // Allocate blob for each group
    std::vector<std::vector<uint8_t>> h_blobs(n_groups, std::vector<uint8_t>(L.bytes));
    std::vector<uint8_t*> d_blobs(n_groups);
    std::vector<unsigned long long> d_ptrs(n_groups);

    std::mt19937 rng(1234);
    for (int g = 0; g < n_groups; ++g) {
        // Fill random bytes, but valid fp16 scales
        for (auto& b : h_blobs[g]) b = (uint8_t)(rng() % 256);
        // Set fp16 scales in every 54-byte block
        for (size_t o = 0; o < L.bytes; o += 54) {
            uint16_t sc = k::f16_from_f32(0.01f + (float)(rng() % 100) * 0.001f);
            std::memcpy(&h_blobs[g][o + 52], &sc, 2);
        }
        CUDA_CHECK(cudaMalloc(&d_blobs[g], L.bytes));
        CUDA_CHECK(cudaMemcpy(d_blobs[g], h_blobs[g].data(), L.bytes, cudaMemcpyHostToDevice));
        d_ptrs[g] = (unsigned long long) d_blobs[g];
    }

    unsigned long long* d_grp_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grp_ptr, cap_groups * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_grp_ptr, d_ptrs.data(), n_groups * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    std::vector<int32_t> h_grp_start = {0, 2, 4, 6, 8}; // 2 entries per group
    int32_t* d_grp_start = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grp_start, (cap_groups + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_grp_start, h_grp_start.data(), (n_groups + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));

    int32_t* d_n_groups = nullptr;
    CUDA_CHECK(cudaMalloc(&d_n_groups, sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_n_groups, &n_groups, sizeof(int32_t), cudaMemcpyHostToDevice));

    std::vector<int32_t> h_ent_dst = {0, 1, 2, 3, 4, 5, 6, 7};
    int32_t* d_ent_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ent_dst, cap_entries * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_ent_dst, h_ent_dst.data(), n_entries * sizeof(int32_t), cudaMemcpyHostToDevice));

    std::vector<int32_t> h_ent_tok = {0, 1, 2, 3, 4, 5, 6, 7};
    int32_t* d_ent_tok = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ent_tok, cap_entries * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_ent_tok, h_ent_tok.data(), n_entries * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Tokens input
    std::vector<float> h_x(n_tokens * H);
    for (auto& val : h_x) val = ((float)(rng() % 200 - 100) / 100.0f);
    float* d_x = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

    const size_t xq_bytes = (size_t) n_tokens * (H / 32) * 36;
    uint8_t* d_xq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xq, xq_bytes));
    k::quantize_q8_1_rows(d_x, n_tokens, H, d_xq, nullptr);

    const size_t scr_bytes = k::native_expert_scratch_bytes(cap_entries, FF);
    void* d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scratch, scr_bytes));

    const size_t out_floats = (size_t) cap_entries * H;
    float *d_out_old = nullptr, *d_out_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_old, out_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_new, out_floats * sizeof(float)));

    // Run old kernels
    CUDA_CHECK(cudaMemset(d_out_old, 0, out_floats * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_scratch, 0, scr_bytes));
    k::iq_set_old_kernels(true);
    k::native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                             cap_groups, cap_entries, d_xq, d_scratch, d_out_old, nullptr);

    // Run new kernels
    CUDA_CHECK(cudaMemset(d_out_new, 0, out_floats * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_scratch, 0, scr_bytes));
    k::iq_set_old_kernels(false);
    k::native_expert_grouped(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                             cap_groups, cap_entries, d_xq, d_scratch, d_out_new, nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out_old(out_floats), h_out_new(out_floats);
    CUDA_CHECK(cudaMemcpy(h_out_old.data(), d_out_old, out_floats * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_new.data(), d_out_new, out_floats * sizeof(float), cudaMemcpyDeviceToHost));

    size_t bitwise_diff = 0;
    bool all_finite = true;
    for (size_t i = 0; i < (size_t) n_entries * H; ++i) {
        if (std::memcmp(&h_out_old[i], &h_out_new[i], sizeof(float)) != 0) bitwise_diff++;
        if (!std::isfinite(h_out_new[i])) all_finite = false;
    }
    CHECK(bitwise_diff == 0, "native_expert_grouped old vs new differ by %zu floats!", bitwise_diff);
    CHECK(all_finite, "native_expert_grouped produced non-finite outputs!");
    std::printf("  native_expert_grouped (4 groups, 8 entries): bitwise match=%s, all finite=%s PASS\n",
                bitwise_diff == 0 ? "YES" : "NO", all_finite ? "YES" : "NO");

    for (int g = 0; g < n_groups; ++g) cudaFree(d_blobs[g]);
    cudaFree(d_grp_ptr); cudaFree(d_grp_start); cudaFree(d_n_groups);
    cudaFree(d_ent_dst); cudaFree(d_ent_tok); cudaFree(d_x); cudaFree(d_xq);
    cudaFree(d_scratch); cudaFree(d_out_old); cudaFree(d_out_new);
}

int main() {
    std::printf("============================================================\n");
    std::printf("  CUDA Native TQ1_0 (Type 34) Kernel Verification Suite     \n");
    std::printf("============================================================\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::printf("No CUDA device found. Skipping GPU test.\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("GPU: %s (Compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

    test_expert_layout();
    test_dequantize();
    test_mmvq();
    test_grouped_experts();

    std::printf("============================================================\n");
    if (g_fails == 0) {
        std::printf("  ALL CUDA TQ1_0 TESTS PASSED!\n");
    } else {
        std::printf("  TESTS FAILED: %d errors!\n", g_fails);
    }
    std::printf("============================================================\n");
    return g_fails ? 1 : 0;
}
