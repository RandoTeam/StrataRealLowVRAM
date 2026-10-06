// tests/test_tq1_dot_product.cpp - Phase 1.3: CPU SIMD Fallback & Dispatch Auditor
// Isolated verification test for GGML Type 36 / TQ1_0 x Q8_0 dot product.
// Verifies mathematical equivalence between AVX2 SIMD, scalar fallback, and FP32 reference.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <immintrin.h>

#define QK_K 256
#define QK8_0 32

#if defined(_MSC_VER)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT __restrict__
#endif

// ----------------------------------------------------------------------------
// GGML Block Definitions (from ggml-common.h)
// ----------------------------------------------------------------------------

typedef uint16_t ggml_half;

// TQ1_0: 1.6875 bpw (5 elements per byte in qs, 4 elements per byte in qh)
// Size = 48 + 4 + 2 = 54 bytes for 256 elements.
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // (256 - 16)/5 = 48 bytes
    uint8_t qh[QK_K / 64];                  // 256 / 64 = 4 bytes
    ggml_half d;                            // fp16 scale
} block_tq1_0;
static_assert(sizeof(block_tq1_0) == 54, "wrong tq1_0 block size");

// TQ2_0: 2.0625 bpw (2 bits per element)
// Size = 64 + 2 = 66 bytes for 256 elements.
typedef struct {
    uint8_t qs[QK_K / 4]; // 256 / 4 = 64 bytes
    ggml_half d;          // fp16 scale
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == 66, "wrong tq2_0 block size");

// Q8_0: 32 int8 elements with fp16 scale
// Size = 2 + 32 = 34 bytes for 32 elements.
typedef struct {
    ggml_half d;       // fp16 delta
    int8_t qs[QK8_0];  // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 block size");

// ----------------------------------------------------------------------------
// FP16 <-> FP32 conversion helpers
// ----------------------------------------------------------------------------

static inline float ggml_cpu_fp16_to_fp32(ggml_half h) {
#if defined(__F16C__) || defined(__AVX2__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
#else
    // Software conversion fallback
    uint32_t sign = (h >> 15) & 0x0001;
    uint32_t exp  = (h >> 10) & 0x001f;
    uint32_t mant = h & 0x03ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            f = (sign << 31) | ((exp + (-15 + 127)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + (-15 + 127)) << 23) | (mant << 13);
    }
    float res;
    memcpy(&res, &f, sizeof(res));
    return res;
#endif
}

static inline ggml_half ggml_cpu_fp32_to_fp16(float f) {
#if defined(__F16C__) || defined(__AVX2__)
    return (ggml_half) _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), 0));
#else
    // Basic conversion
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x007fffff;
    if (exp > 15) {
        return (ggml_half)(sign | 0x7c00);
    }
    if (exp < -14) {
        return (ggml_half)sign;
    }
    return (ggml_half)(sign | ((exp + 15) << 10) | (mant >> 13));
#endif
}

// ----------------------------------------------------------------------------
// TQ1_0 Reference Quantization & Dequantization (exact GGML logic)
// ----------------------------------------------------------------------------

static const uint8_t k_pow3[6] = {1, 3, 9, 27, 81, 243};

void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            amax = fmaxf(amax, fabsf(x[j]));
        }

        const float d = amax;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = ggml_cpu_fp32_to_fp16(d);

        // 5 elements per byte, along 32 bytes (160 elements)
        for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = (int) lroundf(x[m + n * 32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += (uint8_t) xi;
                }
                q = (uint8_t) (((uint16_t) q * 256 + (243 - 1)) / 243);
                y[i].qs[j + m] = q;
            }
            x += 5 * 32;
        }

        // along 16 bytes (80 elements)
        for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = (int) lroundf(x[m + n * 16] * id) + 1;
                    q *= 3;
                    q += (uint8_t) xi;
                }
                q = (uint8_t) (((uint16_t) q * 256 + (243 - 1)) / 243);
                y[i].qs[j + m] = q;
            }
            x += 5 * 16;
        }

        // 4 elements per byte in qh (16 elements)
        for (size_t j = 0; j < sizeof(y->qh); ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                int xi = (int) lroundf(x[j + m * sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += (uint8_t) xi;
            }
            q *= 3; // Shift to most significant trit
            q = (uint8_t) (((uint16_t) q * 256 + (243 - 1)) / 243);
            y[i].qh[j] = q;
        }
        x += 4 * sizeof(y->qh);
    }
}

void dequantize_row_tq1_0_ref(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = ggml_cpu_fp16_to_fp32(x[i].d);

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = (uint8_t)(x[i].qs[j + m] * k_pow3[n]);
                    int16_t xi = (int16_t)(((uint16_t) q * 3) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = (uint8_t)(x[i].qs[j + m] * k_pow3[n]);
                    int16_t xi = (int16_t)(((uint16_t) q * 3) >> 8);
                    *y++ = (float)(xi - 1) * d;
                }
            }
        }
        for (size_t n = 0; n < 4; ++n) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = (uint8_t)(x[i].qh[j] * k_pow3[n]);
                int16_t xi = (int16_t)(((uint16_t) q * 3) >> 8);
                *y++ = (float)(xi - 1) * d;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Q8_0 Reference Quantization & Dequantization
// ----------------------------------------------------------------------------

void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int64_t nb = k / QK8_0;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            amax = fmaxf(amax, fabsf(x[i * QK8_0 + j]));
        }
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = ggml_cpu_fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; ++j) {
            int v = (int) lroundf(x[i * QK8_0 + j] * id);
            y[i].qs[j] = (int8_t) fmaxf(-128, fminf(127, v));
        }
    }
}

void dequantize_row_q8_0_ref(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int64_t nb = k / QK8_0;

    for (int64_t i = 0; i < nb; i++) {
        const float d = ggml_cpu_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            *y++ = (float) x[i].qs[j] * d;
        }
    }
}

// ----------------------------------------------------------------------------
// Pure FP32 Golden Reference Dot Product
// ----------------------------------------------------------------------------

float dot_product_fp32_ref(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (double) a[i] * (double) b[i];
    }
    return (float) sum;
}

// ----------------------------------------------------------------------------
// Scalar CPU Fallback: ggml_vec_dot_tq1_0_q8_0_scalar
// ----------------------------------------------------------------------------

void ggml_vec_dot_tq1_0_q8_0_scalar(
    int n,
    float * GGML_RESTRICT s,
    size_t bs,
    const void * GGML_RESTRICT vx,
    size_t bx,
    const void * GGML_RESTRICT vy,
    size_t by,
    int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;

    const block_tq1_0 * GGML_RESTRICT x = (const block_tq1_0 *) vx;
    const block_q8_0  * GGML_RESTRICT y = (const block_q8_0  *) vy;
    const int nb = n / QK_K;

    float total_sum = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const block_q8_0 * GGML_RESTRICT y_block = &y[ib * 8]; // 8 q8_0 blocks per tq1_0 block
        float block_sum = 0.0f;

        // Unpack 256 weights into 8 chunks of 32
        // Chunk 0..4: from x[ib].qs[0..31] for n = 0..4
        for (size_t n_trit = 0; n_trit < 5; ++n_trit) {
            int32_t chunk_int = 0;
            const uint8_t p3 = k_pow3[n_trit];
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = (uint8_t)(x[ib].qs[m] * p3);
                int16_t w = (int16_t)(((uint16_t) q * 3) >> 8) - 1;
                chunk_int += (int32_t) w * (int32_t) y_block[n_trit].qs[m];
            }
            block_sum += (float) chunk_int * ggml_cpu_fp16_to_fp32(y_block[n_trit].d);
        }

        // Chunk 5: elements 160..191 (16 from qs[32..47] n=0, 16 from qs[32..47] n=1)
        {
            int32_t chunk_int = 0;
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q0 = (uint8_t)(x[ib].qs[32 + m] * k_pow3[0]);
                int16_t w0 = (int16_t)(((uint16_t) q0 * 3) >> 8) - 1;
                chunk_int += (int32_t) w0 * (int32_t) y_block[5].qs[m];

                uint8_t q1 = (uint8_t)(x[ib].qs[32 + m] * k_pow3[1]);
                int16_t w1 = (int16_t)(((uint16_t) q1 * 3) >> 8) - 1;
                chunk_int += (int32_t) w1 * (int32_t) y_block[5].qs[16 + m];
            }
            block_sum += (float) chunk_int * ggml_cpu_fp16_to_fp32(y_block[5].d);
        }

        // Chunk 6: elements 192..223 (16 from qs[32..47] n=2, 16 from qs[32..47] n=3)
        {
            int32_t chunk_int = 0;
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q2 = (uint8_t)(x[ib].qs[32 + m] * k_pow3[2]);
                int16_t w2 = (int16_t)(((uint16_t) q2 * 3) >> 8) - 1;
                chunk_int += (int32_t) w2 * (int32_t) y_block[6].qs[m];

                uint8_t q3 = (uint8_t)(x[ib].qs[32 + m] * k_pow3[3]);
                int16_t w3 = (int16_t)(((uint16_t) q3 * 3) >> 8) - 1;
                chunk_int += (int32_t) w3 * (int32_t) y_block[6].qs[16 + m];
            }
            block_sum += (float) chunk_int * ggml_cpu_fp16_to_fp32(y_block[6].d);
        }

        // Chunk 7: elements 224..255 (16 from qs[32..47] n=4, 16 from qh[0..3] n=0..3)
        {
            int32_t chunk_int = 0;
            // First 16 elements from qs[32..47] n=4
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q4 = (uint8_t)(x[ib].qs[32 + m] * k_pow3[4]);
                int16_t w4 = (int16_t)(((uint16_t) q4 * 3) >> 8) - 1;
                chunk_int += (int32_t) w4 * (int32_t) y_block[7].qs[m];
            }
            // Next 16 elements from qh[0..3] across n=0..3
            for (size_t n_qh = 0; n_qh < 4; ++n_qh) {
                for (size_t j = 0; j < 4; ++j) {
                    uint8_t qh_val = (uint8_t)(x[ib].qh[j] * k_pow3[n_qh]);
                    int16_t w = (int16_t)(((uint16_t) qh_val * 3) >> 8) - 1;
                    chunk_int += (int32_t) w * (int32_t) y_block[7].qs[16 + n_qh * 4 + j];
                }
            }
            block_sum += (float) chunk_int * ggml_cpu_fp16_to_fp32(y_block[7].d);
        }

        total_sum += block_sum * ggml_cpu_fp16_to_fp32(x[ib].d);
    }

    *s = total_sum;
}

// ----------------------------------------------------------------------------
// AVX2 Vectorized Implementation: ggml_vec_dot_tq1_0_q8_0_avx2
// ----------------------------------------------------------------------------

#define MM256_SET_M128I(hi, lo) _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)

static inline int32_t hsum_i32_8(__m256i v) {
    __m128i v128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
    v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v128);
}

// Computes dot product of 32 unsigned trits (in {0,1,2}) with 32 signed int8s (in y_qs)
// using _mm256_maddubs_epi16 and subtraction of the sum of y_qs.
static inline int32_t dot_chunk_avx2(__m256i qx_trits, const int8_t * y_qs, __m256i ones_8, __m256i ones_16) {
    const __m256i qy = _mm256_loadu_si256((const __m256i *) y_qs);
    // Unsigned qx * signed qy -> 16-bit signed products
    const __m256i p = _mm256_maddubs_epi16(qx_trits, qy);
    // Sum of qy pairs: 1 * qy
    const __m256i qy_sum16 = _mm256_maddubs_epi16(ones_8, qy);
    // (qx - 1) * qy = qx * qy - qy
    const __m256i diff = _mm256_sub_epi16(p, qy_sum16);
    // Pairwise sum of 16-bit to 32-bit
    const __m256i sum32 = _mm256_madd_epi16(diff, ones_16);
    return hsum_i32_8(sum32);
}

void ggml_vec_dot_tq1_0_q8_0_avx2(
    int n,
    float * GGML_RESTRICT s,
    size_t bs,
    const void * GGML_RESTRICT vx,
    size_t bx,
    const void * GGML_RESTRICT vy,
    size_t by,
    int nrc)
{
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void) bs; (void) bx; (void) by; (void) nrc;

    const block_tq1_0 * GGML_RESTRICT x = (const block_tq1_0 *) vx;
    const block_q8_0  * GGML_RESTRICT y = (const block_q8_0  *) vy;
    const int nb = n / QK_K;

    const __m256i ones_8  = _mm256_set1_epi8(1);
    const __m256i ones_16 = _mm256_set1_epi16(1);

    float total_sum = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        const block_q8_0 * GGML_RESTRICT y_block = &y[ib * 8];
        float block_sum = 0.0f;

        // --- 1. Unpack first 32 bytes of qs (160 trits: chunks 0..4) ---
        __m256i qx0 = _mm256_loadu_si256((const __m256i *) x[ib].qs);
        __m256i qx1 = _mm256_add_epi8(qx0, _mm256_add_epi8(qx0, qx0)); // 1 * 3
        __m256i qx2 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx0, 3), _mm256_set1_epi8(-8)), qx0); // 1 * 9
        __m256i qx3 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx1, 3), _mm256_set1_epi8(-8)), qx1); // 3 * 9
        __m256i qx4 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx2, 3), _mm256_set1_epi8(-8)), qx2); // 9 * 9

        qx0 = _mm256_subs_epu8(qx0, ones_8);
        qx1 = _mm256_subs_epu8(qx1, ones_8);
        qx2 = _mm256_subs_epu8(qx2, ones_8);
        qx3 = _mm256_subs_epu8(qx3, ones_8);
        qx4 = _mm256_subs_epu8(qx4, ones_8);

        const __m256i zero256 = _mm256_setzero_si256();
        qx0 = _mm256_avg_epu8(qx0, _mm256_avg_epu8(qx0, zero256));
        qx1 = _mm256_avg_epu8(qx1, _mm256_avg_epu8(qx1, zero256));
        qx2 = _mm256_avg_epu8(qx2, _mm256_avg_epu8(qx2, zero256));
        qx3 = _mm256_avg_epu8(qx3, _mm256_avg_epu8(qx3, zero256));
        qx4 = _mm256_avg_epu8(qx4, _mm256_avg_epu8(qx4, zero256));

        const __m256i mask_3 = _mm256_set1_epi8(3);
        qx0 = _mm256_and_si256(_mm256_srli_epi16(qx0, 6), mask_3);
        qx1 = _mm256_and_si256(_mm256_srli_epi16(qx1, 6), mask_3);
        qx2 = _mm256_and_si256(_mm256_srli_epi16(qx2, 6), mask_3);
        qx3 = _mm256_and_si256(_mm256_srli_epi16(qx3, 6), mask_3);
        qx4 = _mm256_and_si256(_mm256_srli_epi16(qx4, 6), mask_3);

        // Dot product for Chunks 0..4
        block_sum += (float) dot_chunk_avx2(qx0, y_block[0].qs, ones_8, ones_16) * ggml_cpu_fp16_to_fp32(y_block[0].d);
        block_sum += (float) dot_chunk_avx2(qx1, y_block[1].qs, ones_8, ones_16) * ggml_cpu_fp16_to_fp32(y_block[1].d);
        block_sum += (float) dot_chunk_avx2(qx2, y_block[2].qs, ones_8, ones_16) * ggml_cpu_fp16_to_fp32(y_block[2].d);
        block_sum += (float) dot_chunk_avx2(qx3, y_block[3].qs, ones_8, ones_16) * ggml_cpu_fp16_to_fp32(y_block[3].d);
        block_sum += (float) dot_chunk_avx2(qx4, y_block[4].qs, ones_8, ones_16) * ggml_cpu_fp16_to_fp32(y_block[4].d);

        // --- 2. Unpack remaining 16 bytes of qs (80 trits) and 4 bytes of qh (16 trits) ---
        __m128i qx0_128 = _mm_loadu_si128((const __m128i *) (x[ib].qs + 32));
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));
        __m256i qx5_l = _mm256_cvtepu8_epi16(_mm_set1_epi32((int) qh));

        __m128i qx1_128 = _mm_add_epi8(qx0_128, _mm_add_epi8(qx0_128, qx0_128)); // 1 * 3
        __m128i qx2_128 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx0_128, 3), _mm_set1_epi8(-8)), qx0_128); // 1 * 9
        __m128i qx3_128 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx1_128, 3), _mm_set1_epi8(-8)), qx1_128); // 3 * 9
        __m128i qx4_128 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx2_128, 3), _mm_set1_epi8(-8)), qx2_128); // 9 * 9

        __m256i qx01 = MM256_SET_M128I(qx1_128, qx0_128);
        __m256i qx23 = MM256_SET_M128I(qx3_128, qx2_128);

        // 16-bit multiply for qh
        qx5_l = _mm256_mullo_epi16(qx5_l, _mm256_set_epi16(27, 27, 27, 27, 9, 9, 9, 9, 3, 3, 3, 3, 1, 1, 1, 1));
        qx5_l = _mm256_and_si256(qx5_l, _mm256_set1_epi16(0xFF));
        __m128i qx5_128 = _mm_packus_epi16(_mm256_castsi256_si128(qx5_l), _mm256_extracti128_si256(qx5_l, 1));

        __m256i qx45 = MM256_SET_M128I(qx5_128, qx4_128);

        qx01 = _mm256_subs_epu8(qx01, ones_8);
        qx23 = _mm256_subs_epu8(qx23, ones_8);
        qx45 = _mm256_subs_epu8(qx45, ones_8);

        qx01 = _mm256_avg_epu8(qx01, _mm256_avg_epu8(qx01, zero256));
        qx23 = _mm256_avg_epu8(qx23, _mm256_avg_epu8(qx23, zero256));
        qx45 = _mm256_avg_epu8(qx45, _mm256_avg_epu8(qx45, zero256));

        qx01 = _mm256_and_si256(_mm256_srli_epi16(qx01, 6), mask_3);
        qx23 = _mm256_and_si256(_mm256_srli_epi16(qx23, 6), mask_3);
        qx45 = _mm256_and_si256(_mm256_srli_epi16(qx45, 6), mask_3);

        // Dot product for Chunks 5..7
        block_sum += (float) dot_chunk_avx2(qx01, y_block[5].qs, ones_8, ones_16) * ggml_cpu_fp16_to_fp32(y_block[5].d);
        block_sum += (float) dot_chunk_avx2(qx23, y_block[6].qs, ones_8, ones_16) * ggml_cpu_fp16_to_fp32(y_block[6].d);
        block_sum += (float) dot_chunk_avx2(qx45, y_block[7].qs, ones_8, ones_16) * ggml_cpu_fp16_to_fp32(y_block[7].d);

        total_sum += block_sum * ggml_cpu_fp16_to_fp32(x[ib].d);
    }

    *s = total_sum;
}

// ----------------------------------------------------------------------------
// Test Driver & Verifications
// ----------------------------------------------------------------------------

static int g_fails = 0;
#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        std::printf("[FAIL] Line %d: " msg "\n", __LINE__, ##__VA_ARGS__); \
        ++g_fails; \
    } \
} while (0)

void test_exhaustive_byte_trits() {
    std::printf("Running test_exhaustive_byte_trits (all 243 5-trit combinations)...\n");
    for (int t0 = 0; t0 < 3; ++t0) {
        for (int t1 = 0; t1 < 3; ++t1) {
            for (int t2 = 0; t2 < 3; ++t2) {
                for (int t3 = 0; t3 < 3; ++t3) {
                    for (int t4 = 0; t4 < 3; ++t4) {
                        int expected[5] = {t0, t1, t2, t3, t4};
                        uint8_t q = 0;
                        for (int k = 0; k < 5; ++k) {
                            q = (uint8_t)(q * 3 + expected[k]);
                        }
                        uint8_t byte_val = (uint8_t)(((uint16_t) q * 256 + 242) / 243);

                        // Test scalar extraction
                        for (int k = 0; k < 5; ++k) {
                            uint8_t qk = (uint8_t)(byte_val * k_pow3[k]);
                            int extracted = (int)(((uint16_t) qk * 3) >> 8);
                            CHECK(extracted == expected[k], "Scalar trit mismatch for [%d,%d,%d,%d,%d] at k=%d: got %d",
                                  t0, t1, t2, t3, t4, k, extracted);
                        }
                    }
                }
            }
        }
    }
    std::printf("  Exhaustive trit unpack verified: PASS\n");
}

void test_exact_integer_dot() {
    std::printf("Running test_exact_integer_dot (verifying bit-exact integer math)...\n");
    const int n = QK_K; // 256
    std::vector<block_tq1_0> x(1);
    std::vector<block_q8_0> y(8);

    // Set scale = 1.0 for all blocks
    x[0].d = ggml_cpu_fp32_to_fp16(1.0f);
    for (int k = 0; k < 8; ++k) {
        y[k].d = ggml_cpu_fp32_to_fp16(1.0f);
    }

    // Populate weights with alternating pattern
    for (int i = 0; i < 48; ++i) {
        // Trits: [0, 1, 2, 0, 1] -> w in [-1, 0, +1, -1, 0]
        uint8_t q = (uint8_t)((0*3 + 1)*3 + 2);
        q = (uint8_t)((q*3 + 0)*3 + 1);
        x[0].qs[i] = (uint8_t)(((uint16_t)q * 256 + 242) / 243);
    }
    for (int i = 0; i < 4; ++i) {
        // 4 trits: [2, 1, 0, 2] -> w in [+1, 0, -1, +1]
        uint8_t q = (uint8_t)((2*3 + 1)*3 + 0);
        q = (uint8_t)(q*3 + 2);
        q *= 3;
        x[0].qh[i] = (uint8_t)(((uint16_t)q * 256 + 242) / 243);
    }

    // Populate activations with non-trivial signed values
    for (int k = 0; k < 8; ++k) {
        for (int m = 0; m < 32; ++m) {
            y[k].qs[m] = (int8_t)((m * 7 + k * 13) % 255 - 128);
        }
    }

    float s_scalar = 0.0f;
    float s_avx2   = 0.0f;
    ggml_vec_dot_tq1_0_q8_0_scalar(n, &s_scalar, 0, x.data(), 0, y.data(), 0, 1);
    ggml_vec_dot_tq1_0_q8_0_avx2(n, &s_avx2, 0, x.data(), 0, y.data(), 0, 1);

    CHECK(s_scalar == s_avx2, "AVX2 (%f) != Scalar (%f) on exact integer test", s_avx2, s_scalar);
    std::printf("  Exact integer dot result: %f (diff: %f) PASS\n", s_avx2, fabsf(s_avx2 - s_scalar));
}

void test_synthetic_patterns() {
    std::printf("Running test_synthetic_patterns (all-zero, all-positive, all-negative)...\n");
    const int n = QK_K;

    // Pattern 1: All-zero weights (trits = 1, w = 0)
    {
        std::vector<block_tq1_0> x(1);
        std::vector<block_q8_0> y(8);
        x[0].d = ggml_cpu_fp32_to_fp16(2.5f);
        for (int k = 0; k < 8; ++k) {
            y[k].d = ggml_cpu_fp32_to_fp16(0.5f);
            for (int m = 0; m < 32; ++m) y[k].qs[m] = 42;
        }
        // Trit = 1 everywhere:
        // q = ((((1*3+1)*3+1)*3+1)*3+1) = 121. byte = (121*256+242)/243 = 128
        uint8_t zero_byte = (uint8_t)(((uint16_t) 121 * 256 + 242) / 243);
        memset(x[0].qs, zero_byte, sizeof(x[0].qs));
        uint8_t qh_zero = (uint8_t)((((1*3+1)*3+1)*3+1)*3); // 120
        memset(x[0].qh, (uint8_t)(((uint16_t) qh_zero * 256 + 242) / 243), sizeof(x[0].qh));

        float s_scalar = 0.0f, s_avx2 = 0.0f;
        ggml_vec_dot_tq1_0_q8_0_scalar(n, &s_scalar, 0, x.data(), 0, y.data(), 0, 1);
        ggml_vec_dot_tq1_0_q8_0_avx2(n, &s_avx2, 0, x.data(), 0, y.data(), 0, 1);

        CHECK(s_scalar == 0.0f, "All-zero scalar result is %f, expected 0.0f", s_scalar);
        CHECK(s_avx2 == 0.0f, "All-zero AVX2 result is %f, expected 0.0f", s_avx2);
    }

    // Pattern 2: All +1 weights (trits = 2, w = +1)
    {
        std::vector<block_tq1_0> x(1);
        std::vector<block_q8_0> y(8);
        x[0].d = ggml_cpu_fp32_to_fp16(1.0f);
        for (int k = 0; k < 8; ++k) {
            y[k].d = ggml_cpu_fp32_to_fp16(1.0f);
            for (int m = 0; m < 32; ++m) y[k].qs[m] = 1; // Sum of 256 elements * 1 = 256
        }
        uint8_t plus_byte = (uint8_t)(((uint16_t) 242 * 256 + 242) / 243);
        memset(x[0].qs, plus_byte, sizeof(x[0].qs));
        uint8_t qh_plus = (uint8_t)((((2*3+2)*3+2)*3+2)*3); // 240
        memset(x[0].qh, (uint8_t)(((uint16_t) qh_plus * 256 + 242) / 243), sizeof(x[0].qh));

        float s_scalar = 0.0f, s_avx2 = 0.0f;
        ggml_vec_dot_tq1_0_q8_0_scalar(n, &s_scalar, 0, x.data(), 0, y.data(), 0, 1);
        ggml_vec_dot_tq1_0_q8_0_avx2(n, &s_avx2, 0, x.data(), 0, y.data(), 0, 1);

        CHECK(s_scalar == 256.0f, "All-plus scalar result is %f, expected 256.0f", s_scalar);
        CHECK(s_avx2 == 256.0f, "All-plus AVX2 result is %f, expected 256.0f", s_avx2);
    }

    // Pattern 3: All -1 weights (trits = 0, w = -1)
    {
        std::vector<block_tq1_0> x(1);
        std::vector<block_q8_0> y(8);
        x[0].d = ggml_cpu_fp32_to_fp16(1.0f);
        for (int k = 0; k < 8; ++k) {
            y[k].d = ggml_cpu_fp32_to_fp16(1.0f);
            for (int m = 0; m < 32; ++m) y[k].qs[m] = 1;
        }
        memset(x[0].qs, 0, sizeof(x[0].qs));
        memset(x[0].qh, 0, sizeof(x[0].qh));

        float s_scalar = 0.0f, s_avx2 = 0.0f;
        ggml_vec_dot_tq1_0_q8_0_scalar(n, &s_scalar, 0, x.data(), 0, y.data(), 0, 1);
        ggml_vec_dot_tq1_0_q8_0_avx2(n, &s_avx2, 0, x.data(), 0, y.data(), 0, 1);

        CHECK(s_scalar == -256.0f, "All-minus scalar result is %f, expected -256.0f", s_scalar);
        CHECK(s_avx2 == -256.0f, "All-minus AVX2 result is %f, expected -256.0f", s_avx2);
    }

    std::printf("  Synthetic patterns: PASS\n");
}

void test_random_blocks() {
    std::printf("Running test_random_blocks across varying lengths (256, 512, 1024, 2048, 4096)...\n");
    const std::vector<int> lengths = {256, 512, 1024, 2048, 4096};

    for (int n : lengths) {
        int nb = n / QK_K;
        std::vector<float> orig_x(n);
        std::vector<float> orig_y(n);

        // Generate synthetic floating point weights and activations
        for (int i = 0; i < n; ++i) {
            float r_w = (float)(rand() % 3 - 1); // {-1.0, 0.0, 1.0}
            orig_x[i] = r_w * 0.045f;
            orig_y[i] = ((float)(rand() % 200 - 100) / 100.0f) * 1.5f;
        }

        // Quantize
        std::vector<block_tq1_0> qx(nb);
        std::vector<block_q8_0>  qy(n / QK8_0);
        quantize_row_tq1_0_ref(orig_x.data(), qx.data(), n);
        quantize_row_q8_0_ref(orig_y.data(), qy.data(), n);

        // Dequantize to FP32 for golden reference
        std::vector<float> deq_x(n);
        std::vector<float> deq_y(n);
        dequantize_row_tq1_0_ref(qx.data(), deq_x.data(), n);
        dequantize_row_q8_0_ref(qy.data(), deq_y.data(), n);

        float s_fp32_ref = dot_product_fp32_ref(deq_x.data(), deq_y.data(), n);

        float s_scalar = 0.0f;
        float s_avx2   = 0.0f;
        ggml_vec_dot_tq1_0_q8_0_scalar(n, &s_scalar, 0, qx.data(), 0, qy.data(), 0, 1);
        ggml_vec_dot_tq1_0_q8_0_avx2(n, &s_avx2, 0, qx.data(), 0, qy.data(), 0, 1);

        float diff_avx2_scalar = fabsf(s_avx2 - s_scalar);
        float diff_avx2_ref    = fabsf(s_avx2 - s_fp32_ref);
        float rel_err = fabsf(s_fp32_ref) > 1e-4f ? diff_avx2_ref / fabsf(s_fp32_ref) : diff_avx2_ref;

        CHECK(diff_avx2_scalar < 1e-5f, "AVX2 vs Scalar mismatch at N=%d: avx2=%f scalar=%f diff=%e",
              n, s_avx2, s_scalar, diff_avx2_scalar);
        CHECK(rel_err < 1e-4f, "AVX2 vs FP32 ref error at N=%d: avx2=%f ref=%f rel_err=%e",
              n, s_avx2, s_fp32_ref, rel_err);

        std::printf("  N=%-4d: ref=%10.4f  scalar=%10.4f  avx2=%10.4f  diff(avx2,scalar)=%.2e  rel_err(ref)=%.2e  PASS\n",
                    n, s_fp32_ref, s_scalar, s_avx2, diff_avx2_scalar, rel_err);
    }
}

void test_benchmark_speed() {
    std::printf("Running micro-benchmark (100,000 dot products of N=4096)...\n");
    const int n = 4096;
    const int nb = n / QK_K;
    std::vector<block_tq1_0> qx(nb);
    std::vector<block_q8_0>  qy(n / QK8_0);

    for (int i = 0; i < nb; ++i) {
        qx[i].d = ggml_cpu_fp32_to_fp16(0.05f);
        memset(qx[i].qs, 0xAA, sizeof(qx[i].qs));
        memset(qx[i].qh, 0x55, sizeof(qx[i].qh));
    }
    for (int i = 0; i < n / QK8_0; ++i) {
        qy[i].d = ggml_cpu_fp32_to_fp16(0.02f);
        for (int j = 0; j < QK8_0; ++j) qy[i].qs[j] = (int8_t)(j - 16);
    }

    const int iters = 100000;
    float s = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        ggml_vec_dot_tq1_0_q8_0_scalar(n, &s, 0, qx.data(), 0, qy.data(), 0, 1);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        ggml_vec_dot_tq1_0_q8_0_avx2(n, &s, 0, qx.data(), 0, qy.data(), 0, 1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double ms_avx2 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double gflops_scalar = (2.0 * n * iters) / (ms_scalar * 1e6);
    double gflops_avx2   = (2.0 * n * iters) / (ms_avx2 * 1e6);

    std::printf("  Scalar: %8.2f ms (%.2f GFLOP/s)\n", ms_scalar, gflops_scalar);
    std::printf("  AVX2:   %8.2f ms (%.2f GFLOP/s) -> Speedup: %.2fx\n", ms_avx2, gflops_avx2, ms_scalar / ms_avx2);
}

int main() {
    std::printf("============================================================\n");
    std::printf("  Strata Phase 1.3: CPU SIMD Fallback & Dispatch Auditor    \n");
    std::printf("  Test: GGML Type 36 / TQ1_0 x Q8_0 Dot Product Verification\n");
    std::printf("============================================================\n");

    test_exhaustive_byte_trits();
    test_exact_integer_dot();
    test_synthetic_patterns();
    test_random_blocks();
    test_benchmark_speed();

    std::printf("============================================================\n");
    if (g_fails == 0) {
        std::printf("  ALL TESTS PASSED: Mathematical equivalence verified!\n");
    } else {
        std::printf("  FAILED: %d failures encountered!\n", g_fails);
    }
    std::printf("============================================================\n");
    return g_fails != 0;
}
