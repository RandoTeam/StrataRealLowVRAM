#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#include "strata/kernels/dequant_bf16.hpp"

// Test GDN wqkv dimensions: N = 10240, K = 2560, T = 774
constexpr int64_t N = 10240;
constexpr int64_t K = 2560;
constexpr int64_t T = 774;

int main() {
    printf("==================================================\n");
    printf("Microbenchmark: GDN Proj (N=%lld, K=%lld, T=%lld)\n", (long long)N, (long long)K, (long long)T);
    printf("==================================================\n");

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    // Test with VRAM pressure: allocate background buffers up to 3500 MB
    void* bg_buf = nullptr;
    size_t bg_bytes = 3200ULL * 1024 * 1024;
    cudaError_t bg_err = cudaMalloc(&bg_buf, bg_bytes);
    printf("Background VRAM allocation (3200 MB): %s\n", cudaGetErrorString(bg_err));
    if (bg_err == cudaSuccess) cudaMemset(bg_buf, 0xAA, bg_bytes);

    // Q8_0 weights (type 8): 34 bytes per 32 elements
    size_t w_bytes = (size_t) (N * K / 32) * 34;
    void* d_W = nullptr;
    cudaMalloc(&d_W, w_bytes);
    cudaMemset(d_W, 0x55, w_bytes);

    // Scratch for dequantized weights (FP16): N * K * 2 bytes
    uint16_t* d_scratch = nullptr;
    cudaMalloc((void**)&d_scratch, (size_t) N * K * 2);

    // Input X: T * K * 2 bytes (FP16)
    uint16_t* d_X = nullptr;
    cudaMalloc((void**)&d_X, (size_t) T * K * 2);

    // Output Y: T * N * 4 bytes (FP32)
    float* d_Y = nullptr;
    cudaMalloc((void**)&d_Y, (size_t) T * N * 4);

    // Warmup
    strata::kernels::dequant_f16(8, d_W, 0, N, K, d_scratch, stream);
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K,
                 &alpha, d_scratch, CUDA_R_16F, (int) K,
                 d_X, CUDA_R_16F, (int) K,
                 &beta, d_Y, CUDA_R_32F, (int) N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaStreamSynchronize(stream);

    // 1. Measure dequant_f16 alone (10 iterations)
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0, stream);
    for (int i = 0; i < 10; ++i) {
        strata::kernels::dequant_f16(8, d_W, 0, N, K, d_scratch, stream);
    }
    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);
    float ms_dq = 0;
    cudaEventElapsedTime(&ms_dq, ev0, ev1);
    printf("dequant_f16 (Q8_0 -> FP16, %zu MB): %.3f ms per call\n", (size_t)(N * K * 2 >> 20), ms_dq / 10.0f);

    // 2. Measure cublasGemmEx alone (10 iterations)
    cudaEventRecord(ev0, stream);
    for (int i = 0; i < 10; ++i) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K,
                     &alpha, d_scratch, CUDA_R_16F, (int) K,
                     d_X, CUDA_R_16F, (int) K,
                     &beta, d_Y, CUDA_R_32F, (int) N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);
    float ms_gemm = 0;
    cudaEventElapsedTime(&ms_gemm, ev0, ev1);
    printf("cublasGemmEx (FP16xFP16 -> FP32): %.3f ms per call\n", ms_gemm / 10.0f);

    // 3. Total for 1 projection
    printf("Total 1 proj: %.3f ms (x 252 projections = %.1f ms total)\n",
           (ms_dq + ms_gemm) / 10.0f, ((ms_dq + ms_gemm) / 10.0f) * 252.0f);

    cudaFree(d_W);
    cudaFree(d_scratch);
    cudaFree(d_X);
    cudaFree(d_Y);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}
