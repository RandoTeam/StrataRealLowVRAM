#pragma once
// Included only by STRATA_ENABLE_HIP builds. CUDA builds use NVIDIA headers.
#if defined(_WIN32)
// hip_runtime.h pulls in <windows.h> on Windows: keep its min/max macros and the rarely used APIs out of the engine.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
// Do not let HIP's legacy macro corrupt libstdc++ attribute names.
#ifdef __noinline__
#undef __noinline__
#endif
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrIntegrated hipDeviceAttributeIntegrated
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#if HIP_VERSION_MAJOR < 7
#define cudaFreeHost hipHostFree
#else
#define cudaFreeHost hipFreeHost
#endif
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphInstantiateWithFlags hipGraphInstantiateWithFlags
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraph_t hipGraph_t
#define cudaHostAlloc hipHostMalloc
#if HIP_VERSION_MAJOR < 7
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocPortable hipHostMallocPortable
#else
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAllocPortable hipHostAllocPortable
#endif
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(...) (::strata::hip_compat::malloc_host(__VA_ARGS__))
#if defined(_WIN32)
// Windows: hipMemGetInfo counts this process's own allocations only, so the desktop's and other programs' share of
// the card is invisible to it, and VRAM sized by it overfills the card (WDDM then moves memory out to system RAM and
// the GPU slows down).  Its free figure is lowered by what Windows' video memory budget for this process withholds
// (src/core/device.cu; STRATA_WDDM_BUDGET=0: hipMemGetInfo as it is).
namespace strata::hip_compat {
hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes);
}
#define cudaMemGetInfo ::strata::hip_compat::mem_get_info
#else
// Linux: hipMemGetInfo's free figure is the kernel driver's KFD accounting, which on an RX 9070 XT (16 GB, Fedora,
// ROCm 7 wheels) read 924 MiB while amdgpu's own counters had 2,128 MiB free, and a 1,500 MiB hipMalloc beside the
// engine succeeded.  The expert cache, sized and grown by that figure, left more than a GiB of the card unused and
// did not grow back after another program freed VRAM.  The free figure is amdgpu's (mem_info_vram_total less
// mem_info_vram_used, for the card's PCI address) when the files are there (STRATA_KERNEL_VRAM_FREE=0: as it is).
// Not on an integrated GPU: its files count only the BIOS carve-out (a Ryzen 7 9800X3D's Radeon: 512 MiB), while HIP
// gives it system RAM (15,536 MiB free by hipMemGetInfo).
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace strata::hip_compat {
inline bool read_sysfs_u64(const char* path, unsigned long long* v) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    const bool ok = std::fscanf(f, "%llu", v) == 1;
    std::fclose(f);
    return ok;
}
inline hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    const hipError_t e = hipMemGetInfo(free_bytes, total_bytes);
    static const bool off = [] {
        const char* v = std::getenv("STRATA_KERNEL_VRAM_FREE");
        return v && std::strcmp(v, "0") == 0;
    }();
    int dev = 0, integrated = 1;
    char bus[64] = {};
    if (e != hipSuccess || off || hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, dev) != hipSuccess || integrated ||
        hipDeviceGetPCIBusId(bus, (int) sizeof(bus), dev) != hipSuccess)
        return e;
    for (char* c = bus; *c; ++c) *c = (char) std::tolower((unsigned char) *c);
    char path[160];
    unsigned long long total = 0, used = 0;
    std::snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/mem_info_vram_total", bus);
    if (!read_sysfs_u64(path, &total)) return e;
    std::snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/mem_info_vram_used", bus);
    if (!read_sysfs_u64(path, &used) || used > total || total == 0) return e;
    *free_bytes = (size_t) (total - used);
    return e;
}
}  // namespace strata::hip_compat
#define cudaMemGetInfo ::strata::hip_compat::mem_get_info
#endif
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSetDevice hipSetDevice
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

namespace strata::hip_compat {
template <typename T>
inline hipError_t malloc_host(T** pointer, size_t bytes) {
    return hipHostMalloc(reinterpret_cast<void**>(pointer), bytes, hipHostMallocDefault);
}
}  // namespace strata::hip_compat

template <typename Kernel>
inline hipError_t cudaFuncSetAttribute(Kernel kernel, hipFuncAttribute attribute, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                     hipGraphNode_t* error, char* log, size_t size) {
    return hipGraphInstantiate(exec, graph, error, log, size);
}
#define __trap() __builtin_trap()   // the compiled-out sm_80 paths (never selected on AMD)
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, symbol, ...) hipMemcpyFromSymbol(dst, HIP_SYMBOL(symbol), __VA_ARGS__)

#include "intrinsics.hpp"
