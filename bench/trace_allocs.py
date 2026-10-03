import ctypes
import sys

cudart = ctypes.CDLL(r"C:\Python312\Lib\site-packages\nvidia\cu13\bin\x86_64\cudart64_13.dll")

def check_mem(label):
    free_b = ctypes.c_size_t()
    total_b = ctypes.c_size_t()
    ret = cudart.cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b))
    free_mib = free_b.value / (1024 * 1024)
    total_mib = total_b.value / (1024 * 1024)
    print(f"[{label}] ret={ret} | Free: {free_mib:.1f} MiB / {total_mib:.1f} MiB (Used: {total_mib - free_mib:.1f} MiB)")

def alloc(size_bytes, label):
    ptr = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size_bytes))
    print(f"Allocating {label} ({size_bytes / 1024**2:.1f} MiB)... ret={ret}")
    check_mem(f"After {label}")
    return ptr

check_mem("Initial Idle")

# 1. WeightTable arena (pool_bytes)
p1 = alloc(1406 * 1024 * 1024, "WeightTable Pool Arena")

# 2. NativeDense (2018.88 MiB)
# In strata, NativeDense does ~300 allocations. Let's do it as chunks or one big chunk.
p2 = alloc(int(2018.88 * 1024 * 1024), "NativeDense Projections")

# 3. MTP Draft layer (790 MiB)
p3 = alloc(790 * 1024 * 1024, "MTP Draft Layer")

# 4. NativeHead (497.3 MiB)
p4 = alloc(521472000, "Native Q5_K Head")

# 5. Expert Cache 426 slots (840 MiB)
p5 = alloc(int(426 * 2662400), "Expert Cache 426 slots")

print("\nDone testing allocations.")
