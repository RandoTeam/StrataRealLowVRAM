// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_4.h>
#include <cstring>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::platform {

#if defined(_WIN32)
namespace {

bool enable_privilege(const char* priv_name) {
    HANDLE token = NULL;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
        return false;
    }
    LUID luid;
    if (!LookupPrivilegeValueA(NULL, priv_name, &luid)) {
        CloseHandle(token);
        return false;
    }
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    BOOL ok = AdjustTokenPrivileges(token, FALSE, &tp, sizeof(tp), NULL, NULL);
    DWORD err = GetLastError();
    CloseHandle(token);
    return ok && (err == ERROR_SUCCESS);
}

void ensure_process_privileges() {
    static const bool init = []() {
        enable_privilege(SE_INC_WORKING_SET_NAME);
        enable_privilege(SE_LOCK_MEMORY_NAME);
        enable_privilege(SE_INCREASE_QUOTA_NAME);
        return true;
    }();
    (void) init;
}

bool expand_working_set_quota(HANDLE proc, SIZE_T needed_bytes, SIZE_T margin, SIZE_T& out_min, SIZE_T& out_max) {
    SIZE_T cur_min = 0, cur_max = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(proc, &cur_min, &cur_max, &flags)) {
        return false;
    }
    SIZE_T target_min = cur_min + needed_bytes + margin;
    SIZE_T target_max = cur_max > target_min + margin ? cur_max : target_min + margin;

    // 1. Try with disabled hard limits (soft limits)
    if (SetProcessWorkingSetSizeEx(proc, target_min, target_max,
                                   QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        out_min = target_min;
        out_max = target_max;
        return true;
    }

    // 2. Try SetProcessWorkingSetSizeEx without flags
    if (SetProcessWorkingSetSizeEx(proc, target_min, target_max, 0)) {
        out_min = target_min;
        out_max = target_max;
        return true;
    }

    // 3. Fallback to basic SetProcessWorkingSetSize
    if (SetProcessWorkingSetSize(proc, target_min, target_max)) {
        out_min = target_min;
        out_max = target_max;
        return true;
    }

    return false;
}

}  // namespace

LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }

    ensure_process_privileges();

    // Headroom guardian: Safeguard physical RAM before committing locked pages.
    const uint64_t avail_ram = available_physical_memory();
    const uint64_t headroom = resident_headroom_bytes();
    if (avail_ram > 0 && avail_ram <= headroom) {
        r.note = "refused: available physical RAM (" +
                 std::to_string(avail_ram >> 20) + " MiB) is at or below required headroom (" +
                 std::to_string(headroom >> 20) + " MiB)";
        return r;
    }

    // If requesting more than fits safely within headroom, clamp request to safeguard system.
    uint64_t lock_limit = bytes;
    if (avail_ram > 0 && avail_ram < headroom + bytes) {
        lock_limit = avail_ram > headroom ? avail_ram - headroom : 0;
        if (lock_limit == 0) {
            r.note = "refused: locking " + std::to_string(bytes >> 20) +
                     " MiB would breach physical RAM headroom of " +
                     std::to_string(headroom >> 20) + " MiB";
            return r;
        }
    }

    HANDLE self = GetCurrentProcess();
    const SIZE_T margin = (SIZE_T) 512 << 20;
    SIZE_T actual_min = 0, actual_max = 0;
    if (!expand_working_set_quota(self, (SIZE_T) lock_limit, margin, actual_min, actual_max)) {
        r.note = "SetProcessWorkingSetSize failed (error " + std::to_string(GetLastError()) + ")";
    }

    const uint64_t chunk = 1ull << 30; // 1 GiB chunks
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < lock_limit; off += chunk) {
        const uint64_t n = lock_limit - off < chunk ? lock_limit - off : chunk;

        if (!VirtualLock(base + off, (SIZE_T) n)) {
            const DWORD err = GetLastError();
            // Handle ERROR_WORKING_SET_QUOTA (1453):
            if (err == ERROR_WORKING_SET_QUOTA) {
                // Dynamically re-expand process working set quota for this chunk and retry
                expand_working_set_quota(self, (SIZE_T) n, margin * 2, actual_min, actual_max);
                if (VirtualLock(base + off, (SIZE_T) n)) {
                    r.locked_bytes = off + n;
                    continue;
                }

                // If 1 GiB chunk still exceeds working set quota, lock in smaller sub-chunks
                bool sub_ok = true;
                const uint64_t sub_chunk = 64ull << 20; // 64 MiB
                for (uint64_t sub_off = 0; sub_off < n; sub_off += sub_chunk) {
                    const uint64_t sn = n - sub_off < sub_chunk ? n - sub_off : sub_chunk;
                    if (!VirtualLock(base + off + sub_off, (SIZE_T) sn)) {
                        if (GetLastError() == ERROR_WORKING_SET_QUOTA) {
                            expand_working_set_quota(self, (SIZE_T) sn, (SIZE_T) 128 << 20, actual_min, actual_max);
                            if (VirtualLock(base + off + sub_off, (SIZE_T) sn)) {
                                r.locked_bytes = off + sub_off + sn;
                                continue;
                            }
                        }
                        sub_ok = false;
                        break;
                    }
                    r.locked_bytes = off + sub_off + sn;
                }
                if (sub_ok) {
                    continue;
                }
            }

            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (r.locked_bytes >> 20)) +
                     " of " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = r.locked_bytes > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (r.locked_bytes >> 20)) +
             " MiB via working-set quota expansion + VirtualLock";
    if (r.locked_bytes < bytes) {
        r.note += " (clamped to safeguard " + std::to_string(headroom >> 20) + " MiB physical RAM headroom)";
    }
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) { why = "dxgi.dll not found"; return false; }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why = "no DXGI adapter has the CUDA device's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage = info.CurrentUsage;
                ok = budget > 0;
                why = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr) a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

uint64_t total_physical_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
}

uint64_t available_physical_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullAvailPhys : 0;
}

uint64_t resident_headroom_bytes() {
    const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB");
    if (v != nullptr && std::atof(v) >= 0.0) {
        return (uint64_t) (std::atof(v) * 1073741824.0);
    }
    return 4ull << 30;
}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

uint64_t available_physical_memory() {
    const long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

uint64_t resident_headroom_bytes() {
    const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB");
    if (v != nullptr && std::atof(v) >= 0.0) {
        return (uint64_t) (std::atof(v) * 1073741824.0);
    }
    return 4ull << 30;
}
#endif

}  // namespace strata::platform
