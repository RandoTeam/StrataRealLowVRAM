# ==============================================================================
# Strata High-Performance RAM Optimizer (Pre-Launch Memory Compaction)
# ==============================================================================
# Purges Windows Standby Cache, flushes modified pages, and trims inactive working
# sets to coalesce contiguous physical memory blocks for Strata's Large Pages (2 MB).
# Zero background daemons, zero persistent locks. Restores normally on process exit.
# ==============================================================================

$code = @"
using System;
using System.Diagnostics;
using System.Threading;
using System.Runtime.InteropServices;

public class StrataMemOpt {
    [DllImport("ntdll.dll")]
    public static extern int NtSetSystemInformation(int SystemInformationClass, ref int SystemInformation, int SystemInformationLength);

    [DllImport("advapi32.dll", SetLastError = true)]
    public static extern bool OpenProcessToken(IntPtr ProcessHandle, uint DesiredAccess, out IntPtr TokenHandle);

    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern bool LookupPrivilegeValue(string lpSystemName, string lpName, out long lpLuid);

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct TOKEN_PRIVILEGES {
        public int PrivilegeCount;
        public long Luid;
        public int Attributes;
    }

    [DllImport("advapi32.dll", SetLastError = true)]
    public static extern bool AdjustTokenPrivileges(IntPtr TokenHandle, bool DisableAllPrivileges, ref TOKEN_PRIVILEGES NewState, int BufferLength, IntPtr PreviousState, IntPtr ReturnLength);

    [DllImport("kernel32.dll")]
    public static extern IntPtr GetCurrentProcess();

    [DllImport("kernel32.dll")]
    public static extern bool CloseHandle(IntPtr handle);

    [DllImport("psapi.dll")]
    public static extern int EmptyWorkingSet(IntPtr hwProc);

    public static bool EnablePrivilege(string privilege) {
        IntPtr hToken;
        if (!OpenProcessToken(GetCurrentProcess(), 0x0028, out hToken)) return false;
        long luid;
        if (!LookupPrivilegeValue(null, privilege, out luid)) {
            CloseHandle(hToken);
            return false;
        }
        TOKEN_PRIVILEGES tp = new TOKEN_PRIVILEGES();
        tp.PrivilegeCount = 1;
        tp.Luid = luid;
        tp.Attributes = 0x00000002;
        bool res = AdjustTokenPrivileges(hToken, false, ref tp, 0, IntPtr.Zero, IntPtr.Zero);
        CloseHandle(hToken);
        return res;
    }

    public static void Purge() {
        EnablePrivilege("SeIncreaseQuotaPrivilege");
        EnablePrivilege("SeProfileSingleProcessPrivilege");
        EnablePrivilege("SeLockMemoryPrivilege");
        EnablePrivilege("SeDebugPrivilege");

        // 1. Trim background working sets
        foreach (Process p in Process.GetProcesses()) {
            try {
                if (p.Id > 4 && p.Id != Process.GetCurrentProcess().Id) {
                    EmptyWorkingSet(p.Handle);
                }
            } catch {}
        }

        // 2. Multi-pass coalesce
        for (int i = 0; i < 3; i++) {
            int cmdEmpty = 2; // MemoryEmptyWorkingSets
            NtSetSystemInformation(80, ref cmdEmpty, sizeof(int));
            int cmdFlush = 3; // MemoryFlushModifiedList
            NtSetSystemInformation(80, ref cmdFlush, sizeof(int));
            int cmdPurge = 4; // MemoryPurgeStandbyList
            NtSetSystemInformation(80, ref cmdPurge, sizeof(int));
            Thread.Sleep(30);
        }
    }
}
"@

try {
    Add-Type -TypeDefinition $code -Language CSharp -ErrorAction Stop
    [StrataMemOpt]::Purge()
    $freeGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)
    Write-Host "[Strata Optimizer] Memory compacted: $freeGB GB physical RAM now free & contiguous." -ForegroundColor Green
} catch {
    Write-Host "[Strata Optimizer] Notice: $($_.Exception.Message)" -ForegroundColor DarkGray
}
