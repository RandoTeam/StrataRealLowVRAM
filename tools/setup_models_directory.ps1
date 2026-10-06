# tools/setup_models_directory.ps1
# Configures relative models/ directory using NTFS junctions or links without copying large weights.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$TargetDir = "",

    [Parameter(Position = 1)]
    [string[]]$SourceDirs = @(),

    [Parameter()]
    [ValidateSet("Auto", "Junction", "HardLink", "SymLink")]
    [string]$Mode = "Auto",

    [Parameter()]
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if (-not $scriptDir) {
    $scriptDir = (Get-Location).Path
}
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $TargetDir = Join-Path $repoRoot "models"
}

if ($SourceDirs.Count -eq 0) {
    $SourceDirs = @(
        "C:\VietnAi\qwen3.6\models",
        (Join-Path $repoRoot "..\Strata-data\models")
    )
}

$resolvedTarget = [System.IO.Path]::GetFullPath($TargetDir)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Strata Models Directory Configuration Utility" -ForegroundColor Cyan
Write-Host " Target Directory: $resolvedTarget" -ForegroundColor Gray
Write-Host " Mode:             $Mode" -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor Cyan

# Find existing source directories
$validSourceDirs = @()
foreach ($src in $SourceDirs) {
    try {
        if (Test-Path -LiteralPath $src -PathType Container) {
            $fullSrc = [System.IO.Path]::GetFullPath($src)
            $validSourceDirs += $fullSrc
            Write-Host "[+] Found valid source: $fullSrc" -ForegroundColor Green
        } else {
            Write-Host "[-] Source directory not found (skipping): $src" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[-] Source check failed (skipping): $src" -ForegroundColor Yellow
    }
}

if ($validSourceDirs.Count -eq 0) {
    Write-Error "No valid source directories found. Please provide an existing weights directory via -SourceDirs."
    exit 1
}

$primarySource = $validSourceDirs[0]

# Check existing target status
if (Test-Path -LiteralPath $resolvedTarget) {
    $targetItem = Get-Item -LiteralPath $resolvedTarget -Force
    $isJunction = ($targetItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
    if ($isJunction) {
        Write-Host "[*] Target is an existing NTFS Reparse Point / Junction." -ForegroundColor Yellow
        if ($Force) {
            Write-Host "[*] Removing existing junction due to -Force..." -ForegroundColor Yellow
            [System.IO.Directory]::Delete($resolvedTarget)
        } else {
            Write-Host "[+] Preserving active junction. Linked Models in target:" -ForegroundColor Green
            $models = Get-ChildItem -LiteralPath $resolvedTarget -Filter "*.gguf" -File
            foreach ($m in $models) {
                $gb = [math]::Round($m.Length / 1GB, 2)
                Write-Host ("  - {0,-45} ({1,6} GB)" -f $m.Name, $gb) -ForegroundColor White
            }
            exit 0
        }
    }
}

# Determine execution strategy
$useJunction = ($Mode -eq "Junction") -or ($Mode -eq "Auto" -and (-not (Test-Path -LiteralPath $resolvedTarget)))

if ($useJunction) {
    Write-Host "[*] Creating NTFS Directory Junction: $resolvedTarget -> $primarySource" -ForegroundColor Cyan
    New-Item -ItemType Junction -Path $resolvedTarget -Target $primarySource | Out-Null
    Write-Host "[+] Junction created successfully." -ForegroundColor Green
    
    # Verify linked models
    $models = Get-ChildItem -LiteralPath $resolvedTarget -Filter "*.gguf" -File
    Write-Host "`nDiscovered Models in target:" -ForegroundColor Cyan
    foreach ($m in $models) {
        $gb = [math]::Round($m.Length / 1GB, 2)
        Write-Host ("  - {0,-45} ({1,6} GB)" -f $m.Name, $gb) -ForegroundColor White
    }
    exit 0
}

# File-level linking mode (HardLink / SymLink)
if (-not (Test-Path -LiteralPath $resolvedTarget)) {
    Write-Host "[*] Creating target directory: $resolvedTarget" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $resolvedTarget -Force | Out-Null
}

$linkedCount = 0
foreach ($srcDir in $validSourceDirs) {
    # Scan root of source directory for model weights (avoid cache subdirectories)
    $files = Get-ChildItem -LiteralPath $srcDir -Filter "*.gguf" -File
    foreach ($file in $files) {
        $destFile = Join-Path -Path $resolvedTarget -ChildPath $file.Name
        if (Test-Path -LiteralPath $destFile) {
            if ($Force) {
                Remove-Item -LiteralPath $destFile -Force
            } else {
                Write-Host "[-] Already exists, skipping: $($file.Name)" -ForegroundColor Gray
                continue
            }
        }

        $linkSuccess = $false
        # Try HardLink first on same volume (zero privileges required, 0 byte duplication)
        if ($Mode -in @("Auto", "HardLink")) {
            try {
                New-Item -ItemType HardLink -Path $destFile -Target $file.FullName -ErrorAction Stop | Out-Null
                Write-Host ("[+] HardLink: {0,-45} -> {1}" -f $file.Name, $file.FullName) -ForegroundColor Green
                $linkSuccess = $true
            } catch {
                if ($Mode -eq "HardLink") {
                    Write-Error "Failed to create HardLink for $($file.Name): $_"
                }
            }
        }

        # Fallback to SymLink if HardLink not used or failed
        if (-not $linkSuccess -and $Mode -in @("Auto", "SymLink")) {
            try {
                New-Item -ItemType SymbolicLink -Path $destFile -Target $file.FullName -ErrorAction Stop | Out-Null
                Write-Host ("[+] SymLink:  {0,-45} -> {1}" -f $file.Name, $file.FullName) -ForegroundColor Green
                $linkSuccess = $true
            } catch {
                Write-Warning "Could not create SymbolicLink for $($file.Name) (requires developer mode or elevation): $_"
            }
        }

        if ($linkSuccess) {
            $linkedCount++
        }
    }
}

Write-Host "`n[+] Setup complete. Target models directory ready at '$resolvedTarget'." -ForegroundColor Green
