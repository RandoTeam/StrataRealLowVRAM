# tools/sync_upstream.ps1 - Synchronize StrataRealLowVRAM with upstream Strata and StrataGP
param(
    [switch]$CheckOnly,
    [string]$UpstreamRemote = "upstream",
    [string]$StrataGpRemote = "stratagp"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "StrataRealLowVRAM Upstream Sync Tool" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Fetch remotes
Write-Host "[1/4] Fetching latest commits from upstream and stratagp..." -ForegroundColor Yellow
git fetch $UpstreamRemote
git fetch $StrataGpRemote

# 2. Inspect incoming commits
$upstreamCommits = git log --oneline HEAD..$UpstreamRemote/main
if ($upstreamCommits) {
    Write-Host "`n[!] New commits found in upstream/main:" -ForegroundColor Green
    $upstreamCommits | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
} else {
    Write-Host "`n[OK] Upstream is up to date with HEAD." -ForegroundColor Green
}

$stratagpCommits = git log --oneline HEAD..$StrataGpRemote/main
if ($stratagpCommits) {
    Write-Host "`n[!] New commits found in stratagp/main:" -ForegroundColor Green
    $stratagpCommits | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
}

if ($CheckOnly) {
    exit 0
}

# 3. Conflict Zone Status
Write-Host "`n[2/4] Inspecting local modified/conflict zones..." -ForegroundColor Yellow
$conflictFiles = @(
    "src/program/generate.cpp",
    "strata-coder-iq1_m.json",
    "README.md",
    "serve/server.py"
)

foreach ($f in $conflictFiles) {
    $diff = git diff $UpstreamRemote/main -- $f
    if ($diff) {
        Write-Host "  * $f differs from upstream (managed fork optimization)" -ForegroundColor Yellow
    } else {
        Write-Host "  * $f matches upstream" -ForegroundColor Green
    }
}

# 4. Patch verification
Write-Host "`n[3/4] Verifying local patches in patches/..." -ForegroundColor Yellow
Get-ChildItem -Path "patches" -Filter "*.patch" | ForEach-Object {
    Write-Host "  * Found patch: $($_.Name)" -ForegroundColor Green
}

Write-Host "`n[4/4] Sync summary complete. To rebase patches after merge, use:" -ForegroundColor Cyan
Write-Host "  git merge $UpstreamRemote/main --no-commit" -ForegroundColor Gray
Write-Host "  git apply --check patches/0001-wddm-expert-cache-4gb.patch" -ForegroundColor Gray
