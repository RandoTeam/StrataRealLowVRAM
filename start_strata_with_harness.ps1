# ============================================================
# Strata (Qwen3.8 Coder / Q2_0) + DeepSeek Harness Launcher
# ============================================================

param(
    [string]$Model = "coder"
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Strata + DeepSeek Harness"

$strataDir = $PSScriptRoot
if ($Model -match "q2") {
    $modelTitle = "Qwen3.8 Full Q2_0"
    $strataBat = "$strataDir\run-q2_0.bat"
} else {
    $modelTitle = "Qwen3.8 Coder IQ1_M"
    $strataBat = "$strataDir\run-coder-iq1_m.bat"
}

Write-Host ""
Write-Host "  +======================================================+" -ForegroundColor Cyan
Write-Host "  |   Strata ($modelTitle) + DeepSeek Harness       |" -ForegroundColor Cyan
Write-Host "  +======================================================+" -ForegroundColor Cyan
Write-Host ""

function Test-HttpPort($url) {
    try {
        $res = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2 -ErrorAction Stop
        return ($res.StatusCode -eq 200)
    } catch {
        return $false
    }
}

Write-Host "  [1/2] Checking Strata server (http://127.0.0.1:8080/health)..." -ForegroundColor Yellow
$strataReady = Test-HttpPort "http://127.0.0.1:8080/health"

if (-not $strataReady) {
    $runningStrata = Get-Process -Name "strata" -ErrorAction SilentlyContinue
    if (-not $runningStrata) {
        Write-Host "        Compacting RAM and freeing standby cache for 2 MB Large Pages..." -ForegroundColor Cyan
        & powershell.exe -ExecutionPolicy Bypass -File "$strataDir\tools\optimize_memory.ps1"
        Write-Host "        Starting Strata server ($modelTitle) in a new window..." -ForegroundColor Green
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$strataBat`"" -WorkingDirectory $strataDir
    } else {
        Write-Host "        Strata process is already loading weights..." -ForegroundColor Yellow
    }

    Write-Host "        Waiting for Strata to finish loading weights into RAM/VRAM..." -ForegroundColor DarkGray
    $elapsed = 0
    while (-not $strataReady -and $elapsed -lt 300) {
        Start-Sleep -Seconds 2
        $elapsed += 2
        $strataReady = Test-HttpPort "http://127.0.0.1:8080/health"
        if (-not $strataReady -and ($elapsed % 10 -eq 0)) {
            Write-Host "        ... still loading ($elapsed s elapsed)" -ForegroundColor DarkGray
        }
    }
}

if ($strataReady) {
    Write-Host "        [OK] Strata is ONLINE and READY on port 8080!" -ForegroundColor Green
} else {
    Write-Host "        [ERROR] Strata did not respond within 300s. Check the Strata window." -ForegroundColor Red
    Write-Host "        Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
    exit 1
}

Write-Host ""
Write-Host "  [2/2] Starting DeepSeek Harness Web UI (http://127.0.0.1:3080)..." -ForegroundColor Yellow

$dshReady = Test-HttpPort "http://127.0.0.1:3080"
if ($dshReady) {
    Write-Host "        [OK] DeepSeek Harness is already running! Opening browser..." -ForegroundColor Green
    Start-Process "http://127.0.0.1:3080"
    Start-Sleep -Seconds 2
    exit 0
}

Set-Location "$env:USERPROFILE"
$dshCmd = "$env:APPDATA\npm\dsh.cmd"
if (Test-Path $dshCmd) {
    & $dshCmd web
} else {
    & npx.cmd @deepseek-ai/dsh web
}
