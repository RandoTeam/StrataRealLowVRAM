# ============================================================
# Strata / llama-server + DeepSeek Harness Unified Launcher
# ============================================================

param(
    [string]$Model = ""
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Strata / llama-server + DeepSeek Harness"

$strataDir = "C:\Users\Ilia V\Documents\antigravity\calm-noether\Strata"

Write-Host ""
Write-Host "  +===========================================================+" -ForegroundColor Cyan
Write-Host "  |   Strata / llama-server + DeepSeek Harness Launcher       |" -ForegroundColor Cyan
Write-Host "  +===========================================================+" -ForegroundColor Cyan
Write-Host ""

if (-not $Model) {
    Write-Host "  Select model to launch:" -ForegroundColor Cyan
    Write-Host "    [1] Qwen 3.8 Flash Next Coder (IQ1_M) - Strata (Fastest Coder, 84.2% SWE-bench)" -ForegroundColor White
    Write-Host "    [2] Qwen 3.8 Flash Next Full (Q2_0)    - Strata (Full Reasoning, 91.9% LCB)" -ForegroundColor White
    Write-Host "    [3] Qwen 3.6 35B A3B Uncensored        - HauhauCS Aggressive via llama-server (25-32 tok/s)" -ForegroundColor White
    Write-Host ""
    $choice = Read-Host "  Enter choice [1-3] (Default: 1)"
    if (-not $choice) { $choice = "1" }
    $Model = $choice
}

$isQwen36 = $false
if ($Model -eq "3" -or $Model -match "35b" -or $Model -match "qwen36" -or $Model -match "uncensored") {
    $isQwen36 = $true
    $modelTitle = "Qwen 3.6 35B A3B Uncensored (HauhauCS Aggressive via llama-server)"
    $serverBat = "$strataDir\start_qwen36_llama.bat"
    $healthUrl = "http://127.0.0.1:8081/health"
    $healthFallbackUrl = "http://127.0.0.1:8081/v1/models"
    $processName = "llama-server"
    $ramNote = "11.7 GB into RAM/VRAM"
    $port = 8081
} elseif ($Model -eq "2" -or $Model -match "q2" -or $Model -match "full") {
    $modelTitle = "Qwen 3.8 Flash Next Full (Q2_0)"
    $serverBat = "$strataDir\run-q2_0.bat"
    $healthUrl = "http://127.0.0.1:8080/health"
    $healthFallbackUrl = "http://127.0.0.1:8080/v1/models"
    $processName = "strata"
    $ramNote = "27.8 GB into RAM/VRAM"
    $port = 8080
} else {
    $modelTitle = "Qwen 3.8 Flash Next Coder (IQ1_M)"
    $serverBat = "$strataDir\run-coder-iq1_m.bat"
    $healthUrl = "http://127.0.0.1:8080/health"
    $healthFallbackUrl = "http://127.0.0.1:8080/v1/models"
    $processName = "strata"
    $ramNote = "23.4 GB into RAM/VRAM"
    $port = 8080
}

function Test-HttpPort($url) {
    try {
        $res = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2 -ErrorAction Stop
        return ($res.StatusCode -eq 200)
    } catch {
        return $false
    }
}

Write-Host "  Selected: $modelTitle" -ForegroundColor Green
Write-Host "  [1/2] Checking model server ($healthUrl)..." -ForegroundColor Yellow
$serverReady = (Test-HttpPort $healthUrl) -or (Test-HttpPort $healthFallbackUrl)

if (-not $serverReady) {
    $runningProc = Get-Process -Name $processName -ErrorAction SilentlyContinue
    if (-not $runningProc) {
        Write-Host "        Compacting RAM and freeing standby cache for Large Pages..." -ForegroundColor Cyan
        & powershell.exe -ExecutionPolicy Bypass -File "$strataDir\tools\optimize_memory.ps1"
        Write-Host "        Starting $modelTitle in a new window..." -ForegroundColor Green
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$serverBat`"" -WorkingDirectory $strataDir
    } else {
        Write-Host "        $processName process is already loading weights..." -ForegroundColor Yellow
    }

    Write-Host "        Waiting for server to finish loading $ramNote..." -ForegroundColor DarkGray
    $elapsed = 0
    while (-not $serverReady -and $elapsed -lt 300) {
        Start-Sleep -Seconds 2
        $elapsed += 2
        $serverReady = (Test-HttpPort $healthUrl) -or (Test-HttpPort $healthFallbackUrl)
        if (-not $serverReady -and ($elapsed % 10 -eq 0)) {
            Write-Host "        ... still loading ($elapsed s elapsed)" -ForegroundColor DarkGray
        }
    }
}

if ($serverReady) {
    Write-Host "        [OK] $modelTitle is ONLINE and READY on port $port!" -ForegroundColor Green
} else {
    Write-Host "        [ERROR] Model server did not respond within 300s. Check the server console window." -ForegroundColor Red
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
