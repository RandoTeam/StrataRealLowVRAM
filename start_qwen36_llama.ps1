# ============================================================
# Qwen 3.6 35B A3B Uncensored (HauhauCS Aggressive) Launcher
# Backend: llama-server.exe (CUDA, 65k context, spec drafting)
# ============================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Qwen 3.6 35B A3B llama-server (Port 8081)"

$llamaExe = "C:\Users\Ilia V\llama-server\llama-server.exe"
if (-not (Test-Path $llamaExe)) {
    $found = Get-Command "llama-server.exe" -ErrorAction SilentlyContinue
    if ($found) {
        $llamaExe = $found.Source
    } else {
        Write-Error "llama-server.exe not found at $llamaExe or in PATH"
        exit 1
    }
}

Write-Host ""
Write-Host "  +======================================================+" -ForegroundColor Cyan
Write-Host "  |   Qwen 3.6 35B A3B Uncensored (HauhauCS Aggressive)   |" -ForegroundColor Cyan
Write-Host "  |   llama-server on port 8081 (65k ctx, 25-32 tok/s)   |" -ForegroundColor Cyan
Write-Host "  +======================================================+" -ForegroundColor Cyan
Write-Host ""

$argsList = @(
    "-hf", "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:IQ2_M",
    "-c", "65536",
    "-ngl", "18",
    "-fa", "1",
    "-ctk", "q4_0",
    "-ctv", "q4_0",
    "-t", "6",
    "--spec-default",
    "--port", "8081"
)

Write-Host "  [1/1] Launching llama-server with GPU offloading and quantized KV cache..." -ForegroundColor Green
& $llamaExe @argsList $args
