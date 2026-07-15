param(
    [int]$ApiPort = 8008,
    [int]$WebPort = 5173,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$webRoot = Join-Path $projectRoot "workbench"
$logRoot = Join-Path $projectRoot "logs"
New-Item -ItemType Directory -Force $logRoot | Out-Null

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Create it with: py -3.11 -m venv .venv"
}

& $python -c "import fastapi, ortools, cv2, shapely, multipart" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing Construction v2 Python dependencies..."
    & $python -m pip install -r (Join-Path $projectRoot "requirements-construction.txt")
}

if (-not (Test-Path -LiteralPath (Join-Path $webRoot "node_modules\vite"))) {
    Write-Host "Installing web workbench dependencies..."
    & npm.cmd install --prefix $webRoot --no-audit --no-fund
}

function Test-Port([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-Port $ApiPort)) {
    Write-Host "Starting Construction API on port $ApiPort..."
    Start-Process -FilePath $python `
        -ArgumentList "scripts\run_construction_api.py", "--port", $ApiPort `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput (Join-Path $logRoot "construction_api.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "construction_api.stderr.log") `
        -WindowStyle Hidden
}

if (-not (Test-Port $WebPort)) {
    Write-Host "Starting web workbench on port $WebPort..."
    Start-Process -FilePath "npm.cmd" `
        -ArgumentList "run", "dev", "--", "--port", $WebPort `
        -WorkingDirectory $webRoot `
        -RedirectStandardOutput (Join-Path $logRoot "construction_web.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "construction_web.stderr.log") `
        -WindowStyle Hidden
}

$deadline = (Get-Date).AddSeconds(60)
do {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 2
        $web = Invoke-WebRequest "http://127.0.0.1:$WebPort" -UseBasicParsing -TimeoutSec 2
        if ($health.status -eq "ready" -and $web.StatusCode -eq 200) {
            break
        }
    } catch {
        Start-Sleep -Seconds 2
    }
} while ((Get-Date) -lt $deadline)

if ($health.status -ne "ready" -or $web.StatusCode -ne 200) {
    throw "Workbench startup timed out. Inspect logs\construction_api.stderr.log and logs\construction_web.stderr.log."
}

$url = "http://127.0.0.1:$WebPort"
Write-Host ""
Write-Host "Construction workbench ready: $url" -ForegroundColor Green
Write-Host "Design: $($health.design_id) | Modules: $($health.module_count) | Robots: $($health.robot_count)"
Write-Host "Solver: $($health.cp_sat)"
if (-not $NoBrowser) {
    Start-Process $url
}
