param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$WheelDir = "$env:USERPROFILE\Downloads\pytorch-wheels",
    [string]$TorchVersion = "2.11.0+cu130"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found at '$Python'. Create .venv first."
}

$encodedVersion = $TorchVersion.Replace("+", "%2B")
$wheelName = "torch-$TorchVersion-cp311-cp311-win_amd64.whl"
$wheelUrl = "https://download.pytorch.org/whl/cu130/torch-$encodedVersion-cp311-cp311-win_amd64.whl"
$wheelPath = Join-Path $WheelDir $wheelName

New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

if (-not (Test-Path $wheelPath)) {
    Write-Host "Downloading $wheelUrl"
    Write-Host "This is a large wheel and can take a while."
    Start-BitsTransfer -Source $wheelUrl -Destination $wheelPath
}

Write-Host "Installing $wheelPath"
& $Python -m pip install --force-reinstall --no-deps --no-index $wheelPath

Write-Host "Validating CUDA from the active project environment"
& $Python scripts\check_gpu_runtime.py --runtime-profile configs\assembly_profiles\local_gpu.yaml
