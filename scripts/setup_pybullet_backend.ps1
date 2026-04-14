$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$toolsDir = Join-Path $projectRoot ".tools\micromamba"
$micromambaExe = Join-Path $toolsDir "micromamba.exe"
$archivePath = Join-Path $toolsDir "micromamba.tar.bz2"
$envPrefix = Join-Path $projectRoot ".mmenv"

$env:PYTHONNOUSERSITE = "1"

New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null

if (-not (Test-Path $micromambaExe)) {
    Write-Host "Downloading micromamba..."
    Invoke-WebRequest -Uri "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile $archivePath
    tar xf $archivePath -C $toolsDir
    Move-Item -Force (Join-Path $toolsDir "Library\bin\micromamba.exe") $micromambaExe
    if (Test-Path (Join-Path $toolsDir "Library")) {
        Remove-Item -Recurse -Force (Join-Path $toolsDir "Library")
    }
    if (Test-Path (Join-Path $toolsDir "info")) {
        Remove-Item -Recurse -Force (Join-Path $toolsDir "info")
    }
    Remove-Item -Force $archivePath
}

if ((Test-Path (Join-Path $envPrefix "python.exe")) -or (Test-Path (Join-Path $envPrefix "python.exe"))) {
    try {
        & $micromambaExe run -p $envPrefix python -s -c "import pybullet, numpy, pydantic, networkx, yaml" | Out-Null
        Write-Host "Existing local PyBullet environment is ready."
        Write-Host "Run:"
        Write-Host "  .\.tools\micromamba\micromamba.exe run -p .\.mmenv python scripts\run_demo.py --task pick_and_place_red_to_tray --backend pybullet --gui"
        exit 0
    } catch {
        Write-Host "Existing local PyBullet environment needs repair. Rebuilding..."
    }
}

Write-Host "Creating or updating the local PyBullet environment..."
& $micromambaExe create -y -p $envPrefix -c conda-forge python=3.11 pybullet pip

Write-Host "Installing project dependencies into the PyBullet environment..."
& $micromambaExe run -p $envPrefix python -s -m pip install --no-user -r (Join-Path $projectRoot "requirements.txt")

Write-Host ""
Write-Host "PyBullet backend is ready."
Write-Host "Run:"
Write-Host "  .\.tools\micromamba\micromamba.exe run -p .\.mmenv python scripts\run_demo.py --task pick_and_place_red_to_tray --backend pybullet --gui"
