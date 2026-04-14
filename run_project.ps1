param(
    [ValidateSet("mock", "pybullet")]
    [string]$Backend = "mock",
    [string]$Task = "pick_and_place_red_to_tray",
    [switch]$ListTasks,
    [switch]$Gui,
    [switch]$RebuildBackend
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Ensure-Venv {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating local .venv..."
        uv venv --python 3.11 .venv | Out-Host
    }

    $importCheck = & $venvPython -c "import numpy,pydantic,networkx,yaml" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing project dependencies into .venv..."
        uv pip install --python $venvPython -r requirements.txt | Out-Host
    }

    return $venvPython
}

function Test-PyBulletEnv {
    $envPython = Join-Path $projectRoot ".mmenv\python.exe"
    if (-not (Test-Path $envPython)) {
        return $false
    }

    $probe = '".\.tools\micromamba\micromamba.exe" run -p .\.mmenv python -s -c "import pybullet, numpy, pydantic, networkx, yaml" 1>nul 2>nul'
    cmd /c $probe | Out-Null
    return ($LASTEXITCODE -eq 0)
}

if ($Backend -eq "pybullet") {
    if ($RebuildBackend -or -not (Test-PyBulletEnv)) {
        & powershell -ExecutionPolicy Bypass -File (Join-Path $projectRoot "scripts\setup_pybullet_backend.ps1")
    } else {
        Write-Host "Using existing local PyBullet environment."
    }

    $arguments = @("run", "-p", ".\.mmenv", "python", "scripts\run_demo.py")
    if ($ListTasks) {
        $arguments += "--list"
    } else {
        $arguments += @("--task", $Task, "--backend", "pybullet")
        if ($Gui) {
            $arguments += "--gui"
        }
    }
    & ".\.tools\micromamba\micromamba.exe" @arguments
    exit $LASTEXITCODE
}

$pythonExe = Ensure-Venv
$demoArgs = @("scripts\run_demo.py")
if ($ListTasks) {
    $demoArgs += "--list"
} else {
    $demoArgs += @("--task", $Task, "--backend", "mock")
}
& $pythonExe @demoArgs
exit $LASTEXITCODE
