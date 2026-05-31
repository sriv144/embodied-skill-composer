#!/usr/bin/env bash
# Linux / macOS quickstart for embodied-skill-composer.
#
# Mirrors run_project.ps1 (Windows / PowerShell). Creates a local .venv,
# installs core + RL deps, installs the package in editable mode, then runs
# the regression test suite.
#
# Usage:
#   bash scripts/quickstart.sh                # core + RL + tests
#   bash scripts/quickstart.sh --with-mujoco  # also install MuJoCo extras
#   bash scripts/quickstart.sh --skip-tests   # set up env only
set -euo pipefail

WITH_MUJOCO=0
SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --with-mujoco) WITH_MUJOCO=1 ;;
    --skip-tests)  SKIP_TESTS=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d .venv ]; then
  echo "[quickstart] Creating virtualenv at .venv"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[quickstart] Upgrading pip"
python -m pip install --upgrade pip wheel

echo "[quickstart] Installing core requirements"
pip install -r requirements.txt

echo "[quickstart] Installing RL requirements"
pip install -r requirements-rl.txt

if [ "$WITH_MUJOCO" -eq 1 ]; then
  echo "[quickstart] Installing MuJoCo simulation extras"
  pip install -r requirements-sim-mujoco.txt
fi

echo "[quickstart] Installing project in editable mode"
pip install -e .

if [ "$SKIP_TESTS" -eq 1 ]; then
  echo "[quickstart] Skipping tests (per --skip-tests)"
  exit 0
fi

echo "[quickstart] Running pytest"
python -m pytest -q --basetemp .pytest_tmp

echo
echo "[quickstart] Environment is ready. Try:"
echo "  python scripts/run_demo.py --task pick_and_place_red_to_tray"
echo "  python scripts/eval_assembly_options.py --policy scripted"
