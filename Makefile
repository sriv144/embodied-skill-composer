# Cross-platform developer entrypoints for embodied-skill-composer.
# Mirrors the PowerShell commands in README.md so Linux / macOS
# contributors get parity one-liners. Override PYTHON or
# RUNTIME_PROFILE on the command line, e.g.:
#   make assembly-learned RUNTIME_PROFILE=configs/assembly_profiles/local_gpu.yaml

PYTHON ?= python
RUNTIME_PROFILE ?= configs/assembly_profiles/local_dev.yaml

.PHONY: help install install-rl install-mujoco install-pybullet install-dev \
	test lint format typecheck \
	bench-scripted bench-rl \
	assembly-scripted assembly-learned assembly-benchmark \
	mujoco-scripted mujoco-learned \
	train-assembly-options train-assembly-marl \
	check-gpu clean

help: ## Show this help message.
	@awk 'BEGIN {FS = ":.*?## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?## / { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

## ---- install ----

install: ## Install core Python requirements (CPU regression baseline).
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-rl: install ## Add stable-baselines3 / RL extras on top of core.
	$(PYTHON) -m pip install -r requirements-rl.txt

install-mujoco: install-rl ## Add MuJoCo 3D simulation extras.
	$(PYTHON) -m pip install -r requirements-sim-mujoco.txt

install-pybullet: install ## Add PyBullet extras.
	$(PYTHON) -m pip install -r requirements-pybullet.txt

install-dev: install-rl ## Core + RL + editable install of the package.
	$(PYTHON) -m pip install -e .

## ---- quality ----

test: ## Run the pytest suite (CPU regression path).
	$(PYTHON) -m pytest -q --tb=short

lint: ## Ruff lint on src + tests.
	$(PYTHON) -m ruff check src tests

format: ## Ruff format on src + tests.
	$(PYTHON) -m ruff format src tests

typecheck: ## Mypy the src tree (advisory; never fails the make run).
	- $(PYTHON) -m mypy src

## ---- tabletop / warehouse demos ----

bench-scripted: ## Scripted warehouse benchmark (classical CV perception).
	$(PYTHON) scripts/run_benchmark.py --perception classical_cv --policy scripted

bench-rl: ## RL pickup-policy warehouse benchmark.
	$(PYTHON) scripts/run_benchmark.py --perception classical_cv --policy rl

## ---- collaborative assembly ----

assembly-scripted: ## Evaluate the scripted option oracle.
	$(PYTHON) scripts/eval_assembly_options.py --policy scripted

assembly-learned: ## Evaluate the learned hierarchical policy.
	$(PYTHON) scripts/eval_assembly_options.py --policy learned --runtime-profile $(RUNTIME_PROFILE)

assembly-benchmark: ## Compare scripted / learned / low-level MARL policies.
	$(PYTHON) scripts/benchmark_assembly_policies.py --runtime-profile $(RUNTIME_PROFILE)

train-assembly-options: ## Train the hierarchical team-options policy.
	$(PYTHON) scripts/train_assembly_options.py --runtime-profile $(RUNTIME_PROFILE)

train-assembly-marl: ## Train the retained low-level MARL baseline.
	$(PYTHON) scripts/train_assembly_marl.py --runtime-profile $(RUNTIME_PROFILE)

## ---- MuJoCo 3D backend ----

mujoco-scripted: ## Scripted MuJoCo 3D assembly episode (headless).
	$(PYTHON) scripts/run_mujoco_assembly.py --policy scripted

mujoco-learned: ## Learned MuJoCo 3D assembly episode (GPU profile).
	$(PYTHON) scripts/run_mujoco_assembly.py --policy learned --runtime-profile configs/assembly_profiles/mujoco_local.yaml

## ---- utilities ----

check-gpu: ## Validate torch + CUDA availability against the GPU profile.
	$(PYTHON) scripts/check_gpu_runtime.py --runtime-profile configs/assembly_profiles/local_gpu.yaml

clean: ## Remove pytest / ruff / mypy / bytecode caches.
	rm -rf .pytest_cache .ruff_cache .mypy_cache .pytest_tmp
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
