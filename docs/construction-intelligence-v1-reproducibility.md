# Construction Intelligence v1 Reproducibility

## Canonical environments

`pyproject.toml` is the dependency source of truth. The `requirements*.txt` files remain as compatibility
entry points and install the corresponding extras. CI resolves the complete Python 3.11/Linux environment
from `constraints/ci-py311-linux.txt` in strict hash mode. Regenerate that lock from the repository root with:

```powershell
uv pip compile pyproject.toml --extra construction --extra construction-rl --extra dev --extra rl --extra sim-coppelia --extra sim-mujoco --python-platform linux --python-version 3.11 --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --torch-backend cpu --emit-index-url --generate-hashes --no-header --output-file constraints\ci-py311-linux.txt
```

A complete editable CPU research environment on Windows is:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[construction,construction-rl,dev,rl,sim-coppelia,sim-mujoco]"
.\.venv\Scripts\python.exe -m pip check
```

The corresponding Linux setup is:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[construction,construction-rl,dev,rl,sim-coppelia,sim-mujoco]"
.venv/bin/python -m pip check
```

For the local RTX 4060, install `requirements-rl-cuda-cu130.txt` after the base extras. Every learned-policy
manifest records the source commit, dirty-tree flag and digest, canonical configuration digest,
Python/platform details, package versions, Torch CUDA build, CUDA availability, and GPU name. Research
profile launches require a clean source tree.

## Durable local training

Training launched by the REST API or `scripts/train_construction_intelligence.py` enters a SQLite-backed
single-GPU queue. The public run states are `queued`, `running`, `cancel_requested`, `interrupted`,
`resuming`, `completed`, `failed`, and `cancelled`. Inputs, attempts, PID, heartbeat, progress, source
commit, configuration digest, latest checkpoint, and append-only events survive API restarts. Events are
also mirrored to one JSONL file per run.

```powershell
# Blocking: streams persisted events until the job is quiescent.
.\.venv\Scripts\python.exe scripts\train_construction_intelligence.py --profile unit --yes

# Queue and return after the worker starts.
.\.venv\Scripts\python.exe scripts\train_construction_intelligence.py --profile research --yes --detach

# Inspect or control a run from another terminal.
.\.venv\Scripts\python.exe scripts\manage_construction_runs.py list
.\.venv\Scripts\python.exe scripts\manage_construction_runs.py status RUN_ID
.\.venv\Scripts\python.exe scripts\manage_construction_runs.py events RUN_ID
.\.venv\Scripts\python.exe scripts\manage_construction_runs.py cancel RUN_ID
.\.venv\Scripts\python.exe scripts\manage_construction_runs.py resume RUN_ID
```

The worker saves `checkpoints/latest.pt` after every behavior-cloning epoch and PPO update. Configured
fraction snapshots are immutable named checkpoints. A v2 resumable checkpoint contains actor, critic,
PPO and behavior-cloning optimizer states, transition/update/episode counters, learning-curve rows,
Python/NumPy/Torch CPU/CUDA RNG states, the full configuration, configuration and design digests,
environment schema, source commit and tree digest, and checkpoint lineage.

Resume is deliberately strict. A schema, algorithm, environment, configuration, design, source commit,
or source-tree mismatch is rejected before weights or optimizer state are applied. A stale worker is marked
`interrupted`; a live orphan with a fresh heartbeat retains its claim, preventing a second GPU job.

The interactive API binds only to `127.0.0.1`; its Coppelia health probe is also pinned to loopback.
Host and WebSocket-origin checks protect that single-user boundary. GitHub Pages remains read-only.

## Required Python gates

The protected Python check verifies lock drift, dependency integrity, byte compilation, Ruff,
full-source mypy, and the pytest suite with branch measurement enabled. Coverage.py reports the combined
statement/branch total; the release floor is 80%. The measured pre-test-hardening total was 75.56%, so
the floor was not lowered to match the old baseline.
