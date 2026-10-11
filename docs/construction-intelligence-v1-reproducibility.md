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

The subprocess entry point starts its five-second claim heartbeat before importing the ML stack or
validating a resume. Training and matrix-evaluation dependency initialization have a five-minute
monotonic deadline; their started events are emitted only after initialization succeeds. If it expires,
the worker records `worker_launch_failed`, stops renewing its lease, and cannot begin training; the
dispatcher must fence and terminate the owned process before freeing its GPU slot. Claims and
cancellation are checked again after initialization. This prevents slow cold imports from consuming
the normal sixty-second stale-worker lease without making a hung import an immortal job. These
startup changes do not alter frozen experiment source or relax checkpoint compatibility checks.

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
If a worker exits before its first checkpoint, resume restarts that same immutable configuration from
transition zero, clears the partial progress/artifact pointer, and records
`resume_requested: restart_from_beginning` in the append-only event stream. A run that names a checkpoint
which is missing on disk is rejected rather than silently restarted.

The interactive API binds only to `127.0.0.1`; its Coppelia health probe is also pinned to loopback.
Host and WebSocket-origin checks protect that single-user boundary. GitHub Pages remains read-only.

## Required Python gates

The protected Python check verifies lock drift, dependency integrity, byte compilation, Ruff,
full-source mypy, and the pytest suite with branch measurement enabled. It enforces two independent
coverage gates:

- **Line coverage >=87.85%**, computed from integer `lines-covered / lines-valid` in `coverage.xml` by
  `scripts/check_line_coverage.py`. The canonical floor is
  `tool.construction_intelligence.coverage.line_fail_under` in `pyproject.toml`. The checker compares the
  exact ratio, not the rounded XML `line-rate` or a displayed percentage, and rejects invalid counters,
  zero denominators, and invalid/non-finite floor values.
- **Combined statement/branch coverage >=82.19%**, retained as Coverage.py's separate `fail_under` gate.
  This combined percentage must not be presented as line coverage.

Run both checks locally after generating the report:

```powershell
python -m pytest -q --cov=embodied_skill_composer --cov-report=term-missing --cov-report=xml
python scripts/check_line_coverage.py --coverage-xml coverage.xml
```

CI uploads `coverage.xml` as the `python-coverage` artifact for 90 days, including when tests or either
coverage gate fail and a report exists. The artifact's workflow run identifies the tested commit.

### Current verified line-baseline provenance

The successful PR #25 Linux run
[38111292762](https://github.com/sriv144/embodied-skill-composer/actions/runs/38111292762), at source commit
`bd06db9a0443d4719be8d4aca34db882128d617e`, reports 18,252 executable statements and 2,219 missed
statements: **16,033/18,252 =87.84242822704361% line coverage**. Its complete Python gate passed with
1,222 tests passed, three skipped, full-source mypy over 136 files, and **85.25% combined** coverage.

This is the current successfully measured post-cleanup line baseline. The required line floor is the
higher of 80%, the preserved historical baseline below, and this current measured baseline, rounded
upward to two decimal places: **87.85%**. The baseline source commit, workflow run, and exact integer
counts are recorded alongside the canonical floor in `pyproject.toml`; subsequent measurements must
not lower it. This current verified measurement establishes the floor without claiming to reconstruct
a missing historical Windows measurement.

### Preserved historical line-baseline provenance

The preserved successful Phase-2 PR #20 Linux runs
[29483361158](https://github.com/sriv144/embodied-skill-composer/actions/runs/29483361158) and
[29483359164](https://github.com/sriv144/embodied-skill-composer/actions/runs/29483359164) both report
9,465 executable statements and 1,523 missed statements: **7,942/9,465 =83.90913893291072% line coverage**.
Their source commit is `059985c88405d01673c02585596647d702264685`; its `src`, `scripts`, dependency definitions,
and coverage configuration match the clean post-cleanup commit
`5d30174c1dcaa701cde5b3f00d344b92813f47bd`. This preserved observed post-cleanup baseline rounds upward
to **83.91%**, below the current verified **87.85%** floor.

Those remote runs passed 212 tests and skipped three, with **81.58% combined** coverage. The separately
documented local clean-commit result of 213 passed/two skipped and **82.19% combined** coverage did not
preserve its line counts or XML report. Its exact historical local line baseline cannot be recovered
from that combined percentage and is not claimed to equal either 83.91% or 87.85%. A future isolated run
of the archived commit would be a documented reconstruction under its recorded environment, not the
missing historical measurement. The enforced floor is bound to the current verified remote measurement
above, not an unverifiable historical Windows value.
