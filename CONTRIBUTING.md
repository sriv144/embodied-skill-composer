# Contributing

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-rl.txt
python -m pytest -q --basetemp .pytest_tmp
```

## Main Checks

Use these commands before committing:

```powershell
python -m pytest -q --basetemp .pytest_tmp
python -m compileall src scripts tests
python scripts\benchmark_assembly_policies.py --runtime-profile configs\assembly_profiles\local_dev.yaml --episodes 3
```

Expected benchmark summary:

- `scripted_options`: success `1.000`
- `learned_options`: success `1.000`
- `low_level_learned`: success `0.000`

## Git / Publishing

This repository is intended to live in a **private GitHub repo** during active development.

If `gh auth status` fails, refresh GitHub CLI authentication first:

```powershell
gh auth login -h github.com
```

Once authenticated, create and push the private repo:

```powershell
gh repo create embodied-skill-composer --private --source=. --remote=origin --push
```
