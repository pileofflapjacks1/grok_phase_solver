# Release checklist — grok-phase-solver

Current version: **0.2.1** (tag `v0.2.1`)

## Build (always)

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build *.egg-info src/*.egg-info
python -m build
python -m twine check dist/*
# Expect: PASSED for sdist + wheel
```

Artifacts:

- `dist/grok_phase_solver-0.2.1.tar.gz`
- `dist/grok_phase_solver-0.2.1-py3-none-any.whl`

## Git tag

```bash
git tag -a v0.2.1 -m "v0.2.1: GUI, partial-φ UX, experimental scoreboard"
git push origin main
git push origin v0.2.1
```

## PyPI upload

Create an API token at https://pypi.org/manage/account/token/  
(scope: entire account or project `grok-phase-solver`).

```bash
# One-shot upload (token as password; username must be __token__)
python -m twine upload dist/grok_phase_solver-0.2.1* \
  -u __token__ -p "pypi-AgEIcHlwaS5vcmc..."

# Or env vars:
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/grok_phase_solver-0.2.1*
```

Test PyPI first (optional):

```bash
python -m twine upload --repository testpypi dist/grok_phase_solver-0.2.1*
```

Install from PyPI after publish:

```bash
python -m pip install grok-phase-solver
python -m pip install "grok-phase-solver[gui]"   # Streamlit UI
gps-solve --help
gps-gui
```

## What is / is not in the wheel

| Included | Not included |
|----------|----------------|
| Source under `src/grok_phase_solver/` | SHELX binaries (`ShelX/`) |
| Package metadata + console scripts | PhAI weights (`third_party/…`) |
| | Large local scoreboard recompute jobs (JSON/md in repo are fine if packaged; data may be git-only) |

Users clone the GitHub repo for demos, scoreboards, and docs; `pip install` for the library + CLIs.

## Post-release smoke

```bash
python -m pip install -U grok-phase-solver
python -c "import grok_phase_solver; print(grok_phase_solver.__version__)"
gps-solve --help
```
