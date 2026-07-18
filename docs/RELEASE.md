# Release checklist — grok-phase-solver

Current version: **0.3.0** (tag `v0.3.0`)

## Build (always)

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.3.0*
# Expect: PASSED for sdist + wheel
```

Artifacts:

- `dist/grok_phase_solver-0.3.0.tar.gz`
- `dist/grok_phase_solver-0.3.0-py3-none-any.whl`

## Git tag + GitHub Release

```bash
git tag -a v0.3.0 -m "v0.3.0: Melgalvis synthetics, XL retrain, CIF seeds"
git push origin main
git push origin v0.3.0

gh release create v0.3.0 \
  --title "v0.3.0 — Melgalvis synthetics" \
  --notes-file docs/RELEASE_NOTES_v0.3.0.md \
  dist/grok_phase_solver-0.3.0-py3-none-any.whl \
  dist/grok_phase_solver-0.3.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI upload

Create an API token at https://pypi.org/manage/account/token/  
(scope: project `grok-phase-solver` or entire account).

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD='pypi-...'   # do not commit
python -m twine upload dist/grok_phase_solver-0.3.0*
unset TWINE_PASSWORD
```

Or:

```bash
python -m twine upload dist/grok_phase_solver-0.3.0* \
  -u __token__ -p 'pypi-YOUR_TOKEN'
```

Install after publish:

```bash
python -m pip install -U grok-phase-solver
python -c "import grok_phase_solver; print(grok_phase_solver.__version__)"
```

## What is / is not in the wheel

| Included | Not included |
|----------|----------------|
| Source under `src/grok_phase_solver/` | SHELX binaries (`ShelX/`) |
| Console scripts (`gps-solve`, `gps-gui`, …) | PhAI weights |
| Melgalvis generator code | Large local scoreboard NPZs optional on PyPI (code only) |

## Post-release smoke

```bash
python -m pip install -U grok-phase-solver
python -c "import grok_phase_solver; print(grok_phase_solver.__version__)"  # 0.3.0
gps-solve --help
```
