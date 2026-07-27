# Release process — grok-phase-solver

Current version: **0.5.0** (tag `v0.5.0`)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.5.0*
```

## GitHub tag + release

```bash
git tag -a v0.5.0 -m "v0.5.0: diffusion hybrid, SG, predicted seeds, UQ"
git push origin main
git push origin v0.5.0

gh release create v0.5.0 \
  --title "v0.5.0 — diffusion hybrid + SG + predicted seeds" \
  --notes-file docs/RELEASE_NOTES_v0.5.0.md \
  dist/grok_phase_solver-0.5.0-py3-none-any.whl \
  dist/grok_phase_solver-0.5.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
python -m twine upload dist/grok_phase_solver-0.5.0*
```

## Pre-release checklist

- [x] `pytest -q` green (166 passed, 2026-07-27)
- [x] Version aligned: pyproject, `__version__`, CHANGELOG → **0.5.0**
- [x] Tag `v0.5.0` pushed; GitHub Release published with wheel, sdist, paper PDF
- [x] `python -m build` + `twine check` PASSED
- [ ] **PyPI upload** — needs your API token (agent env has no credentials):
  ```bash
  cd /Users/joe/Projects/grok_phase_solver
  # Create token: https://pypi.org/manage/account/token/
  export TWINE_USERNAME=__token__
  export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...   # paste full token
  python -m twine upload dist/grok_phase_solver-0.5.0*
  ```
- [x] No overclaims (diffusion experimental; seed bar; partial-φ)

**Release URL:** https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.5.0
