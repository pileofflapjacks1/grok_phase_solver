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

- [ ] `pytest -q` green
- [ ] Version aligned: pyproject, `__version__`, CHANGELOG
- [ ] No overclaims (diffusion experimental; seed bar; partial-φ)
- [ ] Optional: smoke `gps-solve --help` and diffusion method on demo
