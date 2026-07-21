# Release process — grok-phase-solver

Current version: **0.4.0** (tag `v0.4.0`)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.4.0*
```

Artifacts:

- `dist/grok_phase_solver-0.4.0.tar.gz`
- `dist/grok_phase_solver-0.4.0-py3-none-any.whl`

## GitHub tag + release

```bash
git tag -a v0.4.0 -m "v0.4.0: Carrozzini AI-PhaSeed hybrid, seed quality, DM+AI"
git push origin main
git push origin v0.4.0

gh release create v0.4.0 \
  --title "v0.4.0 — Carrozzini AI-PhaSeed alignment" \
  --notes-file docs/RELEASE_NOTES_v0.4.0.md \
  dist/grok_phase_solver-0.4.0-py3-none-any.whl \
  dist/grok_phase_solver-0.4.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI upload (user token required)

```bash
python -m twine upload dist/grok_phase_solver-0.4.0*
```

## Pre-release checklist

- [ ] `pytest -q` green
- [ ] Version aligned: `pyproject.toml`, `__init__.__version__`, CHANGELOG
- [ ] RELEASE_NOTES_v0.4.0.md accurate
- [ ] No overclaims (hard ab initio limits, partial-φ bar)
- [ ] Optional: re-run `scripts/run_ai_phaseed_extended_benchmark.py`
