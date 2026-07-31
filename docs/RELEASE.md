# Release process — grok-phase-solver

Current version: **0.10.0** (tag `v0.10.0` when published)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.10.0*
```

## GitHub tag + release

```bash
git tag -a v0.10.0 -m "v0.10.0: HA curriculum, HDM research, AI-PhaSeed filters"
git push origin main
git push origin v0.10.0

gh release create v0.10.0 \
  --title "v0.10.0 — HA curriculum + Hybrid Difference Map (research)" \
  --notes-file docs/RELEASE_NOTES_v0.10.0.md \
  dist/grok_phase_solver-0.10.0-py3-none-any.whl \
  dist/grok_phase_solver-0.10.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/grok_phase_solver-0.10.0*
```

## Pre-release checklist (0.10.0)

- [x] `pytest -q` green (197 passed)
- [x] Version aligned → **0.10.0**
- [x] Docs: RELEASE_NOTES, HDM + v8 math notes
- [ ] Tag + GitHub Release
- [ ] PyPI upload
- [x] No overclaims; HDM off auto; partial-φ hard path documented

**Prior:** v0.9.0 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.9.0
