# Release process — grok-phase-solver

Current version: **0.9.0** (tag `v0.9.0` when published)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.9.0*
```

## GitHub tag + release

```bash
git tag -a v0.9.0 -m "v0.9.0: GraphPhaseNet v7, Carrozzini bins, packing curriculum"
git push origin main
git push origin v0.9.0

gh release create v0.9.0 \
  --title "v0.9.0 — GraphPhaseNet v7 + Carrozzini bins" \
  --notes-file docs/RELEASE_NOTES_v0.9.0.md \
  dist/grok_phase_solver-0.9.0-py3-none-any.whl \
  dist/grok_phase_solver-0.9.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...   # https://pypi.org/manage/account/token/
python -m twine upload dist/grok_phase_solver-0.9.0*
```

## Pre-release checklist (0.9.0)

- [x] `pytest -q` green (193 passed)
- [x] Version aligned: pyproject, `__version__`, CHANGELOG → **0.9.0**
- [x] Scoreboards: `strong_prior_v7` (pilot), docs
- [x] Docs: RELEASE_NOTES_v0.9.0, graph_phase_net_v7, graphai_external
- [ ] Tag + GitHub Release assets
- [ ] PyPI upload (maintainer token; can wait)
- [x] No overclaims (seed bar honest; partial-φ hard path)

**Prior:** v0.8.0 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.8.0
