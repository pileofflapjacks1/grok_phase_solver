# Release process — grok-phase-solver

Current version: **0.13.0** (tag `v0.13.0` when published)

## Build

```bash
cd /path/to/grok_phase_solver
source ~/anaconda3/bin/activate   # macOS if `python` missing
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.13.0*
```

## GitHub tag + release

```bash
git tag -a v0.13.0 -m "v0.13.0: GraphPhaseNet v11 + CrystalX typing + XDXD research"
git push origin main
git push origin v0.13.0

gh release create v0.13.0 \
  --title "v0.13.0 — GraphPhaseNet v11 + CrystalX typing + XDXD research" \
  --notes-file docs/RELEASE_NOTES_v0.13.0.md \
  dist/grok_phase_solver-0.13.0-py3-none-any.whl \
  dist/grok_phase_solver-0.13.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/grok_phase_solver-0.13.0*
```

## Pre-release checklist (0.13.0)

- [x] `pytest -q` green (215 passed)
- [x] Version aligned → **0.13.0**
- [x] Docs: RELEASE_NOTES, graph_phase_net_v11, crystalx_typing, generative_structure
- [x] Scoreboard `strong_prior_v11` (quick; honest — does not clear 30% bar)
- [ ] Tag + GitHub Release
- [ ] PyPI upload (maintainer token)
- [x] No overclaims; generative/xdxd off auto; no GraPhAI redistribution

**Prior:** v0.12.0 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.12.0
