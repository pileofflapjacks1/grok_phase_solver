# Release process — grok-phase-solver

Current version: **0.12.0** (tag `v0.12.0` when published)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.12.0*
```

## GitHub tag + release

```bash
git tag -a v0.12.0 -m "v0.12.0: GraphPhaseNet v10 + generative structure research"
git push origin main
git push origin v0.12.0

gh release create v0.12.0 \
  --title "v0.12.0 — GraphPhaseNet v10 + generative structure (research)" \
  --notes-file docs/RELEASE_NOTES_v0.12.0.md \
  dist/grok_phase_solver-0.12.0-py3-none-any.whl \
  dist/grok_phase_solver-0.12.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/grok_phase_solver-0.12.0*
```

## Pre-release checklist (0.12.0)

- [x] `pytest -q` green (210 passed)
- [x] Version aligned → **0.12.0**
- [x] Docs: RELEASE_NOTES_v0.12.0, graph_phase_net_v10, generative_structure
- [x] Scoreboard `strong_prior_v10` committed
- [x] Tag + GitHub Release → https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.12.0
- [ ] PyPI upload (maintainer token)
- [x] No overclaims; generative_structure off auto; partial-φ hard path documented

**This release:** https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.12.0  
**Prior:** v0.11.0 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.11.0
