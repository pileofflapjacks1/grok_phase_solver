# Release process — grok-phase-solver

Current version: **0.11.0** (tag `v0.11.0` when published)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.11.0*
```

## GitHub tag + release

```bash
git tag -a v0.11.0 -m "v0.11.0: Melgalvis large-cell, GraphPhaseNet v9, AI-PhaSeed harden"
git push origin main
git push origin v0.11.0

gh release create v0.11.0 \
  --title "v0.11.0 — Melgalvis large-cell + GraphPhaseNet v9" \
  --notes-file docs/RELEASE_NOTES_v0.11.0.md \
  dist/grok_phase_solver-0.11.0-py3-none-any.whl \
  dist/grok_phase_solver-0.11.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/grok_phase_solver-0.11.0*
```

## Pre-release checklist (0.11.0)

- [x] `pytest -q` green (205 passed)
- [x] Version aligned → **0.11.0**
- [x] Docs: RELEASE_NOTES_v0.11.0, synthetic_melgalvis, graph_phase_net_v9
- [x] Scoreboard `strong_prior_v9` (quick; honest — does not clear 30% bar)
- [ ] Tag + GitHub Release
- [ ] PyPI upload
- [x] No overclaims; partial-φ hard path documented; no external weight redistribution

**Prior:** v0.10.0 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.10.0
