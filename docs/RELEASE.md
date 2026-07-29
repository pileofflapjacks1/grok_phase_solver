# Release process — grok-phase-solver

Current version: **0.8.0** (tag `v0.8.0` when published)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.8.0*
```

## GitHub tag + release

```bash
git tag -a v0.8.0 -m "v0.8.0: GraphPhaseNet v6, seed classifier, hard-path polish"
git push origin main
git push origin v0.8.0

gh release create v0.8.0 \
  --title "v0.8.0 — GraphPhaseNet v6 + seed quality classifier" \
  --notes-file docs/RELEASE_NOTES_v0.8.0.md \
  dist/grok_phase_solver-0.8.0-py3-none-any.whl \
  dist/grok_phase_solver-0.8.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...   # https://pypi.org/manage/account/token/
python -m twine upload dist/grok_phase_solver-0.8.0*
```

## Pre-release checklist (0.8.0)

- [x] `pytest -q` green (187 passed)
- [x] Version aligned: pyproject, `__version__`, CHANGELOG → **0.8.0**
- [x] Scoreboards: `strong_prior_v6`, `seed_quality_rf`, COD hard-path
- [x] Docs: RELEASE_NOTES_v0.8.0, graph_phase_net_v6, TODO/README
- [x] Tag + GitHub Release assets
- [ ] PyPI upload (maintainer token; can wait)
- [x] No overclaims (seed bar honest; partial-φ hard path)

**Prior:** v0.7.0 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.7.0
