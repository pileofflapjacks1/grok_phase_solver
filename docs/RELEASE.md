# Release process — grok-phase-solver

Current version: **0.13.1** (tag `v0.13.1` when published)

## Build

```bash
cd /path/to/grok_phase_solver
source ~/anaconda3/bin/activate   # macOS if `python` missing
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.13.1*
```

## GitHub tag + release

```bash
git tag -a v0.13.1 -m "v0.13.1: COD Vol-band panel on release tag"
git push origin main
git push origin v0.13.1

gh release create v0.13.1 \
  --title "v0.13.1 — COD Vol-band experimental panel" \
  --notes-file docs/RELEASE_NOTES_v0.13.1.md \
  dist/grok_phase_solver-0.13.1-py3-none-any.whl \
  dist/grok_phase_solver-0.13.1.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/grok_phase_solver-0.13.1*
```

## Pre-release checklist (0.13.1)

- [x] Version aligned → **0.13.1**
- [x] COD Vol-band scoreboard on main
- [x] RELEASE_NOTES_v0.13.1
- [x] Tag + GitHub Release → https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.1
- [ ] PyPI upload (maintainer token — run twine locally)
- [x] No overclaims; mid-band numbers honest

**This release:** https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.1  
**Prior:** v0.13.0 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.0
