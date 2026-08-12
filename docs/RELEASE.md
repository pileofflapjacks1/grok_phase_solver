# Release process — grok-phase-solver

Current version: **0.13.2** (tag `v0.13.2` when published)

## Build

```bash
cd /path/to/grok_phase_solver
source ~/anaconda3/bin/activate   # macOS if `python` missing
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.13.2*
```

## GitHub tag + release

```bash
git tag -a v0.13.2 -m "v0.13.2: Vol-band next-action chooser"
git push origin main
git push origin v0.13.2

gh release create v0.13.2 \
  --title "v0.13.2 — Vol-band next-action chooser" \
  --notes-file docs/RELEASE_NOTES_v0.13.2.md \
  dist/grok_phase_solver-0.13.2-py3-none-any.whl \
  dist/grok_phase_solver-0.13.2.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/grok_phase_solver-0.13.2*
```

## Pre-release checklist (0.13.2)

- [x] Version aligned → **0.13.2**
- [x] Next-action chooser on main
- [x] RELEASE_NOTES_v0.13.2
- [x] Tag + GitHub Release → https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.2
- [ ] PyPI upload (maintainer token — run twine locally)
- [x] No overclaims; science still 0.13.1 freeze

**This release:** https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.2  
**Prior:** v0.13.1 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.1
