# Release process — grok-phase-solver

Current version: **0.13.4** (tag `v0.13.4` when published)

## Build

```bash
cd /path/to/grok_phase_solver
source ~/anaconda3/bin/activate   # macOS if `python` missing
python -m pip install -U build twine
rm -rf dist build src/*.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.13.4*
```

## GitHub tag + release

```bash
git tag -a v0.13.4 -m "v0.13.4: --retry-with-peaks second pass"
git push origin main
git push origin v0.13.4

gh release create v0.13.4 \
  --title "v0.13.4 — retry-with-peaks second pass" \
  --notes-file docs/RELEASE_NOTES_v0.13.4.md \
  dist/grok_phase_solver-0.13.4-py3-none-any.whl \
  dist/grok_phase_solver-0.13.4.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/grok_phase_solver-0.13.4*
```

## Pre-release checklist (0.13.4)

- [x] Version aligned → **0.13.4**
- [x] `--retry-with-peaks` on main
- [x] RELEASE_NOTES_v0.13.4
- [x] Tag + GitHub Release → https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.4
- [ ] PyPI upload (maintainer token — run twine locally)
- [x] No overclaims; science still 0.13.1 freeze

**This release:** https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.4  
**Prior:** v0.13.3 at https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.13.3
