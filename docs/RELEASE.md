# Release process — grok-phase-solver

Current version: **0.6.0** (tag `v0.6.0` when published)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.6.0*
```

## GitHub tag + release

```bash
git tag -a v0.6.0 -m "v0.6.0: GraphPhaseNet v5, diffusion score, density viewer"
git push origin main
git push origin v0.6.0

gh release create v0.6.0 \
  --title "v0.6.0 — GraphPhaseNet v5 + diffusion score" \
  --notes-file docs/RELEASE_NOTES_v0.6.0.md \
  dist/grok_phase_solver-0.6.0-py3-none-any.whl \
  dist/grok_phase_solver-0.6.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...   # https://pypi.org/manage/account/token/
python -m twine upload dist/grok_phase_solver-0.6.0*
```

## Pre-release checklist

- [ ] `pytest -q` green
- [ ] Version aligned: pyproject, `__version__`, CHANGELOG → **0.6.0**
- [ ] Tag + GitHub Release assets
- [ ] PyPI upload (maintainer token)
- [x] No overclaims (v5 seed bar honest; diffusion experimental)

**Prior release:** https://github.com/pileofflapjacks1/grok_phase_solver/releases/tag/v0.5.0
