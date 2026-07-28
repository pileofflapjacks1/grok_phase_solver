# Release process — grok-phase-solver

Current version: **0.7.0** (tag `v0.7.0` when published)

## Build

```bash
cd /path/to/grok_phase_solver
python -m pip install -U build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/grok_phase_solver-0.7.0*
```

## GitHub tag + release

```bash
git tag -a v0.7.0 -m "v0.7.0: Melgalvis curriculum, GraPhAI edges, hard-path UX"
git push origin main
git push origin v0.7.0

gh release create v0.7.0 \
  --title "v0.7.0 — Melgalvis curriculum + GraPhAI edges" \
  --notes-file docs/RELEASE_NOTES_v0.7.0.md \
  dist/grok_phase_solver-0.7.0-py3-none-any.whl \
  dist/grok_phase_solver-0.7.0.tar.gz \
  docs/paper/arxiv_draft.pdf
```

## PyPI

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...   # https://pypi.org/manage/account/token/
python -m twine upload dist/grok_phase_solver-0.7.0*
```

## Pre-release checklist

- [ ] `pytest -q` green
- [ ] Version aligned: pyproject, `__version__`, CHANGELOG → **0.7.0**
- [ ] Tag + GitHub Release assets
- [ ] PyPI upload (maintainer token; can wait)
- [x] No overclaims (seed bar honest; partial-φ hard path)

**Prior:** v0.6.0 at https://github.com/pileofflapjacks1/grok_phase_solver/releases
