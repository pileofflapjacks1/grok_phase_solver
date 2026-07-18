# Changelog

## 0.2.1 — 2026-07

### Ship / packaging
- Version bump for first public tag **v0.2.1**
- `__version__` aligned with `pyproject.toml`
- Release notes: `docs/RELEASE.md` (build, tag, PyPI upload)
- **Paper pack:** full manuscript draft (`docs/arxiv_draft.md`), Figs. 1–4
  (`scripts/plot_paper_figures.py` → `docs/figures/paper_fig*`), hub `docs/paper/`
- **PDF:** `docs/paper/arxiv_draft.pdf` via pandoc + tectonic
  (`scripts/build_paper_pdf.py`); author line Grok (xAI)
- **BibTeX:** `docs/paper/references.bib`
- **GitHub Release** v0.2.1 with wheel, sdist, and paper PDF assets
- README: PyPI-first install path; release notes page

### Included since 0.2.0 (product freeze for tag)
- Streamlit **gps-gui** + polish (wizard, CELL parse, peaks retry, SHELXL handoff)
- Lane B partial-φ seed importers + `gps-make-seed`
- Lane C experimental COD scoreboard + `arxiv_draft`
- GraphPhaseNet v4 XL prior weights (honest ~21% seed bar)

## 0.2.0 — 2026-07

### Product / pipeline
- `auto` prefers **ensemble** on easy/high-res; hard uses graph/hard-P1 priors or CF
- **Partial-φ hard path**: `partial_phaseed` + multiple seed sources; demo in `examples/partial_seed_demo/`
- **Lane B seed UX:** `--phase-seed-res`, `--seed-peaks-csv`, `--seed-atoms-csv`,
  isomorphous `--native-hkl`/`--derivative-hkl`, `--patterson-ha`;
  `gps-make-seed` CLI; seed-quality section in `report.md`
- **GUI:** Streamlit app (`gps-gui`, extras `.[gui]`) — upload HKL/INS/seeds,
  run same pipeline, download `trial.res` / zip
- **GUI polish:** scenario wizard, CELL-line parser, quality hints, peaks-as-seed
  retry, SHELXL handoff tab, solver log capture, clearer errors
- **SHELXS / SHELXE** external runners (`ShelX/` gitignored); `shelxs+shelxe` method
- USER_GUIDE decision tree; SHELXL refinement instructions in `report.md`
- `trial.res` export for Olex2/SHELXL

### Science
- Free FOM v2.1, failure taxonomy A/B/C, AI-PhaSeed
- GraphPhaseNet strong prior (v3): Wilson match + strong-seed metrics/loss
- **Lane A v4 XL:** residual MP + Adam + d_in=10 + `--scale-xl` (1200 structs);
  fine-tune `--continue-from` / `--hard-only` / `--seed-focus`
- Oracle partial-φ curves: ≥30% strong φ within 20° → hard strict solves
- Wilson domain-gap closing (`wilson_match.py`)
- SHELXS head-to-head scoreboard vs CF/ensemble/priors

### Review pack (Lane C)
- Expanded **experimental scoreboard**: COD 2016452/2100301 Fobs + Fcalc,
  2017775, oracle partial-φ rows, optional SHELXS / strong prior
- Working draft `docs/arxiv_draft.md`; FOR_REVIEWERS claim **C9**
- COD download list includes 2016452 + HKL flags for 2100301

### Honest limits
- Hard ab initio still ~0% strict success under full priors
- **Seed bar ~21% ≤20° plateaus under 5× scale** (v3 → v4 XL); not cleared to 30%
- Not a general protein ab initio solver
- SHELXD binary not available in default `ShelX/` (SHELXS present when installed)

## 0.1.0 — earlier

- Phase-1 baseline: CF/HIO, COD I/O, metrics, gps-solve core
