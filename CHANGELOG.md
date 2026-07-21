# Changelog

## 0.4.0 — 2026-07

### What’s new in v0.4.0

**Carrozzini 2025 AI-PhaSeed alignment** — hybrid tooling and diagnostics without
breaking pre-0.4 APIs (defaults keep `dm_ai_weight=0`).

- **DM+AI modified tangent** (`dm_ai_weight` / CLI `--ai-dm-hybrid`): AI phases
  enter κ-weighted tangent as a priori info with reliability weights
  (`direct_methods.dm_ai_hybrid_refine`, `phase_extend` schedule).
- **Seed-quality Class 0/1 predictor** (`metrics/seed_quality.py`): features
  max W, N_asym, Vol, seed fraction, free-FOM proxies; heuristic P(success) +
  optional sklearn RF extra (`pip install grok-phase-solver[seed-quality]`).
- **Low-res / large-Vol EDM path** (`low_res_path` / `--low-res-path`): longer
  seed anneal, solvent flatten, more frequent hybrid steps.
- **CLI / GUI**: `--ai-dm-hybrid`, `--dm-ai-weight`, `--low-res-path`,
  `--prior-weight`, `--seed-quality-filter`; GUI seed-quality panel.
- **Docs**: expanded `docs/math/ai_phaseed.md` (eqs, statistical toolkit,
  P2₁/c + generalization notes); citations in FOR_REVIEWERS / arxiv draft /
  `references.bib`.
- **Benchmark**: `scripts/run_ai_phaseed_extended_benchmark.py` +
  `data/processed/ai_phaseed_extended_benchmark.*` (stratified subset harness;
  not a 1505-structure claim).
- **Honest limits unchanged:** hard ab initio seed bar still ~21–22% ≤20°;
  partial-φ remains the hard-data path; Class predictor is operational, not
  the published RF on 1505 COD entries.

### Science
- Carrozzini et al. (2025) J. Appl. Cryst. 58, 1859–1869 DOI 10.1107/S1600576725008271
- PhAI foundation: Larsen et al. Science 2024

## 0.3.0 — 2026-07

### Synthetic data (Melgalvis & Rekis 2026)
- New generator `data/synthetic_melgalvis.py`: log-normal unit-cell volume,
  lattice derivation with realistic ratios/skew, artificial-molecule clusters
  (covalent bonds, element freqs, optional special-position seeds, H addition)
- Training: `--use-melgalvis-gen` / `--melgalvis-mode` on `train_strong_prior.py`
- Shard mode `melgalvis` in `write_training_shard` / `iter_training_samples`
- Math note: `docs/math/synthetic_melgalvis.md`
- Pilot retrain: `strong_prior_melg.md` (N=120; frac≤20° ≈22%)
- **Long Melgalvis XL retrain** (N=1200, H=192, L=4, ~32 min): 
  `strong_prior_melg_xl.npz` — hold-out frac≤20° ≈**22%**, seedOK rate **12.5%**,
  mapCC ≈0.49, hard strict **0/12** (still below 30% bar)

### Seeding UX
- `gps-make-seed --from-cif` for AlphaFold/RoseTTAFold/experimental model fragments

### Honest limits (unchanged in spirit)
- Hard ab initio seed bar still ~21–22% ≤20°; partial-φ remains the hard-data path

## 0.2.1 — 2026-07

### Ship / packaging
- Version bump for first public tag **v0.2.1**
- `__version__` aligned with `pyproject.toml`
- Release notes: `docs/RELEASE.md` (build, tag, PyPI upload)
- **Paper pack:** full manuscript draft (`docs/arxiv_draft.md`), Figs. 1–4
  (`scripts/plot_paper_figures.py` → `docs/figures/paper_fig*`), hub `docs/paper/`
- **PDF:** `docs/paper/arxiv_draft.pdf` via pandoc + tectonic
  (`scripts/build_paper_pdf.py`); authors **Grok (xAI)** and **Joe**
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
