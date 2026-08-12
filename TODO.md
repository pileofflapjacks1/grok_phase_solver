# TODO — grok_phase_solver

AI-driven general solver for the X-ray crystallography phase problem.

**Repo:** https://github.com/pileofflapjacks1/grok_phase_solver  
**Physics core:** \(\rho(\mathbf{r}) = \frac{1}{V}\sum |F| e^{i\varphi} e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}\) — recover \(\varphi\) under positivity, atomicity, symmetry, Parseval.

**Status legend:** `[x]` done · `[ ]` todo · `[~]` partial / research ongoing  

> **Truth statement:** Completing this checklist advances a **correct open framework** and reproducible baselines. It does **not** mean the crystallographic phase problem is solved for arbitrary macromolecules. See `docs/math/uniqueness_and_bounds.md`.

---

## Design principles (all phases)

- [x] Every ML component has a physics fallback / explainability path
- [x] Derive losses, architectures, and post-processing from first principles
- [x] Modular APIs (`ReflectionTable`, `CrystalStructure`) over ad-hoc arrays
- [x] Document math (notebooks + `docs/math/`)
- [x] Open science: MIT license, cite PhAI / Cowtan / COD / gemmi
- [x] Push regularly with clear commit messages
- [x] Iterate: Plan → Code → Test → Analyze math → Refine → Commit

---

## Phase 1 — Baseline reproduction & data pipeline ✅

### Repo & packaging
- [x] Project structure, package, LICENSE, README, GitHub

### I/O module
- [x] CIF (gemmi), HKL CIF, SHELX HKL, ReflectionTable / CrystalStructure
- [x] MTZ read/write via gemmi (`io/mtz.py`)
- [x] Pure-Python CIF fallback (`io/cif_pure.py`)

### Physics foundation
- [x] Form factors, Fcalc, density FFT, reciprocal geometry
- [x] Parseval / Friedel diagnostics (`physics/parseval.py`)
- [x] Math overview + failure modes + uniqueness notes

### Solvers & metrics
- [x] Charge flipping, HIO, random; origin-invariant map CC, MPE, FSC, R
- [x] Centrosymmetric phase constraint option in CF

### Data & PhAI hooks
- [x] COD samples 2100301, 2017775; synthetic degradation suite
- [x] PhAI interface + ERDA docs
- [ ] Official PhAI weights (external download — not redistributed)
- [ ] Reproduce published PhAI numbers (blocked on weights)

---

## Phase 1b — Classical methods (Cowtan) ✅

- [x] Cowtan notes integration
- [x] Patterson + peak pick + autocorrelation identity
- [x] Direct methods (E, triplets, tangent multi-start)
- [x] Notebook 02 derivations
- [x] MIR / MAD / MR simulators + hybrid AI test design
- [x] Difference Patterson HA search (`solvers/difference_patterson.py`)
- [x] Blow–Crick multi-derivative / SIR FOMs (`solvers/mir_blow_crick.py`)

---

## Phase 2 — Enhanced synthetic data & training ✅ / 🚧

### Synthetic data
- [x] Expanded fragment library (imidazole, phosphate, water, chloro-phenyl, …)
- [x] Partial occupancy + heavy-atom injection
- [x] P−1 centrosymmetric expansion helper
- [x] Training shard writer + generator pseudocode
- [x] Wilson plot + domain-gap metrics (`data/wilson.py`)
- [~] Full space-group packing for all IT groups (P1/P−1 solid; general SG via gemmi expand still needs lattice sampling work)
- [~] Millions-scale (mechanism ready; wall-clock generation is user-side)

### Representations & models
- [x] Voxel / Patterson voxel / triplet reflection graph
- [x] Physics losses (NumPy) + optional torch losses module
- [x] PhaseMLP + training script (`models/phase_mlp.py`, `scripts/train_phase_mlp.py`)
- [x] Hybrid benchmark CLI suites A/B/COD (`scripts/run_hybrid_benchmark.py`)
- [ ] Large CNN/Transformer equivariant production model
- [ ] PhAI weight load + fine-tune (needs ERDA weights)
- [ ] Laplacian sharpness in full training curriculum (torch helper present)

---

## Phase 3 — Hybrid & general solvers + new math ✅ / 🚧

### Implemented
- [x] Density modification / solvent flattening (`solvers/density_modification.py`)
- [x] Hybrid seed + polish (CF / HIO / DM) (`solvers/hybrid.py`)
- [x] Phase blending (complex weighted combination)
- [x] Uniqueness / Parseval / Friedel notebook 03 + bounds doc
- [x] Beamline-style degradations already in synthetic suite (noise, completeness, wedge)
- [x] RAAR / DiffMap / ER projectors (`solvers/iterative_retrieval.py`)
- [x] Free FOM + conditional hybrid polish (`free_fom.py`, `conditional_hybrid.py`)
- [x] Free FOM v2: fix vacuous \(R\), \(R_+\) residual, atomicity scores, calibrated gate (`docs/math/free_fom.md`)
- [x] Multistart ensemble CF+RAAR free-FOM pick (`ensemble.py`)
- [x] DiffMap retune grid (β, charge-flip \(P_S\), δσ)
- [x] Physics-recycle net on hard cells (`recycle_net.py`)
- [x] COD 2016452 PhAI+RAAR conditional hybrid benchmark

### Still open research / scale
- [ ] Envelope detection tuned for proteins (solvent_fraction API only)
- [ ] Diffusion generative models conditioned on |F|
- [~] Structure-prediction seeding: `gps-make-seed --from-cif` (AF/RF fragments); full AF-MR pipeline optional
- [ ] Radiation-damage / anisotropy physical models (beyond isotropic B)
- [ ] Agentic auto-benchmark at COD/PDB scale
- [x] Melgalvis & Rekis (2026) synthetic generator + train flag; pilot retrain (frac≤20°~22%)
- [ ] Scale Melgalvis train to 10³–10⁴ + optional PhAI fine-tune on new synth
- [~] Hard-region strict success ~0% ab initio; partial-φ path formalized (oracle/fragment curves)

---

## Phase 4 — Validation, visualization, deployment ✅ / 🚧

- [x] Systematic hybrid benchmark JSON output
- [x] Diagnostic plots script (`scripts/plot_diagnostics.py`)
- [x] arXiv preprint skeleton (`docs/arxiv_skeleton.md`)
- [x] GitHub Actions CI (pytest)
- [x] Math failure documentation
- [x] Head-to-head harness vs SHELXD/SHELXS (runners + dual_space; binaries external)
- [ ] PyMOL / web density viewer
- [ ] PyPI release (**0.13.2** — pending upload)
- [ ] Full public multi-TB dataset mirrors
- [ ] Grok agent beamline design integration

---

## End-user pipeline (scientists)

- [x] `gps-solve` CLI for experimental HKL (+ INS/cell/SG)
- [x] Loaders: SHELX hkl/ins, CIF HKL, MTZ
- [x] Exports: phases, density, peaks, report.md
- [x] User guide + demo (`docs/USER_GUIDE.md`, `examples/demo_solve/`)
- [x] SHELXL-style `.res` trial model export (`trial.res`)
- [x] **Lane B partial-φ UX:** `--phase-seed-res` / `--seed-peaks-csv` /
      `--seed-atoms-csv` / HA pair / `--patterson-ha`; `gps-make-seed`;
      seed-quality section in `report.md` (`solvers/seed_import.py`)
- [x] **GUI (Streamlit):** `gps-gui` / `python -m grok_phase_solver.gui`
      (`gui/app.py` + `gui/backend.py`); optional dep `.[gui]`
- [x] **Vol-band next-action** in `report.md` / GUI (`pipeline/next_action.py`)

## Immediate next actions (honest priority)

### v0.4.0 Carrozzini 2025 track
- [x] `seed_quality_predictor` (`metrics/seed_quality.py`) — heuristic Class 0/1
- [x] `DM+AI tangent hybrid` (`dm_ai_weight` / modified tangent)
- [x] Expanded COD AI-PhaSeed subset bench (`run_ai_phaseed_extended_benchmark.py`)
- [x] Docs + CHANGELOG 0.4.0 + CLI/GUI flags
- [x] Optional: train/persist seed Class 0/1 model on synthetic oracle labels (v0.8)
- [ ] Optional: download larger COD Vol 1000–3500 Å³ subset for stratified bench

### v0.5.0 hybrid / SG / UQ track
- [x] Physics Langevin **diffusion hybrid** (`models/diffusion_phase.py`, methods + CLI)
- [x] Fuller gemmi **SG helpers** (`physics/symmetry.py`) + report diagnostics
- [x] **Predicted-model** seeding (AF/OpenFold3/Boltz CIF) + `combine_phase_seeds`
- [x] Optional **device** backend (`--device` / `--gpu`, torch FFT)
- [x] **Uncertainty** multistart circular + free-FOM bootstrap

### v0.6.0 GraphPhaseNet v5 + score diffusion + UX
- [x] GraphPhaseNet **v5** features (d_in=14) + κ-gated edges
- [x] `run_strong_prior_v5.py` (quick/pilot/scale/scale-xl); pilot scoreboard
- [x] Trainable **PhaseScoreNet** + diffusion_hybrid_v2 path
- [x] GUI multi-plane slices + plotly volume HTML
- [x] MERGE-class `merge_symmetry_equivalents`
- [x] Ensemble threaded `n_jobs`
- [x] Dockerfile

### v0.7.0 Melgalvis curriculum + GraPhAI edges + hard-path UX
- [x] COD-like / hard Melgalvis presets; HA + partial occ; low-res fraction
- [x] GraphPhaseNet v5.1 κ power-law edges + self-loops
- [x] AI-PhaSeed seed-fraction heuristic; multi-seed agreement boost
- [x] Protein-mode / auto solvent DM
- [x] SE(3) diffusion stub (research-only)
- [x] PyPI 0.7.0 upload

### v0.8.0 GraphPhaseNet v6 + seed RF + hard-path polish
- [x] GraphPhaseNet **v6** d_in=18 HA/low-res features + κ-gated edges
- [x] `run_strong_prior_v6.py` (quick/pilot/scale/scale-xl); pilot scoreboard
      (**~24%** frac≤20° on N=200 hold-out; above ~22% plateau; **not** 30% bar)
- [x] Trainable seed-quality classifier (NumPy logistic / sklearn RF) + scoreboard
- [x] Full Fcalc soft prior for fragment seeds (from post-0.7 main)
- [x] SG name aliases; trial_complete research mode; SE(3) research features
- [x] Version 0.8.0 packaging + tests (187 passed)
- [x] PyPI / GitHub Release 0.8.0 (when published by maintainer)

### v0.9.0 GraphPhaseNet v7 + Carrozzini bins + packing
- [x] GraphPhaseNet **v7** d_in=22 multipath + κ×E edges + bin CE loss
- [x] `run_strong_prior_v7.py`; Melgalvis multi-fragment + acta2026 preset
- [x] Seed bin features; multi-seed bin agreement; quadrant discretization
- [x] Solvent estimate volume/N prior; docs graphai_external
- [x] Version 0.9.0 packaging + tests
- [x] Tag + GitHub Release 0.9.0

### v0.10.0 HA curriculum + HDM + stratified benches
- [x] GraphPhaseNet **v8** curriculum (`ha` preset) + stratified Z/HA reporting
- [x] Hybrid Difference Map research path (`--method hdm`)
- [x] AI-PhaSeed bin-quality filter + |E| floor + Vol-band seed fraction
- [x] `run_cod_stratified_bench.py` skeleton; `ha_heavy_config`
- [x] Version 0.10.0 packaging + tests
- [ ] PyPI 0.10.0 upload (maintainer token) — optional if 0.11.0 supersedes

### v0.11.0 Melgalvis large-cell + GraphPhaseNet v9 + seed hardening
- [x] Melgalvis ring scaffolds, void packing, `large_cell` / stronger HA presets
- [x] GraphPhaseNet **v9** d_in=26 + `run_strong_prior_v9.py` + Vol-band stratify
- [x] AI-PhaSeed multi-bin filter, continuous multi-seed agreement, Class 0/1 diagnostics
- [x] Docs: synthetic_melgalvis, graph_phase_net_v9, FOR_REVIEWERS C19, CHANGELOG
- [x] Tests `test_v011_melgalvis_graph_seed.py`; version 0.11.0
- [x] Tag + GitHub Release 0.11.0

### v0.12.0 GraphPhaseNet v10 + generative research path
- [x] GraphPhaseNet **v10** d_in=30 + generative_structure research method
- [x] Tag + GitHub Release 0.12.0

### v0.13.0 GraphPhaseNet v11 + CrystalX typing + XDXD coords
- [x] GraphPhaseNet **v11** d_in=34 + `run_strong_prior_v11.py`
- [x] Melgalvis intermolecular contacts + `xdxd_lowres_config`
- [x] CrystalX-inspired peak→atom typing → `trial.res`
- [x] XDXD-inspired multi-start coordinate proposal (`xdxd_structure`)
- [x] GraPhAI external discovery skeleton (`third_party/graphai`)
- [x] Auto routing: very low-res/sparse partial-φ recommendation
- [x] Docs + tests; version 0.13.0
- [ ] Cluster **scale-xl** retrain (5k–10k) toward 30% seed bar
- [ ] Wire real GraPhAI local API when user installs Zenodo package
- [x] Expand experimental COD Vol 1000–3500 Å³ panel (`cod_stratified_bench` 6 COD, mid-band filled)
- [ ] True SE(3) / trained XDXD weights (external research)
- [ ] Optional cctbx backend; OMC25-scale data mirror
- [ ] PyPI 0.13.2 upload (pending)
- [ ] Streamlit multi-plane density / PyMOL export helper
- [ ] Full agent-style experimental action recommender



1. [x] PhAI weights runner + scoreboard  
2. [x] Scientist-facing `gps-solve` pipeline  
3. [x] Free-FOM science fix + calibration (v2)  
4. [x] Solvability failure taxonomy A/B/C (`metrics/failure_taxonomy.py`)  
4b. [x] Free-FOM v2.1 anti-false-atomicity + PhAI-seeded taxonomy  
4c. [x] AI-PhaSeed (PhAI seed + phase extension + free-FOM polish)  
4d. [x] Domain-matched hard-P1 prior (OI training + free-FOM origin search)  
5. [x] Improve auto method + wire AI-PhaSeed / ensemble into `gps-solve`  
6. [x] Peak → SHELXL `.res` fragment export (`trial.res`)  
7. [x] Experimental HKL scoreboard (COD 2017775 + controls)  
8. [x] External validation vs SHELXD — runner + dual_space baseline + `run_shelxd_h2h.py` (binary optional academic install)  
8b. [x] SHELXS head-to-head — `shelxs_runner.py` + `run_shelxs_h2h.py` (detect `ShelX/shelxs`, gitignore binaries)  
9. [x] Wilson domain-gap metrics + **close-the-gap matching** (`wilson_match.py`, template, `--wilson-match` train)  
10. [x] Stronger prior architecture (GraphPhaseNet triplet GNN + AI-PhaSeed)  
11. [x] Scale graph prior v2 (250 structs, H=128/L=3, curriculum multi-pass, triplet aux, vectorized Â) — mapCC≈0.51 matches hP1, beats CF; still 0% strict hard solves  
12. [x] SHELXS H2H with local academic binary (`ShelX/shelxs`); re-run SHELXD if that binary is added later  
13. [x] Partial-φ / fragment seed API + hard-cliff curves (`partial_seed.py`, `run_partial_seed_benchmark.py`)  
14. [x] A+B: Wilson-matched retrain + strong-seed metrics/loss (v3; hold-out strong MPE≈59°, frac≤20°≈21% vs 30% bar; still 0% strict)  
15. [x] Product trio: auto→ensemble (easy), partial-φ hard path + demo, SHELXS+SHELXE polish + SHELXL docs  
16. [~] **Lane A (v4):** residual GNN + Adam + d_in=10 + **1200-struct XL** train  
      (`--scale-xl --wilson-match`). Hold-out frac≤20° still **~21%** (bar 30%);  
      seedOK rate 5–12%; strict hard solves still 0%. Checkpoint `strong_prior.npz`  
      + `strong_prior_v4_xl.npz` / `_ft.npz`. **Mean seed bar not cleared by scale alone.**  
17. [ ] Further scale (10⁴ cells / torch equivariant) **or** accept ceiling and invest in partial-φ UX  
18. [x] **v0.2.1 ship:** version align, tag, `docs/RELEASE.md`; build+twine check  
      PyPI **0.13.2** pending upload
19. [x] **Lane C:** expanded experimental COD Fobs scoreboard (2016452, 2100301,
      2017775) + Fcalc controls + oracle partial-φ rows; `arxiv_draft.md`;
      FOR_REVIEWERS C9. SHELXD binary still not present (SHELXS used when available).  
20. [x] **Paper pack:** Figs. 1–4 from scoreboards; full methods draft; `docs/paper/`  
21. [x] BibTeX + pandoc PDF + GitHub Release v0.2.1 assets  
22. [x] Authors: Grok (xAI) and Joe  
23. [ ] Affiliations / funding + arXiv submit (optional)  
24. [x] **Paper / claim freeze to v0.13.1:** draft + FOR_REVIEWERS + Fig. 6 Vol-band;
      GraphPhaseNet v9–v11 recorded as negative; no software bump
25. [x] **Partial-φ next-action product:** Vol-band + seed-source chooser in
      `report.md` / `solve_summary.json` / GUI banner

---

## Quick reference — key paths

| Area | Path |
|------|------|
| **User solve** | `gps-solve`, `pipeline/solve.py`, `docs/USER_GUIDE.md` |
| Package | `src/grok_phase_solver/` |
| Classical solvers | `solvers/{patterson,direct_methods,charge_flipping,hio,baseline}.py` |
| Diff. Patterson / MIR | `solvers/{difference_patterson,mir_blow_crick}.py` |
| Hybrid / DM | `solvers/{hybrid,density_modification,phase_recycle,conditional_hybrid}.py` |
| Ensemble / RAAR / DiffMap | `solvers/{ensemble,iterative_retrieval,recycle_net}.py` |
| Physics | `physics/{structure_factors,density,patterson,parseval,form_factors}.py` |
| Benchmark | `scripts/run_{scoreboard,frontier,ensemble,diffmap_retune,cod_hybrid}_benchmark.py` |
| Math | `docs/math/`, notebooks 01–03 |
| COD samples | `data/raw/cod/` |

---

*Narrative roadmap: [`docs/roadmap.md`](docs/roadmap.md). Uniqueness & non-claims: [`docs/math/uniqueness_and_bounds.md`](docs/math/uniqueness_and_bounds.md).*
