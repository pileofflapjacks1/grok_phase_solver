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
- [ ] Full AlphaFold-MR production pipeline
- [ ] Radiation-damage / anisotropy physical models (beyond isotropic B)
- [ ] Agentic auto-benchmark at COD/PDB scale
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
- [ ] PyPI release
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
- [ ] GUI / web upload

## Immediate next actions (honest priority)

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
18. [~] PyPI packaging ready (build/check); full upload when credentials available  
19. [x] **Lane C:** expanded experimental COD Fobs scoreboard (2016452, 2100301,
      2017775) + Fcalc controls + oracle partial-φ rows; `arxiv_draft.md`;
      FOR_REVIEWERS C9. SHELXD binary still not present (SHELXS used when available).

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
