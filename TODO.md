# TODO — grok_phase_solver

AI-driven general solver for the X-ray crystallography phase problem.

**Repo:** https://github.com/pileofflapjacks1/grok_phase_solver  
**Physics core:** \(\rho(\mathbf{r}) = \frac{1}{V}\sum |F| e^{i\varphi} e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}\) — recover \(\varphi\) under positivity, atomicity, symmetry, Parseval.

**Status legend:** `[x]` done · `[ ]` todo · `[~]` in progress

---

## Design principles (all phases)

- [x] Every ML component has a physics fallback / explainability path
- [x] Derive losses, architectures, and post-processing from first principles
- [x] Modular APIs (`ReflectionTable`, `CrystalStructure`) over ad-hoc arrays
- [x] Document math (notebooks + `docs/math/`)
- [x] Open science: MIT license, cite PhAI / Cowtan / COD / gemmi
- [ ] Push regularly with clear commit messages (ongoing)
- [ ] Iterate: Plan → Code → Test → Analyze math → Refine → Commit

---

## Phase 1 — Baseline reproduction & data pipeline

**Goal:** Robust I/O, physics baselines, COD samples, metrics, repo skeleton.

### Repo & packaging
- [x] Create project structure (`src/`, `data/`, `notebooks/`, `docs/`, `tests/`, `scripts/`)
- [x] `pyproject.toml` / installable package `grok_phase_solver`
- [x] MIT LICENSE, README, `.gitignore`
- [x] Public GitHub repo + initial push

### I/O module
- [x] CIF reader (gemmi small-structure path)
- [x] COD-style HKL / structure-factor CIF
- [x] SHELX-style free-format HKL
- [x] `ReflectionTable` + `CrystalStructure` dataclasses
- [ ] MTZ support (optional / stub)
- [ ] Pure-Python CIF fallback without gemmi

### Physics foundation
- [x] Atomic form factors + Debye–Waller
- [x] Structure factor direct sum \(F(hkl)\)
- [x] Electron density via inverse FFT
- [x] Reciprocal geometry (d-spacings, hkl generation, absences)
- [x] Math overview (`docs/math/phase_problem_overview.md`)
- [x] Document failure modes (`docs/math/baseline_failure_modes.md`)

### Solvers (physics baselines)
- [x] Charge flipping (Oszlányi–Sütő)
- [x] Hybrid Input–Output (HIO / Fienup)
- [x] Random-phase null baseline
- [x] Unified `run_physics_baseline` API + CLI (`gps-baseline`)

### Metrics
- [x] Mean phase error (wrapped)
- [x] Origin / enantiomorph-invariant MPE
- [x] Map CC + **origin-invariant** map CC (FFT shift)
- [x] Fourier shell correlation (FSC)
- [x] R-factor / R-free hooks

### Data
- [x] COD download helpers
- [x] Sample COD 2100301 (small organic, P2₁/c)
- [x] Sample COD 2017775 (roxithromycin + experimental HKL)
- [x] Synthetic random organics + degradation suite (noise, completeness, resolution)
- [x] Simulate synthetic data pipeline (vary resolution / completeness / noise)

### PhAI integration (hooks)
- [x] Document PhAI (Science 2024) + ERDA archive
- [x] `PhAIInterface` stub + `third_party/phai/README.md`
- [ ] Download / place official PhAI weights under `third_party/phai/weights/`
- [ ] Reproduce PhAI phasing on published test sets (when weights available)

### Baselines run & docs
- [x] Reproduce physics phasing on synthetic + COD Fcalc
- [x] Record metrics (mapCC ~0.87 synthetic; ~0.83 COD @ 0.9 Å)
- [x] Notebook 01: math + baseline walkthrough
- [x] Phase-1 script: `scripts/run_phase1_baseline.py`

---

## Phase 1b — Classical methods (Cowtan)

**Goal:** Patterson, direct methods, experimental-phasing taxonomy for comparison and hybrid design.

### Cowtan (2001) integration
- [x] Review full ELS article (`phaseproblem.pdf`)
- [x] Notes: `docs/cowtan_phase_problem_notes.md`
- [x] Map article sections → code modules

### Patterson methods
- [x] Derive / implement \(P(\mathbf{u})\) from \(|F|^2\)
- [x] Autocorrelation identity check \(P \leftrightarrow \mathrm{autocorr}(\rho)\)
- [x] Peak picking + vector-recovery diagnostics
- [x] Baseline method `patterson`

### Direct methods
- [x] Normalized \(E\)-values
- [x] Triplet enumeration + Cochran-style weights
- [x] Tangent-formula multi-start solver
- [x] Baseline method `direct_methods`
- [x] Notebook 02: Patterson & triplet derivations

### Experimental phasing simulators
- [x] MIR simulation + Harker phase indication
- [x] MAD multi-λ + Bijvoet pairs (simplified \(f',f''\))
- [x] MR (model Fcalc at true / noisy pose)
- [x] Hybrid AI test design: `docs/hybrid_ai_tests.md`
- [ ] Difference-Patterson heavy-atom search automation
- [ ] Full multi-derivative MIR combination (Blow–Crick style FOMs)

---

## Phase 2 — Enhanced synthetic data & training

**Goal:** Scale physically valid structures; train/extend neural nets with physics-informed losses.

### Synthetic data at scale
- [x] Fragment-based builder (benzene / carboxyl / peptide stubs)
- [x] Training shard writer (`synthetic_v2.write_training_shard`)
- [ ] Expand fragment library (COD/ZINC-guided, more functional groups)
- [ ] Sample unit cells respecting space-group constraints
- [ ] Space-group-aware packing + clash / energy filters
- [ ] Disorder, partial occupancies, heavy atoms
- [ ] Millions-scale pipeline (HDF5 / LMDB / streaming)
- [ ] Document generators with pseudocode + statistical validation
- [ ] Domain-gap metrics (synthetic vs experimental \(|F|\) / Wilson plots)

### Representations
- [ ] Graph representations of reflections / atoms
- [ ] Voxelized density inputs
- [ ] Resolution-dependent falloff / form-factor tables in training

### Models & training
- [x] Physics-informed losses (NumPy): phase, positivity, modulus, triplets
- [ ] Port losses to PyTorch / TensorFlow
- [ ] CNN / Transformer hybrid architecture
- [ ] Equivariant layers for symmetry
- [ ] Physics losses: Laplacian peak sharpness, Fourier consistency (full)
- [ ] Phase-seeding + density-modification hybrids (AI-PhaSeed style)
- [ ] Torch training loop + checkpointing + logging
- [ ] Triplet-invariant auxiliary head
- [ ] Train on multi-resolution / incomplete data curricula

### PhAI extend
- [ ] Load official PhAI weights
- [ ] Reimplement / wrap inference for our `ReflectionTable`
- [ ] Compare PhAI vs CF vs DM on shared test sets
- [ ] Fine-tune or distill into grok_phase_solver models

### Hybrid benchmark harness
- [ ] CLI: `scripts/run_hybrid_benchmark.py` (suites A–E in hybrid_ai_tests.md)
- [ ] Automated tables: method × resolution × completeness × mapCC

---

## Phase 3 — Hybrid & general solvers + new math

**Goal:** Production-minded hybrids; macromolecules; new theory.

### Iterative hybrid loops
- [ ] NN-seeded charge flipping
- [ ] NN-seeded HIO / difference-map / charge-flipping polish
- [ ] Density modification (solvent flattening, histogram matching)
- [ ] Envelope detection for macromolecules (solvent content)

### Experimental hybrids
- [ ] MIR/MAD: NN phase combination given multi-channel \(|F|\)
- [ ] MR: AlphaFold / homology model as prior + AI polish
- [ ] Non-centrosymmetric large cells
- [ ] Radiation damage & anisotropy models

### New mathematics & generative models
- [ ] Information-theoretic / PDE constraints on \(\rho\)
- [ ] Diffusion / generative models conditioned on amplitudes
- [ ] Provable bounds from autocorrelation / Patterson
- [ ] Convergence notes for iterative projection algorithms
- [ ] Uniqueness / oversampling notebooks (expand)

### Agentic testing
- [ ] Auto-benchmark on COD / PDB subsets
- [ ] Beamline simulation (completeness, missing wedge, dose)

---

## Phase 4 — Validation, visualization, deployment & open science

**Goal:** Rigorous eval, usable tools, preprint, public artifacts.

### Validation
- [ ] Cross-validation on withheld experimental data
- [ ] Compare to SHELXD / SHELXT, Phenix, other baselines
- [ ] Report mean phase error, map CC, R-factors, FSC systematically
- [ ] Document mathematical failure cases in paper-ready form

### Visualization & UX
- [ ] Electron density visualization (PyMOL API and/or web viewer)
- [ ] Phase-error analysis plots
- [ ] Interactive notebooks / dashboard

### Release
- [x] Reproducible code on GitHub
- [ ] Full dataset release (synthetic + processed real; size limits / mirrors)
- [ ] Detailed math papers / notebooks (theorems, derivations)
- [ ] arXiv preprint skeleton
- [ ] Model zoo + trained weights
- [ ] LICENSE already MIT; dual-license note if needed (Apache option)

### Future / agents
- [ ] Integrate with Grok agents for autonomous experiment design (beamline params)
- [ ] Continuous integration (pytest on PR)
- [ ] Packaging for PyPI (`pip install grok-phase-solver`)

---

## Immediate next actions (priority queue)

1. [ ] PyTorch training loop on fragment NPZ shards + physics losses  
2. [ ] Hybrid benchmark CLI (suites A–B first: ab initio + MIR)  
3. [ ] Obtain PhAI weights from ERDA and wire inference  
4. [ ] Expand synthetic packing / space groups for COD-like domain  
5. [ ] Centrosymmetric phase constraints in CF/DM for P2₁/c  

---

## Quick reference — key paths

| Area | Path |
|------|------|
| Package | `src/grok_phase_solver/` |
| Classical solvers | `solvers/{patterson,direct_methods,charge_flipping,hio,baseline}.py` |
| Physics | `physics/{structure_factors,density,patterson,form_factors}.py` |
| MIR/MAD/MR | `data/experimental_phasing.py` |
| Phase-2 synth | `data/synthetic_v2.py` |
| Losses | `models/losses.py` |
| Math docs | `docs/math/`, `docs/cowtan_phase_problem_notes.md` |
| Hybrid tests | `docs/hybrid_ai_tests.md` |
| Notebooks | `notebooks/01_*.md`, `notebooks/02_*.md` |
| COD samples | `data/raw/cod/` |

---

*Update checkboxes as work lands. Detailed narrative roadmap: [`docs/roadmap.md`](docs/roadmap.md).*
