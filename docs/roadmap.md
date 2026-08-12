# Roadmap — grok_phase_solver

## Phase 1 — Baseline & data pipeline ✅

- [x] Repository structure (`src/`, `data/`, `notebooks/`, `docs/`)
- [x] I/O: CIF (gemmi), HKL CIF, SHELX-style HKL
- [x] Physics: form factors, Fcalc, density FFT, reciprocal geometry
- [x] Solvers: charge flipping, HIO, random null baseline
- [x] Metrics: MPE (origin-invariant), map CC, FSC, R-factor
- [x] COD download helpers + sample structures (2100301, 2017775)
- [x] Synthetic random organics + degradation suite
- [x] PhAI interface stub + third_party docs
- [x] Math overview document
- [x] Reproducible baseline numbers on COD 2100301 / synthetic grid
  - Synthetic CF origin-invariant mapCC ≈ 0.87 @ 1.0–1.2 Å
  - COD 2100301 CF mapCC ≈ 0.83 @ 0.9 Å; degrades at lower resolution
- [ ] Optional: pure-Python CIF fallback without gemmi

## Phase 1b — Cowtan classical methods ✅

- [x] Review Cowtan ELS (2001); `docs/cowtan_phase_problem_notes.md`
- [x] Patterson function + peak picking + autocorrelation check
- [x] Direct methods: E-values, triplets, tangent multi-solution
- [x] Notebook 02: Patterson & triplet derivations
- [x] MIR / MAD / MR simulators + hybrid AI test design
- [x] Wire into baseline API (`patterson`, `direct_methods`)

## Phase 3.5 — Carrozzini 2025 AI-PhaSeed alignment (v0.4.0) ✅

- [x] Modified tangent DM+AI hybrid (`dm_ai_weight`)
- [x] Seed-quality Class 0/1 predictor (heuristic; optional sklearn RF)
- [x] Low-res / large-Vol EDM path
- [x] CLI / GUI flags + report diagnostics
- [x] Extended subset benchmark harness
- [ ] Full COD 1505-style stratified panel (optional download)

## Phase 3.6 — Diffusion / SG / predicted seeds / UQ (v0.5.0) ✅ / 🚧

- [x] Physics Langevin diffusion hybrid (experimental)
- [x] gemmi SG helpers (expand, centro, absences diagnostics)
- [x] Predicted-model CIF seeding + multi-seed combine
- [x] Optional device FFT + phase uncertainty diagnostics
- [ ] Trained equivariant diffusion weights (external)
- [ ] Full Fobs symmetry averaging / twinning product path
- [ ] GUI 3D density viewer

## Phase 2 — Enhanced synthetic data & training 🚧

- [x] Fragment-based molecule builder (benzene / carboxyl / peptide)
- [x] Training shard writer (`synthetic_v2.write_training_shard`)
- [x] Physics-informed losses (NumPy: phase, positivity, modulus, triplets)
- [x] Melgalvis packing + large-cell / HA / xdxd curricula (v0.11–v0.13)
- [x] GraphPhaseNet multipath priors v5–v11 + scale-xl scripts
- [x] CrystalX-inspired peak typing (heuristic) → trial.res
- [x] Generative / XDXD-style research coordinate proposal (not auto)
- [x] GraPhAI external discovery skeleton (no weight redistribution)
- [ ] Space-group aware packing for all IT groups
- [ ] Millions-scale dataset pipeline (HDF5 / LMDB)
- [ ] Trained equivariant / XDXD weights (external)
- [ ] Cluster scale-xl clear of 30% seed bar
- [x] Expanded COD Vol 1000–3500 experimental panel (local 6-structure / 48-run; not 1505-COD)
- [x] Paper / claim freeze to v0.13.1 (Figs. 1–6; Vol-band headline; v3–v11 negative)

## Phase 3 — Hybrid solvers & new mathematics

- [ ] NN-seeded CF / HIO / difference-map loops
- [ ] Envelope detection for macromolecules
- [ ] Diffusion / generative models conditioned on amplitudes
- [ ] Patterson / autocorrelation theoretical bounds
- [ ] Non-centrosymmetric large cells; protein MR + AlphaFold hooks
- [ ] Radiation damage & anisotropy models
- [ ] Agentic auto-benchmark on COD/PDB subsets

## Phase 4 — Validation, visualization, deployment

- [ ] Comparison to SHELXD/SHELXT, Phenix.twinning, etc.
- [ ] Density visualization (PyMOL / NGL / web)
- [ ] Interactive notebooks + dashboards
- [x] arXiv preprint draft (claim freeze v0.13.1; submit still optional)
- [ ] Public datasets + model zoo
- [ ] Grok agent integration for beamline experiment design

## Design principles (all phases)

1. Every ML component has a physics fallback.
2. Derive losses from first principles; document in `docs/math/`.
3. Prefer modular APIs (`ReflectionTable`, `CrystalStructure`) over ad-hoc arrays.
4. MIT/Apache open science; cite PhAI, Cowtan, COD, gemmi, ITC.
