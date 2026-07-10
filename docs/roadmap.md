# Roadmap — grok_phase_solver

## Phase 1 — Baseline & data pipeline ✅ (in progress)

- [x] Repository structure (`src/`, `data/`, `notebooks/`, `docs/`)
- [x] I/O: CIF (gemmi), HKL CIF, SHELX-style HKL
- [x] Physics: form factors, Fcalc, density FFT, reciprocal geometry
- [x] Solvers: charge flipping, HIO, random null baseline
- [x] Metrics: MPE (origin-invariant), map CC, FSC, R-factor
- [x] COD download helpers + sample structures (2100301, 2017775)
- [x] Synthetic random organics + degradation suite
- [x] PhAI interface stub + third_party docs
- [x] Math overview document
- [ ] Reproducible baseline numbers on COD 2100301 / synthetic grid
- [ ] Optional: pure-Python CIF fallback without gemmi

## Phase 2 — Enhanced synthetic data & training

- [ ] Fragment-based molecule builder (COD/ZINC-guided)
- [ ] Space-group aware packing + clash/energy filters
- [ ] Millions-scale dataset pipeline (HDF5 / LMDB)
- [ ] Graph + voxel dual representations
- [ ] CNN/Transformer hybrid with equivariant layers
- [ ] Physics-informed losses (positivity, Laplacian atomicity, Fourier consistency)
- [ ] PhAI weight load / reimplementation
- [ ] Domain-gap metrics (experimental vs synthetic |F|)

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
- [ ] arXiv preprint skeleton
- [ ] Public datasets + model zoo
- [ ] Grok agent integration for beamline experiment design

## Design principles (all phases)

1. Every ML component has a physics fallback.
2. Derive losses from first principles; document in `docs/math/`.
3. Prefer modular APIs (`ReflectionTable`, `CrystalStructure`) over ad-hoc arrays.
4. MIT/Apache open science; cite PhAI, COD, gemmi, ITC.
