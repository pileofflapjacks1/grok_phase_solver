# Phase-1 Baseline: Success Modes and Mathematical Failures

Empirical notes from the initial physics baselines (charge flipping / HIO / random).

## Metrics that matter

| Metric | Origin sensitive? | Use |
|--------|-------------------|-----|
| Fixed-origin map CC | Yes | Debugging only |
| **Origin-invariant map CC** | No (FFT shift + inversion) | Primary real-space quality |
| Raw mean phase error (MPE) | Yes | Often misleadingly ~90° |
| Origin/enantiomorph-invariant MPE | Partially (discrete \(t\)) | Secondary |

**Important:** A correct density shifted by a lattice translation has near-zero fixed-origin CC and ~90° raw MPE. Always report origin-invariant map CC for ab initio tests.

## Observed results (representative)

### Synthetic P1, 5 random atoms, \(d_{\min}=0.9\) Å

| Method | mapCC (OI) | Notes |
|--------|------------|-------|
| Random phases | ~0.28 | Non-zero because correct \(|F|\) already encode Patterson/autocorrelation |
| Charge flipping | **~0.95** | Near-perfect density recovery |

### COD 2100301 (C₇H₅NO₄, P2₁/c, 68 atoms/cell), \(d_{\min}=1.0\) Å

| Method | mapCC (OI) | R (history) |
|--------|------------|-------------|
| Random | ~0.23 | — |
| Charge flipping | **~0.45** | ~0.38 |

Partial recovery: better than random, not yet atomic interpretability. Room for PhAI seeding + density modification (Phase 2–3).

### Experimental HKL COD 2017775

- 28 457 reflections, \(d\) from 0.45–13.7 Å, space group P2₁2₁2₁.
- Amplitudes loaded; full experimental phasing deferred until NN + hybrid loops (no free phases in file for automatic ground truth without separate Fcalc).

## Why charge flipping can fail

1. **Insufficient atomicity**  
   At low resolution peaks merge; the flip operator no longer separates atoms from noise.  
   *Symptom:* mapCC collapses as \(d_{\min}\) increases (see resolution series).

2. **Non-convex modulus constraint**  
   \(\mathcal{P}_F\) is a projection onto a non-convex set. Multiple fixed points exist (trivial associates + false solutions).  
   *Symptom:* R decreases while mapCC stays moderate (consistent moduli, wrong phases).

3. **Missing / weak symmetry exploitation**  
   Centrosymmetric groups have phases \(0/\pi\); unconstrained complex phases waste degrees of freedom.  
   *Mitigation (planned):* force real \(F\) for centrosymmetric SG.

4. **Weak-phase randomization schedule**  
   Permanent randomization of weak reflections prevents late-stage refinement.  
   *Current fix:* randomize only in the first third of iterations.

5. **Homometric / pseudo-translational structures**  
   Distinct densities can share \(|F|\). Rare for small organics; more relevant for minerals / high symmetry.

6. **Random “organic” generators**  
   Phase-1 synthetic atoms lack bonding geometry; form-factor weighted peaks still work at high resolution but domain gap vs real COD molecules remains (Phase 2).

## Connection to uniqueness theory

- **CDI / support oversampling:** uniqueness up to trivial associates when support is known and Fourier data oversampled (Hayes et al.). Crystals are critically sampled on \(\mathbf{h}\in\mathbb{Z}^3\); uniqueness relies on **atomicity**, not support oversampling.
- **Direct methods:** triplet invariants \(\varphi_{\mathbf{h}}+\varphi_{\mathbf{k}}+\varphi_{-\mathbf{h}-\mathbf{k}}\) concentrate for sparse atomic \(\rho\); fail when \(N_{\mathrm{atoms}}\) large relative to \(N_{\mathrm{refl}}\).
- **PhAI prior:** learns \(p(\varphi\mid|F|)\) on synthetic distributions—effective regularization when classical projections fail at ~2 Å.

## Action items for Phase 2

1. Centrosymmetric phase constraint in CF/HIO  
2. Multi-start CF + histogram matching  
3. NN phase seeds → short CF polish  
4. Domain-gap: compare synthetic vs COD \(|F|(s)\) Wilson plots  
