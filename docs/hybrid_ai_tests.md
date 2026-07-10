# Hybrid AI test design (MIR / MAD / MR + ab initio)

Grounded in Cowtan (2001) experimental phasing taxonomy and our Phase-1 ab initio baselines.

## Design principles

1. **Every AI channel has a classical control** (Patterson HA find, direct methods, CF/HIO, simple Harker phases).
2. **Multi-channel inputs** match how experiments actually present data (not single \(|F|\) only).
3. **Metrics:** origin-invariant map CC, weighted MPE, FOM calibration, HA site recovery (Tol Å).
4. **Failure injection:** non-isomorphism, noise, incompleteness, wrong hand, low anomalous signal.

## Test suite A — Ab initio classical vs AI

| ID | Data | Methods | Pass criterion (v0) |
|----|------|---------|---------------------|
| A1 | Synthetic P1, \(N\le 8\), \(d_{\min}\le 1.0\) Å | CF, DM, random | CF mapCC_OI > 0.7 |
| A2 | COD 2100301 Fcalc series 0.9–2.0 Å | CF, DM | mapCC decreases with \(d_{\min}\) documented |
| A3 | Same as A2 @ 2.0 Å | PhAI seed (when available) + CF polish | beat CF-alone mapCC |

## Test suite B — MIR hybrid

**Generator:** `simulate_mir(structure, heavy_element, n_heavy, …)`  
**Features:** `hybrid_feature_stack_mir` → \([|F_P|, |F_{PH}|, |F_H|, \Delta|F|, \cos\varphi_H, \sin\varphi_H]\).

| ID | Scenario | Classical | AI target |
|----|----------|-----------|-----------|
| B1 | 1 Au, perfect iso, 2% noise | Harker `mir_phase_indication` | Predict \(\varphi_P\) from features |
| B2 | 2 derivatives, different sites | Combined FOM product | Multi-input fusion net |
| B3 | Non-iso: jitter protein coords 0.3 Å on derivative | Classical FOM drops | Robust training with iso noise |
| B4 | HA unknown | Patterson on \(\|F_{PH}|^2-|F_P|^2|\) | HA substructure CNN + phasing |

**Metrics:** mapCC_OI of native map; correlation of predicted FOM with true \(\cos(\Delta\varphi)\).

## Test suite C — MAD hybrid

**Generator:** `simulate_mad` (remote / inflection / peak; \(f',f''\)).

| ID | Scenario | Classical | AI target |
|----|----------|-----------|-----------|
| C1 | Se, 2 sites, high f'' | Bijvoet difference Patterson | Multi-λ amplitude encoder |
| C2 | Tiny anomalous (f''=1) | Fails classical | Denoising / learned prior |
| C3 | Friedel incomplete 50% | Weak | Imputation + phase |

**Inputs per reflection:** \(\{|F_\lambda^+|, |F_\lambda^-|\}_\lambda\), \(s=\sin\theta/\lambda\), optional HA mask.

## Test suite D — MR + AI polish

**Generator:** `simulate_mr` (perfect or phase-jittered model).

| ID | Scenario | Classical | AI target |
|----|----------|-----------|-----------|
| D1 | Perfect model @ 2 Å | Model phases | Identity (sanity) |
| D2 | Model missing 30% atoms | Partial Fcalc | Density completion net |
| D3 | Wrong translation 2 Å | Low RF/TF | Reject / re-search |
| D4 | AF2 model, 2 Å RMS | MR score | Phase combination residual learning |

## Test suite E — Phase improvement loop

After any of B–D:

1. Build map from current phases + \(|F_{\mathrm{obs}}|\)
2. Solvent flatten / positivity / NCS average (classical)
3. Optional NN residual density modifier
4. Recombine phases; iterate

**Metric:** free-set R / mapCC vs holdout if available; FSC half-maps on synthetic splits.

## Benchmark harness (planned CLI)

```bash
python scripts/run_hybrid_benchmark.py --suite B --json-out data/processed/hybrid_B.json
```

Status: generators + classical MIR indication land in Phase 2.0; full harness next.

## Mapping to PhAI / Phase 2 training

- Ab initio path = PhAI-style \(|F|\to\varphi\) prior (suite A).
- Experimental path = **conditioned** models \(p(\varphi\mid |F|, \text{HA}, \lambda, F_{\mathrm{model}})\).
- Physics losses: Fourier consistency, positivity, triplet FOM auxiliary head, HA occupancy.
