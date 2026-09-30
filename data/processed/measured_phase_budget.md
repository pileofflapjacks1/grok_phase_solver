# Measured-phase information budget

```
Ticket:              P2 (+ optional P1 audit / P3 trail / hygiene)
Information source:  measured-φ (simulated) / fragment / n/a (docs)
Method:              partial_phaseed + measured_phase_noise
Vol-band / d_min / N: hard_p1_0_n14_d1.7 Vol=308 Å³ d_min=1.7 N=14; cod_2012000_fcalc Vol=1027 Å³ d_min=1.0 N=118
Seed quality:        frac≤20° (all strong, mean) = 32%   (bar = 30%)   C4 crossings = 6/18
Strict success:      mixed across grid  (6/18 grid points)
  mapCC_OI= 0.575 (mean)   peak recovery= 0.814 (mean)   R1= 0.415 (mean)
Free FOM:            rank only; composite=0.756  R₊=0.216
Claim status:        does not touch freeze C1–C25; new simulator only
PR / path:           feat/measured-phase-budget
Next action:         If a lab measured-φ seed has ≥~30% of strong |E| within 20°, run partial_phaseed then inspect trial.res in Olex2 / SHELXL.
What we will not do: GraphPhaseNet v12; claim detector exists
Blocker for Joe:     none
```

Hardware **does not exist**. This sweep is a simulator plugged into the
existing partial-φ path. See [docs/math/measured_phase_budget.md](../../docs/math/measured_phase_budget.md).

Mode: **quick**. C4 bar: ≥ 30% of
strong-|E| phases correct within 20° (non-centro TREF window).

C4 crossings on this run: hard_p1_0_n14_d1.7 σ=10° frac=1.00 ≤20°=95%; hard_p1_0_n14_d1.7 σ=20° frac=1.00 ≤20°=75%; hard_p1_0_n14_d1.7 σ=45° frac=1.00 ≤20°=39%; cod_2012000_fcalc σ=10° frac=1.00 ≤20°=96%; cod_2012000_fcalc σ=20° frac=1.00 ≤20°=69%; cod_2012000_fcalc σ=45° frac=1.00 ≤20°=38%

## Cells

| id | SG | Vol (Å³) | d_min | N | n_refl |
|----|----|----------|-------|---|--------|
| hard_p1_0_n14_d1.7 | P1 | 308 | 1.7 | 14 | 264 |
| cod_2012000_fcalc | P1211 | 1027 | 1.0 | 118 | 4308 |

## Grid results

| cell | σ_φ (°) | frac measured | ≤20° meas | ≤20° all strong | C4 | mapCC | peak | R1 | solved | FOM | s |
|------|---------|---------------|-----------|-----------------|----|-------|------|----|--------|-----|---|
| hard_p1_0_n14_d1.7 | 10 | 0.10 | 100% | 10% | no | 0.419 | 0.73 | 0.43 | False | 0.745 | 0.5 |
| hard_p1_0_n14_d1.7 | 10 | 0.30 | 92% | 28% | no | 0.510 | 0.73 | 0.39 | False | 0.755 | 1.0 |
| hard_p1_0_n14_d1.7 | 10 | 1.00 | 95% | 95% | yes | 0.778 | 0.64 | 0.37 | True | 0.802 | 0.7 |
| hard_p1_0_n14_d1.7 | 20 | 0.10 | 75% | 8% | no | 0.421 | 0.73 | 0.43 | False | 0.746 | 0.5 |
| hard_p1_0_n14_d1.7 | 20 | 0.30 | 67% | 20% | no | 0.490 | 0.73 | 0.42 | False | 0.737 | 0.7 |
| hard_p1_0_n14_d1.7 | 20 | 1.00 | 75% | 75% | yes | 0.751 | 0.64 | 0.39 | True | 0.798 | 0.7 |
| hard_p1_0_n14_d1.7 | 45 | 0.10 | 38% | 4% | no | 0.417 | 0.64 | 0.46 | False | 0.746 | 0.5 |
| hard_p1_0_n14_d1.7 | 45 | 0.30 | 50% | 15% | no | 0.473 | 0.73 | 0.45 | False | 0.777 | 0.7 |
| hard_p1_0_n14_d1.7 | 45 | 1.00 | 39% | 39% | yes | 0.630 | 0.82 | 0.36 | False | 0.736 | 0.7 |
| cod_2012000_fcalc | 10 | 0.10 | 99% | 10% | no | 0.373 | 0.94 | 0.51 | False | 0.725 | 6.7 |
| cod_2012000_fcalc | 10 | 0.30 | 96% | 29% | no | 0.703 | 0.96 | 0.42 | True | 0.778 | 7.9 |
| cod_2012000_fcalc | 10 | 1.00 | 96% | 96% | yes | 0.916 | 0.89 | 0.29 | True | 0.789 | 8.9 |
| cod_2012000_fcalc | 20 | 0.10 | 73% | 7% | no | 0.366 | 0.93 | 0.51 | False | 0.720 | 6.9 |
| cod_2012000_fcalc | 20 | 0.30 | 70% | 21% | no | 0.651 | 0.96 | 0.44 | False | 0.765 | 8.1 |
| cod_2012000_fcalc | 20 | 1.00 | 69% | 69% | yes | 0.855 | 0.89 | 0.30 | True | 0.787 | 9.1 |
| cod_2012000_fcalc | 45 | 0.10 | 32% | 3% | no | 0.253 | 0.93 | 0.52 | False | 0.710 | 7.4 |
| cod_2012000_fcalc | 45 | 0.30 | 37% | 11% | no | 0.435 | 0.91 | 0.49 | False | 0.712 | 8.0 |
| cod_2012000_fcalc | 45 | 1.00 | 38% | 38% | yes | 0.912 | 0.89 | 0.28 | True | 0.778 | 8.3 |

## Summary by σ_φ

| σ_φ (°) | n | solved | rate | mean ≤20° all strong | mean mapCC |
|---------|---|--------|------|----------------------|------------|
| 10.0 | 6 | 3 | 50% | 45% | 0.616 |
| 20.0 | 6 | 2 | 33% | 33% | 0.589 |
| 45.0 | 6 | 1 | 17% | 18% | 0.520 |

## Summary by measured fraction

| frac | n | solved | rate | mean ≤20° all strong | mean mapCC |
|------|---|--------|------|----------------------|------------|
| 0.1 | 6 | 0 | 0% | 7% | 0.375 |
| 0.3 | 6 | 1 | 17% | 21% | 0.544 |
| 1.0 | 6 | 5 | 83% | 69% | 0.807 |

## How to regenerate

```bash
python scripts/run_measured_phase_budget.py --quick
python scripts/run_measured_phase_budget.py --full
```

Strict success is never redefined: mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5
**and** R1 ≤ 0.45. Free FOM ranks trials only.

