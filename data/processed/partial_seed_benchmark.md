# Partial-φ / fragment seed hard-cliff benchmark

Science track B: how much **known phase or fragment** information is needed to solve **hard** synthetic cells (n≈12–16, d≈1.5–2.0 Å) via AI-PhaSeed extension.

Cases per condition: **4** hard P1 structures.

## Summary

| Condition | n | solved | rate | mean mapCC | mean peak |
|-----------|---|--------|------|------------|-----------|
| `base=charge_flipping` | 4 | 0 | 0% | 0.516 | 0.58 |
| `base=strong_prior_phaseed` | 4 | 0 | 0% | 0.510 | 0.63 |
| `frag=0.10` | 4 | 0 | 0% | 0.521 | 0.51 |
| `frag=0.25` | 4 | 0 | 0% | 0.526 | 0.58 |
| `frag=0.40` | 4 | 1 | 25% | 0.576 | 0.68 |
| `frag=0.60` | 4 | 2 | 50% | 0.731 | 0.63 |
| `noise=0` | 4 | 4 | 100% | 0.873 | 0.67 |
| `noise=20` | 4 | 3 | 75% | 0.781 | 0.67 |
| `noise=40` | 4 | 1 | 25% | 0.594 | 0.71 |
| `noise=60` | 4 | 0 | 0% | 0.459 | 0.68 |
| `noise=90` | 4 | 0 | 0% | 0.385 | 0.72 |
| `oracle_f=0.00` | 4 | 0 | 0% | 0.462 | 0.70 |
| `oracle_f=0.10` | 4 | 0 | 0% | 0.528 | 0.66 |
| `oracle_f=0.20` | 4 | 1 | 25% | 0.668 | 0.80 |
| `oracle_f=0.30` | 4 | 4 | 100% | 0.873 | 0.67 |
| `oracle_f=0.40` | 4 | 4 | 100% | 0.918 | 0.72 |
| `oracle_f=0.50` | 4 | 4 | 100% | 0.935 | 0.72 |

## Interpretation

- **Oracle fraction curve**: mapCC / solve rate vs fraction of strong |E| phases known exactly. This is the theoretical ceiling for a perfect prior on that seed set.
- **Noise curve**: robustness of a fixed 30% seed to phase error (°).
- **Fragment curve**: random true-atom subsets as MR-lite seeds.
- **Baselines**: CF and GraphPhaseNet prior without partial φ.

If oracle ≥30–40% strong phases **solves** hard cells while full priors do not, the bottleneck is **seed quality**, not the extension engine.

API: `solvers/partial_seed.py` — `oracle_partial_phaseed_solve`, `fragment_phaseed_solve`, `load_phase_seed_csv`.

JSON companion: `partial_seed_benchmark.json`
