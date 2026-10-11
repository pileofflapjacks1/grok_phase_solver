# Q-peak benchmark (`trial.res` vs deposited COD atoms)

Written by `scripts/run_qpeak_benchmark.py`. Do not edit numbers by hand.

Inputs: COD CIF + Fobs HKL (CC0) in `data/raw/cod_qpeak/`. Each set runs
`gps-solve --hkl <id>.hkl --cell … --sg … --out …` (default `--method auto`), then the
scorer in `grok_phase_solver.metrics.qpeak_benchmark` reads `trial.res` + `peaks.csv`.
Full run: `python scripts/run_qpeak_benchmark.py` (~2.5 min CPU). Fast CI check:
`pytest tests/test_qpeak_benchmark.py` (planted atoms + 20-draw baseline).

**Bar (pre-registered, not tuned):** N = deposited non-H atoms; take the N strongest Q
peaks; best alignment over origin shifts {0,½}³ × {identity, inversion} with full
space-group orbits. **PASS:** ≥ 50% of atoms have a Q within 0.5 Å, that rate is
≥ 3× the random baseline (mean of 1000 uniform point sets, seed 12345, same
alignment search), and ghosts (Q > 1.0 Å from every atom image) ≤ 40%.
**PARTIAL:** 25–50% within 0.5 Å and > 3× baseline. **FAIL:** anything else.

| COD | SG | N non-H | ≤0.5 Å | ≤1.0 Å | ghosts | random ≤0.5 mean (sd, max) | random ghost | rate / baseline | P(random ≥ real) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 2200239 | P 1 21/c 1 | 22 | 3/22 (14%) | 5/22 (23%) | 16/22 (73%) | 1.85 (0.75, 5) | 73% | 1.62× | 0.173 | **FAIL** |
| 2200001 | P 21 21 21 | 34 | 3/34 (9%) | 9/34 (26%) | 24/34 (71%) | 2.73 (0.84, 7) | 76% | 1.10× | 0.583 | **FAIL** |
| 2230859 | P 1 21/n 1 | 17 | 0/17 (0%) | 2/17 (12%) | 15/17 (88%) | 1.45 (0.76, 4) | 73% | 0.00× | 1.000 | **FAIL** |

## Planted-atom self-test

Deposited atoms planted as Q peaks after a random symmetry image per atom, inversion
and an origin shift. The scorer must recover N/N within 0.5 Å; jitter adds Gaussian
noise per axis.

| COD | N | exact ≤0.5 Å | jitter σ | jitter ≤0.5 Å |
|---|---|---|---|---|
| 2200239 | 22 | 22/22 | 0.15 Å | 21/22 |
| 2200001 | 34 | 34/34 | 0.15 Å | 33/34 |
| 2230859 | 17 | 17/17 | 0.15 Å | 17/17 |

Caveats: Q peaks sit on the map grid (~0.3–0.4 Å steps), which limits 0.5 Å hits even
for a correct map. `trial.res` is `SFAC C` only, so element types are not scored.
A random point set already has ~70–75% ghosts at 1.0 Å, so the 40% ghost bound is strict.
