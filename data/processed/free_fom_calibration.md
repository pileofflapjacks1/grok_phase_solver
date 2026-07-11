# Free-FOM calibration (v2)

## Math fix

Old `R_after_ER` was computed **after** modulus projection → always ≈ 0 (vacuous). New **R₊** = R-factor of `|FFT(max(ρ,0))|` vs `|F_obs|` (positivity residual) — informative and truth-free.

Composite combines scored R₊, excess kurtosis, peakiness (max/σ + top mass), skew, weak positivity fraction, plus light shell-R₊ and Sayre terms.

## Ranking vs truth mapCC

| Metric | Value |
|--------|-------|
| Spearman ρ(composite, mapCC_OI) | **0.895** |
| Pairwise rank accuracy | **89.3%** (n=698) |
| P(true FOM > random FOM) | **100.0%** |
| FOM inversion rate (wrong beats true) | **0.0%** (0/8) |
| mean (C_true − C_cf) | **0.107** |
| free-FOM version | 2.1 |

## Conditional polish gate (synthetic)

Accept only if composite↑ and R₊ does not regress badly.

| TP | FP | TN | FN | precision | FP rate |
|----|----|----|----|-----------|---------|
| 5 | 0 | 11 | 0 | 1.00 | 0.00 |

## COD 2016452 seed→polish gate

Rewrite trust-region: large \(D_\varphi\) requires \(\Delta R_+ \ge 0.08\).

| d_min | polish | seed | accept | mapCC seed→final | R₊ seed→final | disp | good gate |
|-------|--------|------|--------|------------------|---------------|------|-----------|
| 0.9 | charge_flipping | phai_fair | True | 0.558→0.816 | 0.301→0.175 | 0.641 | True |
| 0.9 | raar | phai_fair | False | 0.558→0.558 | 0.301→0.317 | 0.980 | True |
| 1.2 | charge_flipping | phai_fair | False | 0.607→0.607 | 0.294→0.235 | 0.738 | True |
| 1.2 | raar | phai_fair | False | 0.607→0.607 | 0.294→0.311 | 1.005 | True |
| 1.5 | charge_flipping | phai_fair | False | 0.621→0.621 | 0.288→0.276 | 0.826 | True |
| 1.5 | raar | phai_fair | False | 0.621→0.621 | 0.288→0.288 | 0.893 | True |
| 2.0 | charge_flipping | phai_fair | False | 0.628→0.628 | 0.330→0.278 | 0.803 | True |
| 2.0 | raar | phai_fair | False | 0.628→0.628 | 0.330→0.262 | 1.160 | True |

COD gate correctness: **8/8** decisions match mapCC interest.

## Mean composite / mapCC by phase-set type

| Label | mean composite | mean mapCC | n |
|-------|----------------|------------|---|
| `true` | 0.810 | 1.000 | 8 |
| `partial` | 0.788 | 0.796 | 8 |
| `cf` | 0.703 | 0.653 | 8 |
| `raar` | 0.675 | 0.362 | 8 |
| `random` | 0.654 | 0.277 | 8 |

## Interpretation

- Higher Spearman / pairwise accuracy ⇒ free FOM tracks solution quality.
- Low false-positive gate rate ⇒ fewer harmful CF polishes accepted.
- Free FOM remains a **proxy**, not an oracle; always refine experimentally.

JSON: `data/processed/free_fom_calibration.json`
Runtime: 46.5s
