# Solvability failure taxonomy

Classification of multistart CF+RAAR outcomes under free-FOM ranking.

## Labels

| Code | Meaning | Actionable implication |
|------|---------|------------------------|
| **solved** | Best trial mapCC ≥ 0.7 | Classical path works |
| **near** | mapCC in [0.55, 0.7) | More polish / iters, not more data alone |
| **A** | Selection failure | Improve free FOM / selection |
| **B** | Basin / optimization failure | Better search, multistart, or priors |
| **C** | Information / underdetermination | Need more data or stronger atomic prior |
| **A+B** / **B+C** | Multi-factor | Address both |

## Overall counts

| Label | Count | Rate |
|-------|-------|------|
| `solved` | 8 | 38% |
| `near` | 2 | 10% |
| `A` | 2 | 10% |
| `B` | 1 | 5% |
| `B+C` | 8 | 38% |

## Easy region

n = 9, mean best mapCC = 0.893, mean refl/atom = 220.2

| Label | Count | Rate |
|-------|-------|------|
| `solved` | 8 | 89% |
| `near` | 1 | 11% |

## Hard region

n = 12, mean best mapCC = 0.513, mean refl/atom = 20.3

| Label | Count | Rate |
|-------|-------|------|
| `near` | 1 | 8% |
| `A` | 2 | 17% |
| `B` | 1 | 8% |
| `B+C` | 8 | 67% |

## Per-case table

| region | n | d_min | seed | primary | bestCC | fomPickCC | rpa | κ | C_true | C_rand |
|--------|---|-------|------|---------|--------|-----------|-----|---|--------|--------|
| easy | 4 | 0.9 | 0 | **solved** | 0.86 | 0.85 | 216.0 | 3.80 | 0.75 | 0.65 |
| easy | 4 | 0.9 | 1 | **solved** | 0.94 | 0.94 | 353.0 | 4.61 | 0.77 | 0.62 |
| easy | 4 | 0.9 | 2 | **solved** | 0.91 | 0.91 | 437.0 | 5.90 | 0.76 | 0.63 |
| easy | 6 | 1.0 | 0 | **solved** | 0.94 | 0.94 | 100.7 | 2.62 | 0.77 | 0.65 |
| easy | 6 | 1.0 | 1 | **solved** | 0.92 | 0.87 | 166.3 | 3.55 | 0.78 | 0.66 |
| easy | 6 | 1.0 | 2 | **solved** | 0.94 | 0.94 | 205.7 | 4.10 | 0.77 | 0.64 |
| easy | 8 | 0.9 | 0 | **near** | 0.61 | 0.59 | 108.5 | 2.94 | 0.77 | 0.65 |
| easy | 8 | 0.9 | 1 | **solved** | 0.95 | 0.89 | 176.5 | 3.48 | 0.77 | 0.62 |
| easy | 8 | 0.9 | 2 | **solved** | 0.97 | 0.97 | 218.5 | 4.31 | 0.77 | 0.63 |
| hard | 12 | 1.5 | 0 | **B+C** | 0.52 | 0.31 | 23.5 | 1.26 | 0.85 | 0.66 |
| hard | 12 | 1.5 | 1 | **A** | 0.64 | 0.36 | 24.8 | 1.74 | 0.86 | 0.67 |
| hard | 12 | 1.5 | 2 | **near** | 0.65 | 0.65 | 30.7 | 1.66 | 0.85 | 0.66 |
| hard | 12 | 2.0 | 0 | **B+C** | 0.47 | 0.44 | 9.8 | 1.29 | 0.76 | 0.66 |
| hard | 12 | 2.0 | 1 | **B** | 0.48 | 0.48 | 9.8 | 1.52 | 0.77 | 0.64 |
| hard | 12 | 2.0 | 2 | **B+C** | 0.49 | 0.45 | 12.3 | 1.42 | 0.79 | 0.65 |
| hard | 16 | 1.5 | 0 | **A** | 0.56 | 0.32 | 23.4 | 1.28 | 0.86 | 0.66 |
| hard | 16 | 1.5 | 1 | **B+C** | 0.52 | 0.49 | 21.9 | 1.35 | 0.86 | 0.66 |
| hard | 16 | 1.5 | 2 | **B+C** | 0.53 | 0.37 | 23.0 | 1.32 | 0.85 | 0.66 |
| hard | 20 | 1.5 | 0 | **B+C** | 0.44 | 0.31 | 23.5 | 1.06 | 0.86 | 0.66 |
| hard | 20 | 1.5 | 1 | **B+C** | 0.38 | 0.38 | 21.5 | 1.10 | 0.85 | 0.67 |
| hard | 20 | 1.5 | 2 | **B+C** | 0.49 | 0.29 | 19.9 | 1.21 | 0.85 | 0.66 |

## Scientific implications

- **Selection (A-family):** 2 cases — free-FOM ranking is the bottleneck; continue FOM calibration / ensemble diversity metrics.
- **Basin (B-family):** 9 cases — need better initialization (DM κ, PhAI) or more aggressive multistart, not only better FOMs.
- **Information (C-family):** 8 cases — classical ab initio is underdetermined; neural/atomic priors or higher resolution are required.

Hard region dominated by **B** and/or **C** means the cliff is not mainly a free-FOM bug. Dominance of **A** would mean the solution is often found but discarded.

JSON: `data/processed/failure_taxonomy.json`
Runtime: 67.1s

Math write-up: [`docs/math/failure_taxonomy.md`](../../docs/math/failure_taxonomy.md)
