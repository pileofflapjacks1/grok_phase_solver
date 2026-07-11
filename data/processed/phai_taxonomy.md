# PhAI-seeded failure taxonomy (hard region)

Compare **classical random multistart** vs **PhAI fair seed + multistart** under free-FOM v2.1 (anti-false-atomicity).

PhAI available: **True**
Improved cases (mapCC +0.05 or label upgrade to solved/near): **3/12**

## Label counts

### Classical (random init)

| Label | Count | Rate |
|-------|-------|------|
| `near` | 1 | 8% |
| `A` | 2 | 17% |
| `B` | 1 | 8% |
| `B+C` | 8 | 67% |

mean best mapCC = 0.513
FOM inversion rate = 0%

### PhAI-seeded

| Label | Count | Rate |
|-------|-------|------|
| `solved` | 1 | 8% |
| `near` | 1 | 8% |
| `A` | 1 | 8% |
| `B+C` | 9 | 75% |

mean best mapCC = 0.530
FOM inversion rate = 0%

## Transitions (classical → PhAI)

| Transition | Count |
|------------|-------|
| `B+C→B+C` | 8 |
| `A→near` | 1 |
| `near→solved` | 1 |
| `B→A` | 1 |
| `A→B+C` | 1 |

## Per-case

| n | d_min | seed | classical | bestCC | phai | bestCC | seedCC | improved |
|---|-------|------|-----------|--------|------|--------|--------|----------|
| 12 | 1.5 | 0 | **B+C** | 0.52 | **B+C** | 0.51 | 0.29 | False |
| 12 | 1.5 | 1 | **A** | 0.64 | **near** | 0.69 | 0.35 | True |
| 12 | 1.5 | 2 | **near** | 0.65 | **solved** | 0.79 | 0.22 | True |
| 12 | 2.0 | 0 | **B+C** | 0.47 | **B+C** | 0.51 | 0.41 | False |
| 12 | 2.0 | 1 | **B** | 0.48 | **A** | 0.56 | 0.46 | True |
| 12 | 2.0 | 2 | **B+C** | 0.49 | **B+C** | 0.46 | 0.35 | False |
| 16 | 1.5 | 0 | **A** | 0.56 | **B+C** | 0.53 | 0.33 | False |
| 16 | 1.5 | 1 | **B+C** | 0.52 | **B+C** | 0.52 | 0.32 | False |
| 16 | 1.5 | 2 | **B+C** | 0.53 | **B+C** | 0.52 | 0.25 | False |
| 20 | 1.5 | 0 | **B+C** | 0.44 | **B+C** | 0.45 | 0.24 | False |
| 20 | 1.5 | 1 | **B+C** | 0.38 | **B+C** | 0.39 | 0.30 | False |
| 20 | 1.5 | 2 | **B+C** | 0.49 | **B+C** | 0.44 | 0.29 | False |

## Interpretation

- If PhAI shifts **A+B / B+C → solved/near**, the hard cliff is mainly a **basin/prior** problem that neural seeds help.
- If labels stay **A+B** with higher mapCC but free-FOM inversion remains, need both priors and AFA free FOM.
- If PhAI seed mapCC is high but polish destroys it, conditional gate matters (see free-FOM rewrite trust-region).

JSON: `data/processed/phai_taxonomy.json`
Runtime: 99.5s
