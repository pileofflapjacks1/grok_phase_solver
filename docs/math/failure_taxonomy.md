# Solvability failure taxonomy

## Question

When classical multistart phase retrieval **fails** under strict success criteria, *what kind of failure is it?*

Different answers demand different research investments:

| If the failure is… | Invest in… |
|--------------------|------------|
| **Selection** | Free FOMs, ensemble ranking, hybrid gates |
| **Basin / optimization** | Multistart, better seeds (DM, PhAI), projection schedules |
| **Information** | Higher resolution, more completeness, atomic/neural priors |

## Setup

For a synthetic structure with known phases \(\varphi^\star\):

1. Compute information proxies from \(|F|\) only:
   - reflections per atom \(N_{\mathrm{refl}}/N_{\mathrm{atoms}}\)
   - mean Cochran \(\kappa\) over strong triplets
2. Build free FOMs of \(\varphi^\star\) and of random phases.
3. Run multistart CF + RAAR (\(n_{\mathrm{starts}}\) seeds each).
4. For each trial \(t\): origin-invariant mapCC vs truth, free-FOM composite \(C_t\).

## Labels

### Solved
\[
\max_t \mathrm{mapCC}(t) \ge 0.7
\]
(map part of strict success; peak/R1 checked elsewhere).

### A — Selection failure

Multistart **found** a useful basin, but free FOM did not select it, or free FOM is inverted vs truth:

- \(\max_t \mathrm{mapCC}(t) \ge 0.55\) **and** free-FOM pick has mapCC lower by \(\ge 0.05\), **or**
- some wrong trial has \(C_t > C(\varphi^\star) + \varepsilon\) with mapCC \(< 0.45\) (FOM inversion).

**Interpretation:** the density landscape was partially explored; **ranking** is the bottleneck.

### B — Basin / optimization failure

No trial reaches a good basin, yet free FOM *would* prefer truth if found:

- \(\max_t \mathrm{mapCC}(t) < 0.55\)
- \(C(\varphi^\star)\) clearly above random (\(+\ge 0.04\))
- free-FOM of best trial still well below \(C(\varphi^\star)\)

**Interpretation:** non-convex search never entered the correct basin. More multistart, better initialization (tangent formula, PhAI), or different projectors — **not** free-FOM tuning alone.

### C — Information / underdetermination

Observables do not cleanly identify the true density among modulus-consistent maps:

- reflections/atom \(< 8\), **or**
- mean \(\kappa\) low / few triplets, **or**
- \(C(\varphi^\star) \lesssim C(\mathrm{random})\), **or**
- degeneracy: several wrong maps with **similar** free FOM and all low mapCC

**Interpretation:** the phase problem is **underdetermined** at this resolution/size for positivity+atomicity alone. Need more data or a stronger prior (fragment library, PhAI, heavy atoms).

### Near-solved

\[
0.55 \le \max_t \mathrm{mapCC}(t) < 0.7
\]

A useful basin was entered but not refined to strict success. Implies more iterations, density modification, or a mild prior — not necessarily more data.

### Multi-labels

`A+B`, `B+C` when multiple mechanisms fire. Priority for primary label: selection first when a good basin was found, else basin vs information.

## Relation to free-FOM v2

Free-FOM calibration asks: *does \(C\) correlate with mapCC?*  
Taxonomy asks: *when \(C\) fails to deliver a solution, is that because of \(C\), search, or data?*

A hard region full of **C** and **B** means further FOM polishing has limited upside; priors and information dominate. Dominance of **A** would justify more free-FOM work.

## Code

- `metrics/failure_taxonomy.py` — `diagnose_structure`, `classify_failure`
- `scripts/run_failure_taxonomy.py` → `data/processed/failure_taxonomy.md`

## Caveats

- Thresholds (`mapCC_good=0.55`, `refl/atom`, \(\kappa\)) are project conventions.
- mapCC ≥ 0.7 is necessary but not full strict success (peaks, R1).
- Synthetic P1 equal-ish organics; experimental noise and SG symmetry change rates.
- Free FOM remains a proxy; “FOM inversion” uses that proxy by construction.
