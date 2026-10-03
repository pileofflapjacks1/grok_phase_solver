# R1 gate diagnostic — mid-band Fobs fragment_half

Strict success is unchanged: mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5 **and** carbon-peak R1 ≤ 0.45. Official columns are copied from `cod_stratified_bench.json` (d_min = 1.0 Å). `solved` is not recomputed.

Carbon-peak R1 places the strongest peaks as carbon with B = 5. The side column uses those same peaks and assigns a deposited non-H element when the peak matches a true site within the peak-recovery tolerance (0.15, fractional min-image) after that search's origin shift. Unmatched peaks stay carbon. This side R1 is not a strict-bar input.

| COD | mapCC | peak | R1 carbon | mapCC≥0.7 | peak≥0.5 | R1≤0.45 | solved | R1 typed | matched |
|-----|-------|------|-----------|-----------|----------|---------|--------|----------|---------|
| 2012000 | **0.675** | 0.833 | 0.525 | no | yes | no | False | **0.520** | 26/54 |
| 2013000 | **0.751** | 0.983 | 0.547 | yes | yes | no | False | **0.481** | 51/59 |
| 1543089 | **0.744** | 1.000 | 0.593 | yes | yes | no | False | **0.595** | 78/104 |
| 1543227 | **0.762** | 0.990 | 0.473 | yes | yes | no | False | **0.477** | 70/104 |
| 1544230 | **0.668** | 1.000 | 0.562 | no | yes | no | False | **0.552** | 73/104 |
| 1544651 | **0.737** | 1.000 | 0.567 | yes | yes | no | False | **0.559** | 79/104 |
| 1549607 | **0.663** | 1.000 | 0.542 | no | yes | no | False | **0.551** | 61/104 |
| 1550274 | **0.685** | 1.000 | 0.611 | no | yes | no | False | **0.610** | 62/104 |
| 2016430 | **0.708** | 1.000 | 0.564 | yes | yes | no | False | **0.563** | 159/208 |
| 2221836 | **0.717** | 1.000 | 0.621 | yes | yes | no | False | **0.619** | 164/208 |
| 2227862 | **0.765** | 1.000 | 0.558 | yes | yes | no | False | **0.539** | 156/208 |
| 2233297 | **0.763** | 1.000 | 0.588 | yes | yes | no | False | **0.571** | 178/208 |

## Gate counts (scoreboard, n = 12)

- mapCC ≥ 0.7: **8**
- peak recovery ≥ 0.5: **12**
- R1 ≤ 0.45: **0**
- strict solved: **0**

## Side R1 (typed peaks)

Typed-peak R1 was computed for 12 of 12 rows (mean 0.553, minimum 0.477). 0 of those side values are ≤ 0.45. Unmatched peaks stay carbon, so the side residual stays close to the frozen carbon-peak R1. That count does not change `solved`. Scoreboard carbon-peak R1 (n=12) has mean 0.563 and minimum 0.473.

## COD 1544230

The bench and `solve_structure` now share `d_min_keep_mask` (d ≥ d_min − 1e-9), so reflection (1 9 6) at d=0.999999999694 Å with |F|=0 stays in both lists (1968 amplitudes and 1968 phases). Fobs fragment_half mapCC moved from 0.669 to 0.668. Scoreboard carbon-peak R1 is 0.562 and peak recovery is 1.000. The row is not a strict solve. The typed R1 0.552 in this table is the earlier side measurement on the 1967 reflections the solver used to drop. It is not the scoreboard R1 and it does not set solved.

Regenerate:

```bash
python scripts/run_r1_gate_diagnostic.py
```
