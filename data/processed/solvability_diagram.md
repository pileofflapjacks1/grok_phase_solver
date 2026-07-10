# Solvability phase diagram

Synthetic P1 organics. **Strict success:**

- mapCC_OI ≥ **0.70**
- peak recovery ≥ **0.50** (atoms matched to density peaks, origin-shifted)
- R1 ≤ **0.45** (carbons at top peaks vs |F_obs|)

Trials: 192 (including seeds). JSON: `data/processed/solvability_diagram.json`

## Success rate by method (all conditions)

| Method | Solved | Total | Rate |
|--------|--------|-------|------|
| `charge_flipping` | 18 | 64 | 28% |
| `phase_recycle` | 2 | 64 | 3% |
| `direct_methods` | 1 | 64 | 2% |

## Success rate: charge_flipping vs (n_atoms, d_min) at completeness=1.0

| n_atoms \ d_min | 0.9 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|
| **4** | 50% (CC=0.85) | 50% (CC=0.85) | 50% (CC=0.83) | 50% (CC=0.67) |
| **8** | 50% (CC=0.78) | 0% (CC=0.56) | 0% (CC=0.49) | 0% (CC=0.45) |
| **12** | 100% (CC=0.89) | 50% (CC=0.68) | 0% (CC=0.51) | 0% (CC=0.47) |
| **20** | 0% (CC=0.49) | 0% (CC=0.44) | 0% (CC=0.39) | 0% (CC=0.44) |

## Interpretation

- **High resolution + few atoms:** classical CF should dominate (atomic peak separation).
- **Low resolution or many atoms:** success rate collapses — the open phase problem.
- **Completeness 0.7:** typically harder than full data at same d_min.
- Direct methods here are a **thin educational** multi-start tangent code, not SHELXD.
- Phase recycle (ER positivity) is fast but weaker than CF on atomic-resolution synthetics.

This diagram defines the **baseline frontier** any new method (PhAI, hybrids, new math)
must beat under the **same** success criterion.
