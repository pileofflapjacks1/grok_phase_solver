# Hard-path validation (v0.7.0)

Packaged `examples/partial_seed_demo` with truth metrics (mapCC_OI vs deposited Fcalc phases).

| Run | Method | free FOM | mapCC_OI | peaks | seed size bar |
|-----|--------|----------|----------|-------|---------------|
| `auto` | `strong_prior_phaseed` | 0.714 | **0.215** | 13 | None |
| `partial_15` | `partial_phaseed` | 0.813 | **0.374** | 13 | True |
| `partial_30` | `partial_phaseed` | 0.833 | **0.830** | 13 | True |
| `fragment_res` | `partial_phaseed` | 0.809 | **0.583** | 10 | True |
| `peaks_from_auto` | `partial_phaseed` | 0.797 | **0.286** | 14 | True |

## Takeaways

- **partial_30** (oracle ~30% strong |E|) should beat **auto** on mapCC / free FOM.
- **partial_15** is often below the practical seed bar — weaker maps expected.
- **fragment.res** Fcalc seed is the scientist-facing hard path without oracle phases.
- **peaks_from_auto** is a weak recovery path when ab initio peaks look atomic.

Oracle bar remains: ≥~30% strong |E| phases within ~20° of truth for reliable hard solves.
