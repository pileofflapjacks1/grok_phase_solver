# COD experimental hard-path validation

Experimental **Fobs** with mapCC vs deposited-model Fcalc phases.
Oracle partial seeds use true phases on strong |E| only (Lane B control).
Fragment seed uses ~half of non-H atoms from the deposited CIF.

| Dataset | Run | Method | mapCC | free FOM | R1 | solved | s |
|---------|-----|--------|-------|----------|----|--------|---|
| `COD_2016452_exp` | `auto` | `ensemble` | **0.196** | 0.740 | 0.65 | False | 3.2 |
| `COD_2016452_exp` | `partial_15` | `partial_phaseed` | **0.485** | 0.701 | 0.65 | False | 2.6 |
| `COD_2016452_exp` | `partial_30` | `partial_phaseed` | **0.736** | 0.743 | 0.57 | False | 2.8 |
| `COD_2016452_exp` | `fragment_half` | `partial_phaseed` | **0.218** | 0.750 | 0.66 | False | 2.8 |
| `COD_2100301_exp` | `auto` | `ensemble` | **0.204** | 0.766 | 0.72 | False | 5.3 |
| `COD_2100301_exp` | `partial_15` | `partial_phaseed` | **0.467** | 0.710 | 0.70 | False | 4.8 |
| `COD_2100301_exp` | `partial_30` | `partial_phaseed` | **0.722** | 0.727 | 0.68 | False | 5.1 |
| `COD_2100301_exp` | `fragment_half` | `partial_phaseed` | **0.189** | 0.701 | 0.72 | False | 4.6 |

## Takeaways

- Compare **auto** vs **partial_30** on each COD set.
- **partial_15** often under-seeds vs the ~30% practical bar.
- **fragment_half** is the no-oracle scientist path (quality depends on fragment).
- Easy COD cases (e.g. 2016452) may already solve with PhAI hybrids; partial-φ is critical when ab initio fails.

See also: `examples/partial_seed_demo/HARD_PATH_VALIDATION.md` (synthetic).

