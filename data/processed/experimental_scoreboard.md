# Experimental HKL scoreboard (Lane C)

Methods on **experimental Fobs** (COD) plus **Fcalc controls** from deposited CIFs.
Strict / mapCC success uses Fcalc truth phases matched by Miller index when CIF is available.

- SHELXS binary: **found**
- GraphPhaseNet strong prior: **found**
- SHELXD binary: **not in ShelX/** (dual-space in-repo only; see `shelxd_h2h.md`)

**v0.4.0 companion:** stratified AI-PhaSeed / DM+AI hybrid subset harness →
[`ai_phaseed_extended_benchmark.md`](ai_phaseed_extended_benchmark.md)
(Carrozzini 2025 alignment; not a full 1505-COD replication).

## Results

| Dataset | Method | mapCC | free FOM | peaks | solved | s |
|---------|--------|-------|----------|-------|--------|---|
| demo_solve | `charge_flipping` | nan | 0.684 | 15 | None | 0.4 |
| demo_solve | `ensemble` | nan | 0.776 | 14 | None | 1.6 |
| demo_solve | `ensemble` | nan | 0.776 | 14 | None | 2.8 |
| demo_solve | `strong_prior_phaseed` | nan | 0.796 | 25 | None | 1.6 |
| demo_solve | `shelxs` | nan | 0.788 | 25 | None | 0.2 |
| COD_2016452_Fcalc_0.9 | `charge_flipping` | 0.168 | 0.627 | 25 | False | 0.9 |
| COD_2016452_Fcalc_0.9 | `ensemble` | 0.475 | 0.734 | 25 | False | 3.6 |
| COD_2016452_Fcalc_0.9 | `phai+cf_cond` | 0.478 | 0.722 | 25 | False | 6.8 |
| COD_2016452_Fcalc_0.9 | `phai_phaseed` | 0.434 | 0.778 | 25 | False | 4.4 |
| COD_2016452_Fcalc_0.9 | `ensemble` | 0.475 | 0.734 | 25 | False | 3.7 |
| COD_2016452_Fcalc_0.9 | `strong_prior_phaseed` | 0.144 | 0.733 | 25 | False | 4.5 |
| COD_2016452_Fcalc_0.9 | `shelxs` | 0.723 | 0.759 | 25 | False | 0.5 |
| COD_2016452_Fcalc_0.9_partial30 | `partial_phaseed` | 0.793 | 0.750 | 25 | False | 4.7 |
| COD_2016452_Fcalc_1.2 | `charge_flipping` | 0.233 | 0.719 | 22 | False | 0.5 |
| COD_2016452_Fcalc_1.2 | `ensemble` | 0.336 | 0.740 | 20 | False | 1.9 |
| COD_2016452_Fcalc_1.2 | `phai+cf_cond` | 0.285 | 0.759 | 25 | False | 2.3 |
| COD_2016452_Fcalc_1.2 | `phai_phaseed` | 0.237 | 0.770 | 25 | False | 3.2 |
| COD_2016452_Fcalc_1.2 | `phai_phaseed` | 0.237 | 0.770 | 25 | False | 3.1 |
| COD_2016452_Fcalc_1.2 | `strong_prior_phaseed` | 0.195 | 0.782 | 25 | False | 1.8 |
| COD_2016452_Fcalc_1.2 | `shelxs` | 0.263 | 0.785 | 25 | False | 0.2 |
| COD_2016452_Fcalc_1.2_partial30 | `partial_phaseed` | 0.755 | 0.810 | 25 | False | 1.3 |
| COD_2016452_exp_1.0 | `charge_flipping` | 0.278 | 0.648 | 25 | False | 0.3 |
| COD_2016452_exp_1.0 | `ensemble` | 0.247 | 0.678 | 25 | False | 1.0 |
| COD_2016452_exp_1.0 | `phai+cf_cond` | 0.995 | 0.782 | 25 | True | 2.8 |
| COD_2016452_exp_1.0 | `phai_phaseed` | 0.949 | 0.780 | 25 | True | 2.7 |
| COD_2016452_exp_1.0 | `ensemble` | 0.247 | 0.678 | 25 | False | 1.1 |
| COD_2016452_exp_1.0 | `strong_prior_phaseed` | 0.310 | 0.697 | 25 | False | 1.2 |
| COD_2016452_exp_1.0 | `shelxs` | 0.557 | 0.731 | 25 | False | 0.1 |
| COD_2100301_Fcalc_0.9 | `charge_flipping` | 0.284 | 0.703 | 25 | False | 1.1 |
| COD_2100301_Fcalc_0.9 | `ensemble` | 0.256 | 0.649 | 25 | False | 4.5 |
| COD_2100301_Fcalc_0.9 | `phai+cf_cond` | 0.111 | 0.661 | 25 | False | 3.1 |
| COD_2100301_Fcalc_0.9 | `phai_phaseed` | 0.118 | 0.743 | 25 | False | 5.0 |
| COD_2100301_Fcalc_0.9 | `ensemble` | 0.256 | 0.649 | 25 | False | 4.6 |
| COD_2100301_Fcalc_0.9 | `strong_prior_phaseed` | 0.176 | 0.616 | 25 | False | 4.5 |
| COD_2100301_Fcalc_0.9 | `shelxs` | 0.357 | 0.766 | 25 | False | 0.4 |
| COD_2100301_Fcalc_0.9_partial30 | `partial_phaseed` | 0.758 | 0.787 | 25 | False | 3.1 |
| COD_2100301_Fcalc_1.2 | `charge_flipping` | 0.232 | 0.748 | 16 | False | 0.5 |
| COD_2100301_Fcalc_1.2 | `ensemble` | 0.285 | 0.715 | 20 | False | 2.0 |
| COD_2100301_Fcalc_1.2 | `phai+cf_cond` | 0.248 | 0.703 | 18 | False | 2.4 |
| COD_2100301_Fcalc_1.2 | `phai_phaseed` | 0.166 | 0.729 | 25 | False | 3.3 |
| COD_2100301_Fcalc_1.2 | `phai_phaseed` | 0.166 | 0.729 | 25 | False | 3.3 |
| COD_2100301_Fcalc_1.2 | `strong_prior_phaseed` | 0.187 | 0.718 | 25 | False | 2.0 |
| COD_2100301_Fcalc_1.2 | `shelxs` | 0.277 | 0.798 | 25 | False | 0.3 |
| COD_2100301_Fcalc_1.2_partial30 | `partial_phaseed` | 0.720 | 0.770 | 25 | False | 1.4 |
| COD_2100301_exp_1.0 | `charge_flipping` | 0.228 | 0.686 | 25 | False | 0.3 |
| COD_2100301_exp_1.0 | `ensemble` | 0.280 | 0.730 | 25 | False | 1.2 |
| COD_2100301_exp_1.0 | `phai+cf_cond` | 0.496 | 0.683 | 25 | False | 2.2 |
| COD_2100301_exp_1.0 | `phai_phaseed` | 0.504 | 0.708 | 25 | False | 2.7 |
| COD_2100301_exp_1.0 | `ensemble` | 0.280 | 0.730 | 25 | False | 1.1 |
| COD_2100301_exp_1.0 | `strong_prior_phaseed` | 0.328 | 0.731 | 25 | False | 1.3 |
| COD_2100301_exp_1.0 | `shelxs` | 0.528 | 0.756 | 25 | False | 0.1 |
| COD_2017775_exp_1.2 | `charge_flipping` | 0.193 | 0.737 | 25 | False | 0.8 |
| COD_2017775_exp_1.2 | `ensemble` | 0.193 | 0.737 | 25 | False | 2.8 |
| COD_2017775_exp_1.2 | `phai+cf_cond` | 0.153 | 0.618 | 25 | False | 2.9 |
| COD_2017775_exp_1.2 | `strong_prior_phaseed` | 0.164 | 0.727 | 25 | False | 2.3 |

## Best mapCC per dataset (truth-matched)

| Dataset | Best method | mapCC | solved |
|---------|-------------|-------|--------|
| demo_solve | — | n/a (no truth mapCC) | — |
| COD_2016452_Fcalc_0.9 | `shelxs` | 0.723 | False |
| COD_2016452_Fcalc_0.9_partial30 | `partial_phaseed` | 0.793 | False |
| COD_2016452_Fcalc_1.2 | `ensemble` | 0.336 | False |
| COD_2016452_Fcalc_1.2_partial30 | `partial_phaseed` | 0.755 | False |
| COD_2016452_exp_1.0 | `phai+cf_cond` | 0.995 | True |
| COD_2100301_Fcalc_0.9 | `shelxs` | 0.357 | False |
| COD_2100301_Fcalc_0.9_partial30 | `partial_phaseed` | 0.758 | False |
| COD_2100301_Fcalc_1.2 | `ensemble` | 0.285 | False |
| COD_2100301_Fcalc_1.2_partial30 | `partial_phaseed` | 0.720 | False |
| COD_2100301_exp_1.0 | `shelxs` | 0.528 | False |
| COD_2017775_exp_1.2 | `charge_flipping` | 0.193 | False |

## Headline takeaways

| Result | Detail |
|--------|--------|
| **PhAI on COD 2016452 experimental Fobs** | `phai+cf_cond` mapCC **0.995**, strict **solved**; `phai_phaseed` 0.949 solved |
| **SHELXS** on same exp Fobs | mapCC 0.56 (below PhAI hybrids here) |
| **Oracle partial-φ 30%** on Fcalc | mapCC **0.72–0.79** (extension works; strict may still fail R1/peaks under short settings) |
| **COD 2100301 exp** | Best ~0.53 (SHELXS); no strict solve in this budget |
| **COD 2017775 exp** | mapCC ~0.19 — large cell, ab initio fails as expected |
| **Graph prior** | Not competitive with PhAI/SHELXS on these experimental organics |

## Notes

- **demo_solve**: packaged easy demo (INS); free FOM ranks without truth.
- **COD 2016452**: small P2₁/c organic; Fcalc control + **experimental Fobs** from COD.
- **COD 2100301**: dinicotinic acid P2₁/c (neutron structure); Fcalc + **experimental Fobs**.
- **`*_partial30`**: oracle 30% strong-|E| phases → `partial_phaseed` (Lane B hard path).
- **COD 2017775**: large experimental Fobs (roxithromycin); ab initio expected to struggle.
- Experimental Fobs mapCC uses Fcalc from deposited model as proxy truth (not refined R1).
- Longer PhAI settings: see `cod_hybrid_benchmark.md` (C8 strict Fcalc solve @ 0.9 Å).
- `auto` selects ensemble / PhAI / prior / CF by SG, resolution, and available weights.
- Industrial SHELXD not redistributed; local SHELXS used when present.

Related scoreboards: `cod_hybrid_benchmark.md`, `shelxs_h2h.md`, `strong_prior.md`, `partial_seed_benchmark.md`.

JSON: `data/processed/experimental_scoreboard.json`
