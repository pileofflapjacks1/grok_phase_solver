# Experimental HKL scoreboard

Methods on **experimental-style** data plus COD Fcalc control.
Strict / mapCC success uses deposited structure Fcalc truth when CIF is available.

## Results

| Dataset | Method | mapCC | free FOM | peaks | solved | s |
|---------|--------|-------|----------|-------|--------|---|
| demo_solve | `charge_flipping` | nan | 0.684 | 15 | None | 5.0 |
| demo_solve | `ensemble` | nan | 0.776 | 14 | None | 16.6 |
| demo_solve | `phai+cf_cond` | nan | 0.774 | 13 | None | 40.5 |
| demo_solve | `phai_phaseed` | nan | 0.771 | 25 | None | 28.8 |
| demo_solve | `phai+cf_cond` | nan | 0.774 | 13 | None | 22.2 |
| COD_2016452_Fcalc_0.9 | `charge_flipping` | 0.168 | 0.627 | 25 | False | 9.2 |
| COD_2016452_Fcalc_0.9 | `ensemble` | 0.475 | 0.734 | 25 | False | 42.1 |
| COD_2016452_Fcalc_0.9 | `phai+cf_cond` | 0.478 | 0.722 | 25 | False | 33.8 |
| COD_2016452_Fcalc_0.9 | `phai_phaseed` | 0.434 | 0.778 | 25 | False | 51.0 |
| COD_2016452_Fcalc_0.9 | `phai_phaseed` | 0.434 | 0.778 | 25 | False | 56.5 |
| COD_2016452_Fcalc_1.5 | `charge_flipping` | 0.311 | 0.722 | 14 | False | 2.5 |
| COD_2016452_Fcalc_1.5 | `ensemble` | 0.324 | 0.745 | 13 | False | 11.2 |
| COD_2016452_Fcalc_1.5 | `phai+cf_cond` | 0.221 | 0.741 | 18 | False | 26.0 |
| COD_2016452_Fcalc_1.5 | `phai_phaseed` | 0.244 | 0.801 | 25 | False | 31.5 |
| COD_2016452_Fcalc_1.5 | `phai_phaseed` | 0.244 | 0.801 | 25 | False | 34.1 |
| COD_2017775_exp_1.2 | `charge_flipping` | 0.193 | 0.737 | 25 | False | 12.3 |
| COD_2017775_exp_1.2 | `ensemble` | 0.193 | 0.737 | 25 | False | 31.2 |
| COD_2017775_exp_1.2 | `charge_flipping` | 0.193 | 0.737 | 25 | False | 8.6 |

## Notes

- **demo_solve**: small synthetic-style demo with INS.
- **COD 2016452 Fcalc**: PhAI/AI-PhaSeed control (should show strong hybrids).
- **COD 2017775**: large experimental Fobs (roxithromycin); ab initio expected to struggle — free FOM still ranks without truth; resolution capped for runtime.
- `auto` selects phai_phaseed / ensemble / CF by SG, resolution, and weights.
- `trial.res` is written by `gps-solve` for Olex2/SHELXL loading.
- Dedicated `run_cod_hybrid_benchmark.py` / `run_fair_phai_benchmark.py` use longer
  iterations and report higher COD 2016452 mapCC (PhAI+CF solve @0.9 Å under
  full settings). This scoreboard stresses **pipeline methods + experimental HKL**
  with moderate compute budgets.

JSON: `data/processed/experimental_scoreboard.json`
