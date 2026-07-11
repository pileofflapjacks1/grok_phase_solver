# COD 2016452 conditional hybrid benchmark

Fcalc from deposited structure; **strict SuccessThresholds**. Conditional polish keeps PhAI seed unless free-FOM composite improves.

PhAI weights available: **True**

## d_min = 0.9 Å

| Method | mapCC | peak | R1 | solved | notes |
|--------|-------|------|----|--------|-------|
| `charge_flipping` | 0.350 | 0.94 | 0.63 | False |  |
| `raar` | 0.274 | 0.91 | 0.62 | False |  |
| `ensemble_cf_raar` | 0.580 | 1.00 | 0.47 | False | pick=cf |
| `phai_fair` | 0.558 | 1.00 | 0.61 | False |  |
| `phai+CF_uncond` | 0.867 | 1.00 | 0.34 | True |  |
| `phai+RAAR_uncond` | 0.252 | 0.84 | 0.64 | False |  |
| `phai+CF_cond` | 0.867 | 1.00 | 0.34 | True | accept=True |
| `phai+RAAR_cond` | 0.558 | 1.00 | 0.61 | False | accept=False |

## d_min = 1.2 Å

| Method | mapCC | peak | R1 | solved | notes |
|--------|-------|------|----|--------|-------|
| `charge_flipping` | 0.442 | 0.94 | 0.60 | False |  |
| `raar` | 0.372 | 0.94 | 0.53 | False |  |
| `ensemble_cf_raar` | 0.578 | 0.94 | 0.49 | False | pick=cf |
| `phai_fair` | 0.607 | 1.00 | 0.63 | False |  |
| `phai+CF_uncond` | 0.489 | 0.84 | 0.51 | False |  |
| `phai+RAAR_uncond` | 0.365 | 0.91 | 0.59 | False |  |
| `phai+CF_cond` | 0.489 | 0.84 | 0.51 | False | accept=True |
| `phai+RAAR_cond` | 0.607 | 1.00 | 0.63 | False | accept=False |

## d_min = 1.5 Å

| Method | mapCC | peak | R1 | solved | notes |
|--------|-------|------|----|--------|-------|
| `charge_flipping` | 0.530 | 0.88 | 0.55 | False |  |
| `raar` | 0.376 | 0.94 | 0.58 | False |  |
| `ensemble_cf_raar` | 0.397 | 0.69 | 0.50 | False | pick=cf |
| `phai_fair` | 0.621 | 0.94 | 0.48 | False |  |
| `phai+CF_uncond` | 0.436 | 0.66 | 0.46 | False |  |
| `phai+RAAR_uncond` | 0.395 | 0.88 | 0.53 | False |  |
| `phai+CF_cond` | 0.436 | 0.66 | 0.46 | False | accept=True |
| `phai+RAAR_cond` | 0.621 | 0.94 | 0.48 | False | accept=False |

## d_min = 2.0 Å

| Method | mapCC | peak | R1 | solved | notes |
|--------|-------|------|----|--------|-------|
| `charge_flipping` | 0.479 | 0.66 | 0.52 | False |  |
| `raar` | 0.455 | 0.81 | 0.42 | False |  |
| `ensemble_cf_raar` | 0.464 | 0.53 | 0.54 | False | pick=cf |
| `phai_fair` | 0.628 | 0.75 | 0.54 | False |  |
| `phai+CF_uncond` | 0.444 | 0.75 | 0.47 | False |  |
| `phai+RAAR_uncond` | 0.481 | 0.81 | 0.45 | False |  |
| `phai+CF_cond` | 0.444 | 0.75 | 0.47 | False | accept=True |
| `phai+RAAR_cond` | 0.628 | 0.75 | 0.54 | False | accept=False |

## Interpretation

- Prefer **phai+RAAR_cond** when PhAI seed is already good at low res (conditional gate rejects harmful polish).
- At high res (~0.9 Å), unconditional or conditional polish may both solve.
- Ensemble CF+RAAR is a classical multistart baseline without PhAI.

JSON: `data/processed/cod_hybrid_benchmark.json`
