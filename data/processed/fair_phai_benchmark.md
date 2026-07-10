# Fair PhAI benchmark

Protocol: PhAI official **reindex_monoclinic → merge average → /max(|F|) → notebook grid (max_index=10) → 5 recycle cycles**.

Success criterion (same as solvability diagram): mapCC_OI≥0.70, peak_recovery≥0.50, R1≤0.45.

PhAI weights: `/Users/joe/Projects/grok_phase_solver/third_party/phai/weights/PhAI_model.pth`

| Dataset | d_min | Method | mapCC_OI | peak_rec | R1 | solved |
|---------|-------|--------|----------|----------|----|--------|
| COD_2016452 | 0.9 | `charge_flipping` | 0.350 | 0.97 | 0.63 | no |
| COD_2016452 | 0.9 | `phase_recycle` | 0.379 | 0.97 | 0.56 | no |
| COD_2016452 | 0.9 | `phai_fair` | 0.558 | 1.00 | 0.61 | no |
| COD_2016452 | 0.9 | `phai_fair+CF` | 0.867 | 1.00 | 0.34 | **yes** |
| COD_2016452 | 1.2 | `charge_flipping` | 0.443 | 0.94 | 0.59 | no |
| COD_2016452 | 1.2 | `phase_recycle` | 0.417 | 0.94 | 0.53 | no |
| COD_2016452 | 1.2 | `phai_fair` | 0.607 | 1.00 | 0.63 | no |
| COD_2016452 | 1.2 | `phai_fair+CF` | 0.489 | 0.84 | 0.51 | no |
| COD_2016452 | 1.5 | `charge_flipping` | 0.532 | 0.88 | 0.53 | no |
| COD_2016452 | 1.5 | `phase_recycle` | 0.533 | 0.91 | 0.43 | no |
| COD_2016452 | 1.5 | `phai_fair` | 0.621 | 0.94 | 0.48 | no |
| COD_2016452 | 1.5 | `phai_fair+CF` | 0.436 | 0.66 | 0.46 | no |
| COD_2016452 | 2.0 | `charge_flipping` | 0.479 | 0.66 | 0.52 | no |
| COD_2016452 | 2.0 | `phase_recycle` | 0.590 | 0.75 | 0.40 | no |
| COD_2016452 | 2.0 | `phai_fair` | 0.628 | 0.75 | 0.54 | no |
| COD_2016452 | 2.0 | `phai_fair+CF` | 0.444 | 0.75 | 0.47 | no |
| COD_2100301 | 0.9 | `charge_flipping` | 0.447 | 1.00 | 0.63 | no |
| COD_2100301 | 0.9 | `phase_recycle` | 0.347 | 1.00 | 0.60 | no |
| COD_2100301 | 0.9 | `phai_fair` | 0.482 | 1.00 | 0.68 | no |
| COD_2100301 | 0.9 | `phai_fair+CF` | 0.641 | 1.00 | 0.54 | no |
| COD_2100301 | 1.2 | `charge_flipping` | 0.403 | 0.96 | 0.57 | no |
| COD_2100301 | 1.2 | `phase_recycle` | 0.457 | 0.96 | 0.47 | no |
| COD_2100301 | 1.2 | `phai_fair` | 0.579 | 1.00 | 0.64 | no |
| COD_2100301 | 1.2 | `phai_fair+CF` | 0.389 | 0.92 | 0.51 | no |
| COD_2100301 | 1.5 | `charge_flipping` | 0.506 | 1.00 | 0.51 | no |
| COD_2100301 | 1.5 | `phase_recycle` | 0.463 | 1.00 | 0.48 | no |
| COD_2100301 | 1.5 | `phai_fair` | 0.406 | 1.00 | 0.63 | no |
| COD_2100301 | 1.5 | `phai_fair+CF` | 0.355 | 0.83 | 0.51 | no |

## Interpretation

- **Fair packing** is necessary but not sufficient for reproducing Science paper numbers:
  training distribution, multi-start seeds, and experimental vs Fcalc still matter.
- If `phai_fair` ≪ CF at atomic resolution on Fcalc, CF remains the right default.
- If `phai_fair` > CF at low resolution / incomplete data, that is the target regime.
- `phai_fair+CF` tests whether PhAI provides a useful **seed** for classical polish.

JSON: `data/processed/fair_phai_benchmark.json`
