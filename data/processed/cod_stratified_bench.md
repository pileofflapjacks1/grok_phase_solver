# COD Vol-band stratified bench (v0.13)

Local COD Vol-band panel (Fobs + Fcalc). Not a 1505-structure Carrozzini panel. partial_30 = oracle control; fragment_half = no-oracle scientist path; auto = ab initio. Strict success = mapCC_OI≥0.7 AND peak_recovery≥0.5 AND R1≤0.45.

**d_min** = 1.0 Å · **datasets** = 6 · **rows OK** = 48

## Catalog

| COD | Vol (Å³) | Band | max Z | n non-H | SG | HKL |
|-----|----------|------|-------|---------|----|-----|
| 2012000 | 1027 | `vol_1000_3500` | 8 | 27 | P1211 | yes |
| 2013000 | 1015 | `vol_1000_3500` | 16 | 30 | P-1 | yes |
| 2016452 | 605 | `vol_lt_1000` | 8 | 8 | P121/c1 | yes |
| 2017775 | 4676 | `vol_gt_3500` | 6 | 137 | P212121 | yes |
| 2100301 | 660 | `vol_lt_1000` | 8 | 12 | P121/c1 | yes |
| 2200000 | 665 | `vol_lt_1000` | 8 | 18 | P1211 | yes |

## Results (all runs)

| Dataset | Amp | Run | mapCC | free FOM | R1 | solved | Vol band | max Z | s |
|---------|-----|-----|-------|----------|----|--------|----------|-------|---|
| 2012000 | fcalc | auto | **0.224** | 0.611 | 0.52 | False | `vol_1000_3500` | 8 | 7.5 |
| 2012000 | fcalc | partial_15 | **0.533** | 0.776 | 0.48 | False | `vol_1000_3500` | 8 | 7.4 |
| 2012000 | fcalc | partial_30 | **0.714** | 0.788 | 0.36 | True | `vol_1000_3500` | 8 | 7.5 |
| 2012000 | fcalc | fragment_half | **0.677** | 0.743 | 0.50 | False | `vol_1000_3500` | 8 | 8.3 |
| 2012000 | fobs | auto | **0.181** | 0.693 | 0.54 | False | `vol_1000_3500` | 8 | 6.8 |
| 2012000 | fobs | partial_15 | **0.466** | 0.716 | 0.51 | False | `vol_1000_3500` | 8 | 5.7 |
| 2012000 | fobs | partial_30 | **0.644** | 0.743 | 0.44 | False | `vol_1000_3500` | 8 | 5.8 |
| 2012000 | fobs | fragment_half | **0.675** | 0.776 | 0.53 | False | `vol_1000_3500` | 8 | 6.7 |
| 2013000 | fcalc | auto | **0.302** | 0.675 | 0.60 | False | `vol_1000_3500` | 16 | 9.4 |
| 2013000 | fcalc | partial_15 | **0.580** | 0.784 | 0.53 | False | `vol_1000_3500` | 16 | 8.3 |
| 2013000 | fcalc | partial_30 | **0.764** | 0.737 | 0.45 | True | `vol_1000_3500` | 16 | 8.4 |
| 2013000 | fcalc | fragment_half | **0.754** | 0.680 | 0.46 | False | `vol_1000_3500` | 16 | 9.5 |
| 2013000 | fobs | auto | **0.370** | 0.700 | 0.55 | False | `vol_1000_3500` | 16 | 7.7 |
| 2013000 | fobs | partial_15 | **0.589** | 0.679 | 0.56 | False | `vol_1000_3500` | 16 | 7.0 |
| 2013000 | fobs | partial_30 | **0.669** | 0.620 | 0.56 | False | `vol_1000_3500` | 16 | 5.4 |
| 2013000 | fobs | fragment_half | **0.751** | 0.663 | 0.55 | False | `vol_1000_3500` | 16 | 7.1 |
| 2016452 | fcalc | auto | **0.356** | 0.756 | 0.55 | False | `vol_lt_1000` | 8 | 5.2 |
| 2016452 | fcalc | partial_15 | **0.566** | 0.781 | 0.53 | False | `vol_lt_1000` | 8 | 4.2 |
| 2016452 | fcalc | partial_30 | **0.759** | 0.770 | 0.40 | True | `vol_lt_1000` | 8 | 4.2 |
| 2016452 | fcalc | fragment_half | **0.744** | 0.730 | 0.55 | False | `vol_lt_1000` | 8 | 4.7 |
| 2016452 | fobs | auto | **0.245** | 0.733 | 0.69 | False | `vol_lt_1000` | 8 | 3.8 |
| 2016452 | fobs | partial_15 | **0.599** | 0.715 | 0.56 | False | `vol_lt_1000` | 8 | 3.1 |
| 2016452 | fobs | partial_30 | **0.717** | 0.729 | 0.51 | False | `vol_lt_1000` | 8 | 3.1 |
| 2016452 | fobs | fragment_half | **0.791** | 0.779 | 0.62 | False | `vol_lt_1000` | 8 | 3.4 |
| 2017775 | fcalc | auto | **0.056** | 0.617 | 0.60 | False | `vol_gt_3500` | 6 | 73.6 |
| 2017775 | fcalc | partial_15 | **0.472** | 0.742 | 0.59 | False | `vol_gt_3500` | 6 | 68.5 |
| 2017775 | fcalc | partial_30 | **0.713** | 0.770 | 0.57 | False | `vol_gt_3500` | 6 | 66.0 |
| 2017775 | fcalc | fragment_half | **0.574** | 0.736 | 0.62 | False | `vol_gt_3500` | 6 | 90.6 |
| 2017775 | fobs | auto | **0.093** | 0.758 | 0.54 | False | `vol_gt_3500` | 6 | 81.7 |
| 2017775 | fobs | partial_15 | **0.422** | 0.689 | 0.52 | False | `vol_gt_3500` | 6 | 58.0 |
| 2017775 | fobs | partial_30 | **0.611** | 0.689 | 0.51 | False | `vol_gt_3500` | 6 | 62.5 |
| 2017775 | fobs | fragment_half | **0.411** | 0.754 | 0.54 | False | `vol_gt_3500` | 6 | 89.3 |
| 2100301 | fcalc | auto | **0.428** | 0.704 | 0.60 | False | `vol_lt_1000` | 8 | 7.9 |
| 2100301 | fcalc | partial_15 | **0.542** | 0.778 | 0.59 | False | `vol_lt_1000` | 8 | 6.7 |
| 2100301 | fcalc | partial_30 | **0.780** | 0.807 | 0.48 | False | `vol_lt_1000` | 8 | 6.5 |
| 2100301 | fcalc | fragment_half | **0.784** | 0.746 | 0.55 | False | `vol_lt_1000` | 8 | 7.1 |
| 2100301 | fobs | auto | **0.203** | 0.766 | 0.73 | False | `vol_lt_1000` | 8 | 5.6 |
| 2100301 | fobs | partial_15 | **0.519** | 0.721 | 0.68 | False | `vol_lt_1000` | 8 | 5.2 |
| 2100301 | fobs | partial_30 | **0.713** | 0.720 | 0.67 | False | `vol_lt_1000` | 8 | 5.3 |
| 2100301 | fobs | fragment_half | **0.745** | 0.766 | 0.69 | False | `vol_lt_1000` | 8 | 5.6 |
| 2200000 | fcalc | auto | **0.323** | 0.626 | 0.50 | False | `vol_lt_1000` | 8 | 5.8 |
| 2200000 | fcalc | partial_15 | **0.548** | 0.782 | 0.48 | False | `vol_lt_1000` | 8 | 5.1 |
| 2200000 | fcalc | partial_30 | **0.705** | 0.787 | 0.39 | True | `vol_lt_1000` | 8 | 8.6 |
| 2200000 | fcalc | fragment_half | **0.700** | 0.753 | 0.42 | True | `vol_lt_1000` | 8 | 10.5 |
| 2200000 | fobs | auto | **0.355** | 0.743 | 0.51 | False | `vol_lt_1000` | 8 | 5.7 |
| 2200000 | fobs | partial_15 | **0.505** | 0.749 | 0.48 | False | `vol_lt_1000` | 8 | 6.0 |
| 2200000 | fobs | partial_30 | **0.661** | 0.775 | 0.46 | False | `vol_lt_1000` | 8 | 5.9 |
| 2200000 | fobs | fragment_half | **0.674** | 0.760 | 0.50 | False | `vol_lt_1000` | 8 | 6.7 |

## Summary by Vol band × run

- `vol_1000_3500/auto`: n=4 mean mapCC=**0.269** (median 0.263)
- `vol_1000_3500/fragment_half`: n=4 mean mapCC=**0.714** (median 0.714)
- `vol_1000_3500/partial_15`: n=4 mean mapCC=**0.542** (median 0.557)
- `vol_1000_3500/partial_30`: n=4 mean mapCC=**0.698** (median 0.692)
- `vol_gt_3500/auto`: n=2 mean mapCC=**0.075** (median 0.075)
- `vol_gt_3500/fragment_half`: n=2 mean mapCC=**0.493** (median 0.493)
- `vol_gt_3500/partial_15`: n=2 mean mapCC=**0.447** (median 0.447)
- `vol_gt_3500/partial_30`: n=2 mean mapCC=**0.662** (median 0.662)
- `vol_lt_1000/auto`: n=6 mean mapCC=**0.318** (median 0.339)
- `vol_lt_1000/fragment_half`: n=6 mean mapCC=**0.740** (median 0.745)
- `vol_lt_1000/partial_15`: n=6 mean mapCC=**0.546** (median 0.545)
- `vol_lt_1000/partial_30`: n=6 mean mapCC=**0.723** (median 0.715)

## Vol 1000–3500 Å³ focus (AI-PhaSeed hybrid-friendly band)

| Run/amp | n | mean mapCC | median |
|---------|---|------------|--------|
| `auto/fcalc` | 2 | **0.263** | 0.263 |
| `auto/fobs` | 2 | **0.275** | 0.275 |
| `fragment_half/fcalc` | 2 | **0.715** | 0.715 |
| `fragment_half/fobs` | 2 | **0.713** | 0.713 |
| `partial_15/fcalc` | 2 | **0.557** | 0.557 |
| `partial_15/fobs` | 2 | **0.527** | 0.527 |
| `partial_30/fcalc` | 2 | **0.739** | 0.739 |
| `partial_30/fobs` | 2 | **0.657** | 0.657 |

## Takeaways

- **auto** (ab initio) is typically weak; mapCC often ≪ 0.5 on hard cells.
- **partial_30** (oracle) is the Lane-B control for the ≥~30% strong-φ bar.
- **partial_15** often under-seeds vs that bar.
- **fragment_half** is the no-oracle path; on coherent half-models it should approach partial_30 mapCC (see also `cod_hard_path_validation.md`).
- Vol **1000–3500 Å³** is the Carrozzini / AI-PhaSeed hybrid-friendly band.
- Strict multi-criterion *solved* can fail on R1 under short budgets.

Regenerate:
```bash
python scripts/run_cod_stratified_bench.py --dmin 1.0
```
