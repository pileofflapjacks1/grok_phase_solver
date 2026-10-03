# COD Vol-band stratified bench (v0.13)

Local COD Vol-band panel (Fobs + Fcalc). Not a 1505-structure Carrozzini panel. partial_30 = oracle control; fragment_half = no-oracle scientist path; auto = ab initio. Strict success = mapCC_OI≥0.7 AND peak_recovery≥0.5 AND R1≤0.45.

**d_min** = 1.0 Å · **datasets** = 16 · **rows OK** = 128

## Catalog

| COD | Vol (Å³) | Band | max Z | n non-H | SG | HKL |
|-----|----------|------|-------|---------|----|-----|
| 2012000 | 1027 | `vol_1000_3500` | 8 | 27 | P1211 | yes |
| 2013000 | 1015 | `vol_1000_3500` | 16 | 30 | P-1 | yes |
| 2016452 | 605 | `vol_lt_1000` | 8 | 8 | P121/c1 | yes |
| 2017775 | 4676 | `vol_gt_3500` | 6 | 137 | P212121 | yes |
| 2100301 | 660 | `vol_lt_1000` | 8 | 12 | P121/c1 | yes |
| 2200000 | 665 | `vol_lt_1000` | 8 | 18 | P1211 | yes |
| 1543089 | 1638 | `vol_1000_3500` | 8 | 26 | P121/c1 | yes |
| 1543227 | 1786 | `vol_1000_3500` | 8 | 26 | P121/n1 | yes |
| 1544230 | 1878 | `vol_1000_3500` | 8 | 26 | P121/c1 | yes |
| 1544651 | 1752 | `vol_1000_3500` | 8 | 26 | P121/c1 | yes |
| 1549607 | 1628 | `vol_1000_3500` | 9 | 26 | P121/c1 | yes |
| 1550274 | 1689 | `vol_1000_3500` | 8 | 26 | P121/c1 | yes |
| 2016430 | 2611 | `vol_1000_3500` | 8 | 26 | Pbca | yes |
| 2221836 | 3176 | `vol_1000_3500` | 8 | 26 | Pbca | yes |
| 2227862 | 3077 | `vol_1000_3500` | 8 | 26 | Pbca | yes |
| 2233297 | 3160 | `vol_1000_3500` | 8 | 26 | C12/c1 | yes |

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
| 1543089 | fcalc | auto | **0.164** | 0.676 | 0.62 | False | `vol_1000_3500` | 8 | 15.6 |
| 1543089 | fcalc | partial_15 | **0.620** | 0.782 | 0.55 | False | `vol_1000_3500` | 8 | 13.4 |
| 1543089 | fcalc | partial_30 | **0.784** | 0.793 | 0.48 | False | `vol_1000_3500` | 8 | 13.6 |
| 1543089 | fcalc | fragment_half | **0.724** | 0.717 | 0.57 | False | `vol_1000_3500` | 8 | 16.4 |
| 1543089 | fobs | auto | **0.188** | 0.707 | 0.63 | False | `vol_1000_3500` | 8 | 15.4 |
| 1543089 | fobs | partial_15 | **0.538** | 0.717 | 0.61 | False | `vol_1000_3500` | 8 | 16.5 |
| 1543089 | fobs | partial_30 | **0.730** | 0.743 | 0.55 | False | `vol_1000_3500` | 8 | 28.6 |
| 1543089 | fobs | fragment_half | **0.744** | 0.774 | 0.59 | False | `vol_1000_3500` | 8 | 21.4 |
| 1543227 | fcalc | auto | **0.093** | 0.679 | 0.66 | False | `vol_1000_3500` | 8 | 19.0 |
| 1543227 | fcalc | partial_15 | **0.586** | 0.785 | 0.51 | False | `vol_1000_3500` | 8 | 15.0 |
| 1543227 | fcalc | partial_30 | **0.759** | 0.773 | 0.39 | True | `vol_1000_3500` | 8 | 14.8 |
| 1543227 | fcalc | fragment_half | **0.771** | 0.703 | 0.42 | True | `vol_1000_3500` | 8 | 15.9 |
| 1543227 | fobs | auto | **0.215** | 0.730 | 0.60 | False | `vol_1000_3500` | 8 | 13.3 |
| 1543227 | fobs | partial_15 | **0.535** | 0.719 | 0.55 | False | `vol_1000_3500` | 8 | 11.4 |
| 1543227 | fobs | partial_30 | **0.730** | 0.757 | 0.41 | True | `vol_1000_3500` | 8 | 12.2 |
| 1543227 | fobs | fragment_half | **0.762** | 0.782 | 0.47 | False | `vol_1000_3500` | 8 | 12.4 |
| 1544230 | fcalc | auto | **0.171** | 0.592 | 0.63 | False | `vol_1000_3500` | 8 | 16.8 |
| 1544230 | fcalc | partial_15 | **0.568** | 0.786 | 0.55 | False | `vol_1000_3500` | 8 | 14.5 |
| 1544230 | fcalc | partial_30 | **0.779** | 0.780 | 0.45 | True | `vol_1000_3500` | 8 | 14.5 |
| 1544230 | fcalc | fragment_half | **0.671** | 0.692 | 0.53 | False | `vol_1000_3500` | 8 | 17.3 |
| 1544230 | fobs | auto | **0.165** | 0.686 | 0.63 | False | `vol_1000_3500` | 8 | 15.2 |
| 1544230 | fobs | partial_15 | **0.491** | 0.717 | 0.58 | False | `vol_1000_3500` | 8 | 12.3 |
| 1544230 | fobs | partial_30 | **0.707** | 0.743 | 0.47 | False | `vol_1000_3500` | 8 | 12.5 |
| 1544230 | fobs | fragment_half | **0.668** | 0.777 | 0.56 | False | `vol_1000_3500` | 8 | 15.6 |
| 1544651 | fcalc | auto | **0.101** | 0.715 | 0.67 | False | `vol_1000_3500` | 8 | 18.6 |
| 1544651 | fcalc | partial_15 | **0.617** | 0.782 | 0.52 | False | `vol_1000_3500` | 8 | 15.2 |
| 1544651 | fcalc | partial_30 | **0.809** | 0.774 | 0.39 | True | `vol_1000_3500` | 8 | 14.9 |
| 1544651 | fcalc | fragment_half | **0.759** | 0.698 | 0.46 | False | `vol_1000_3500` | 8 | 15.9 |
| 1544651 | fobs | auto | **0.160** | 0.681 | 0.63 | False | `vol_1000_3500` | 8 | 15.1 |
| 1544651 | fobs | partial_15 | **0.572** | 0.726 | 0.56 | False | `vol_1000_3500` | 8 | 11.9 |
| 1544651 | fobs | partial_30 | **0.723** | 0.747 | 0.46 | False | `vol_1000_3500` | 8 | 12.6 |
| 1544651 | fobs | fragment_half | **0.737** | 0.780 | 0.57 | False | `vol_1000_3500` | 8 | 13.4 |
| 1549607 | fcalc | auto | **0.106** | 0.717 | 0.64 | False | `vol_1000_3500` | 9 | 18.4 |
| 1549607 | fcalc | partial_15 | **0.567** | 0.768 | 0.52 | False | `vol_1000_3500` | 9 | 14.5 |
| 1549607 | fcalc | partial_30 | **0.785** | 0.791 | 0.37 | True | `vol_1000_3500` | 9 | 14.4 |
| 1549607 | fcalc | fragment_half | **0.698** | 0.698 | 0.51 | False | `vol_1000_3500` | 9 | 16.3 |
| 1549607 | fobs | auto | **0.087** | 0.699 | 0.62 | False | `vol_1000_3500` | 9 | 22.6 |
| 1549607 | fobs | partial_15 | **0.620** | 0.768 | 0.52 | False | `vol_1000_3500` | 9 | 16.1 |
| 1549607 | fobs | partial_30 | **0.797** | 0.783 | 0.42 | True | `vol_1000_3500` | 9 | 16.3 |
| 1549607 | fobs | fragment_half | **0.663** | 0.703 | 0.54 | False | `vol_1000_3500` | 9 | 19.3 |
| 1550274 | fcalc | auto | **0.241** | 0.631 | 0.60 | False | `vol_1000_3500` | 8 | 17.0 |
| 1550274 | fcalc | partial_15 | **0.588** | 0.780 | 0.53 | False | `vol_1000_3500` | 8 | 13.9 |
| 1550274 | fcalc | partial_30 | **0.772** | 0.787 | 0.45 | False | `vol_1000_3500` | 8 | 14.6 |
| 1550274 | fcalc | fragment_half | **0.713** | 0.706 | 0.50 | False | `vol_1000_3500` | 8 | 15.8 |
| 1550274 | fobs | auto | **0.113** | 0.761 | 0.61 | False | `vol_1000_3500` | 8 | 12.4 |
| 1550274 | fobs | partial_15 | **0.470** | 0.692 | 0.60 | False | `vol_1000_3500` | 8 | 10.6 |
| 1550274 | fobs | partial_30 | **0.712** | 0.712 | 0.59 | False | `vol_1000_3500` | 8 | 10.7 |
| 1550274 | fobs | fragment_half | **0.685** | 0.768 | 0.61 | False | `vol_1000_3500` | 8 | 13.2 |
| 2016430 | fcalc | auto | **0.079** | 0.650 | 0.66 | False | `vol_1000_3500` | 8 | 31.4 |
| 2016430 | fcalc | partial_15 | **0.568** | 0.761 | 0.57 | False | `vol_1000_3500` | 8 | 24.1 |
| 2016430 | fcalc | partial_30 | **0.779** | 0.791 | 0.49 | False | `vol_1000_3500` | 8 | 23.4 |
| 2016430 | fcalc | fragment_half | **0.722** | 0.752 | 0.54 | False | `vol_1000_3500` | 8 | 28.4 |
| 2016430 | fobs | auto | **0.143** | 0.701 | 0.60 | False | `vol_1000_3500` | 8 | 24.7 |
| 2016430 | fobs | partial_15 | **0.486** | 0.679 | 0.56 | False | `vol_1000_3500` | 8 | 20.4 |
| 2016430 | fobs | partial_30 | **0.692** | 0.685 | 0.53 | False | `vol_1000_3500` | 8 | 19.5 |
| 2016430 | fobs | fragment_half | **0.708** | 0.723 | 0.56 | False | `vol_1000_3500` | 8 | 24.1 |
| 2221836 | fcalc | auto | **0.104** | 0.649 | 0.71 | False | `vol_1000_3500` | 8 | 35.4 |
| 2221836 | fcalc | partial_15 | **0.617** | 0.784 | 0.66 | False | `vol_1000_3500` | 8 | 29.9 |
| 2221836 | fcalc | partial_30 | **0.766** | 0.793 | 0.62 | False | `vol_1000_3500` | 8 | 31.3 |
| 2221836 | fcalc | fragment_half | **0.720** | 0.719 | 0.62 | False | `vol_1000_3500` | 8 | 36.3 |
| 2221836 | fobs | auto | **0.159** | 0.702 | 0.65 | False | `vol_1000_3500` | 8 | 30.5 |
| 2221836 | fobs | partial_15 | **0.545** | 0.682 | 0.62 | False | `vol_1000_3500` | 8 | 24.2 |
| 2221836 | fobs | partial_30 | **0.714** | 0.702 | 0.58 | False | `vol_1000_3500` | 8 | 25.3 |
| 2221836 | fobs | fragment_half | **0.717** | 0.758 | 0.62 | False | `vol_1000_3500` | 8 | 31.2 |
| 2227862 | fcalc | auto | **0.078** | 0.643 | 0.68 | False | `vol_1000_3500` | 8 | 33.0 |
| 2227862 | fcalc | partial_15 | **0.602** | 0.784 | 0.57 | False | `vol_1000_3500` | 8 | 25.3 |
| 2227862 | fcalc | partial_30 | **0.800** | 0.790 | 0.49 | False | `vol_1000_3500` | 8 | 26.8 |
| 2227862 | fcalc | fragment_half | **0.788** | 0.721 | 0.53 | False | `vol_1000_3500` | 8 | 29.4 |
| 2227862 | fobs | auto | **0.143** | 0.720 | 0.61 | False | `vol_1000_3500` | 8 | 25.2 |
| 2227862 | fobs | partial_15 | **0.526** | 0.694 | 0.57 | False | `vol_1000_3500` | 8 | 20.1 |
| 2227862 | fobs | partial_30 | **0.710** | 0.696 | 0.53 | False | `vol_1000_3500` | 8 | 20.0 |
| 2227862 | fobs | fragment_half | **0.765** | 0.745 | 0.56 | False | `vol_1000_3500` | 8 | 24.5 |
| 2233297 | fcalc | auto | **0.111** | 0.647 | 0.68 | False | `vol_1000_3500` | 8 | 24.4 |
| 2233297 | fcalc | partial_15 | **0.632** | 0.789 | 0.58 | False | `vol_1000_3500` | 8 | 20.7 |
| 2233297 | fcalc | partial_30 | **0.769** | 0.800 | 0.56 | False | `vol_1000_3500` | 8 | 20.0 |
| 2233297 | fcalc | fragment_half | **0.753** | 0.718 | 0.59 | False | `vol_1000_3500` | 8 | 22.3 |
| 2233297 | fobs | auto | **0.231** | 0.730 | 0.64 | False | `vol_1000_3500` | 8 | 21.6 |
| 2233297 | fobs | partial_15 | **0.523** | 0.719 | 0.62 | False | `vol_1000_3500` | 8 | 18.7 |
| 2233297 | fobs | partial_30 | **0.714** | 0.757 | 0.57 | False | `vol_1000_3500` | 8 | 17.5 |
| 2233297 | fobs | fragment_half | **0.763** | 0.782 | 0.59 | False | `vol_1000_3500` | 8 | 20.6 |

## Summary by Vol band × run

- `vol_1000_3500/auto`: n=24 mean mapCC=**0.164** (median 0.159)
- `vol_1000_3500/fragment_half`: n=24 mean mapCC=**0.724** (median 0.723)
- `vol_1000_3500/partial_15`: n=24 mean mapCC=**0.560** (median 0.568)
- `vol_1000_3500/partial_30`: n=24 mean mapCC=**0.743** (median 0.745)
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
| `auto/fcalc` | 12 | **0.148** | 0.108 |
| `auto/fobs` | 12 | **0.180** | 0.163 |
| `fragment_half/fcalc` | 12 | **0.729** | 0.723 |
| `fragment_half/fobs` | 12 | **0.720** | 0.727 |
| `partial_15/fcalc` | 12 | **0.590** | 0.587 |
| `partial_15/fobs` | 12 | **0.530** | 0.530 |
| `partial_30/fcalc` | 12 | **0.773** | 0.775 |
| `partial_30/fobs` | 12 | **0.712** | 0.713 |

## Takeaways

- **auto** (ab initio) is typically weak; mapCC often ≪ 0.5 on hard cells.
- **partial_30** (oracle) is the Lane-B control for the ≥~30% strong-φ bar.
- **partial_15** often under-seeds vs that bar.
- **fragment_half** is the no-oracle path; on coherent half-models it should approach partial_30 mapCC (see also `cod_hard_path_validation.md`).
- Vol **1000–3500 Å³** is the Carrozzini / AI-PhaSeed hybrid-friendly band.
- Strict multi-criterion *solved* can fail on R1 under short budgets.
- Mid-band expansion (2026-09): ten additional light-atom CHNOF Fobs cells (~26 non-H). Original six rows are unchanged.
- Pooled Vol 1000–3500 (n=24): fragment_half mean mapCC 0.724, partial_30 0.743, auto 0.164. Fobs-only (n=12): fragment_half 0.720, partial_30 0.712, auto 0.180. fragment_half does not beat partial_30 on the pooled table. No Fobs fragment_half row is a strict solve (carbon-peak R1 stays above 0.45).
- COD **1544230** was rescored after the bench and `solve_structure` shared `d_min_keep_mask` (d ≥ d_min − 1e-9). Other panel cells have no reflection in that window, so their rows were not rerun. Fobs fragment_half mapCC moved from 0.669 to 0.668; carbon-peak R1 is 0.562 and peak recovery is 1.000. That row is still not a strict solve. Fcalc partial_30 on this cell now scores solved (mapCC 0.779, peak recovery 1.000, R1 0.447). See `r1_gate_diagnostic.md`.
- Fobs `fragment_half` gates on this mid-band: mapCC ≥ 0.7 on 8/12, peak recovery ≥ 0.5 on 12/12, carbon-peak R1 ≤ 0.45 on 0/12. A side R1 with deposited element types on those peaks stays above 0.45 (minimum 0.477) and does not change `solved`.

Regenerate:
```bash
python scripts/run_cod_stratified_bench.py --dmin 1.0
```
