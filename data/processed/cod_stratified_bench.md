# COD Vol-band stratified bench (v0.13)

Local COD Vol-band panel (Fobs + Fcalc). Not a 1505-structure Carrozzini panel. partial_30 = oracle control; fragment_half = no-oracle scientist path; auto = ab initio. Strict success = mapCC_OI≥0.7 AND peak_recovery≥0.5 AND R1≤0.45.

**d_min** = 1.0 Å · **datasets** = 16 · **rows OK** = 128

## Catalog

| COD | Vol (Å³) | Band | max Z | n non-H | SG | HKL |
|-----|----------|------|-------|---------|----|-----|
| 1543089 | 1638 | `vol_1000_3500` | 8 | 26 | P121/c1 | yes |
| 1543227 | 1786 | `vol_1000_3500` | 8 | 26 | P121/n1 | yes |
| 1544230 | 1878 | `vol_1000_3500` | 8 | 26 | P121/c1 | yes |
| 1544651 | 1752 | `vol_1000_3500` | 8 | 26 | P121/c1 | yes |
| 1549607 | 1628 | `vol_1000_3500` | 9 | 26 | P121/c1 | yes |
| 1550274 | 1689 | `vol_1000_3500` | 8 | 26 | P121/c1 | yes |
| 2012000 | 1027 | `vol_1000_3500` | 8 | 27 | P1211 | yes |
| 2013000 | 1015 | `vol_1000_3500` | 16 | 30 | P-1 | yes |
| 2016430 | 2611 | `vol_1000_3500` | 8 | 26 | Pbca | yes |
| 2016452 | 605 | `vol_lt_1000` | 8 | 8 | P121/c1 | yes |
| 2017775 | 4676 | `vol_gt_3500` | 6 | 137 | P212121 | yes |
| 2100301 | 660 | `vol_lt_1000` | 8 | 12 | P121/c1 | yes |
| 2200000 | 665 | `vol_lt_1000` | 8 | 18 | P1211 | yes |
| 2221836 | 3176 | `vol_1000_3500` | 8 | 26 | Pbca | yes |
| 2227862 | 3077 | `vol_1000_3500` | 8 | 26 | Pbca | yes |
| 2233297 | 3160 | `vol_1000_3500` | 8 | 26 | C12/c1 | yes |

## Results (all runs)

| Dataset | Amp | Run | mapCC | free FOM | R1 | solved | Vol band | max Z | s |
|---------|-----|-----|-------|----------|----|--------|----------|-------|---|
| 1543089 | fcalc | auto | **0.285** | 0.773 | 0.59 | False | `vol_1000_3500` | 8 | 46.1 |
| 1543089 | fcalc | partial_15 | **0.648** | 0.788 | 0.51 | False | `vol_1000_3500` | 8 | 12.3 |
| 1543089 | fcalc | partial_30 | **0.819** | 0.768 | 0.42 | True | `vol_1000_3500` | 8 | 10.7 |
| 1543089 | fcalc | fragment_half | **0.725** | 0.717 | 0.53 | False | `vol_1000_3500` | 8 | 13.8 |
| 1543089 | fobs | auto | **0.184** | 0.756 | 0.62 | False | `vol_1000_3500` | 8 | 46.8 |
| 1543089 | fobs | partial_15 | **0.538** | 0.717 | 0.61 | False | `vol_1000_3500` | 8 | 14.5 |
| 1543089 | fobs | partial_30 | **0.730** | 0.743 | 0.53 | False | `vol_1000_3500` | 8 | 14.4 |
| 1543089 | fobs | fragment_half | **0.744** | 0.774 | 0.57 | False | `vol_1000_3500` | 8 | 14.7 |
| 1543227 | fcalc | auto | **0.201** | 0.764 | 0.54 | False | `vol_1000_3500` | 8 | 66.6 |
| 1543227 | fcalc | partial_15 | **0.659** | 0.789 | 0.41 | False | `vol_1000_3500` | 8 | 13.6 |
| 1543227 | fcalc | partial_30 | **0.804** | 0.755 | 0.31 | True | `vol_1000_3500` | 8 | 15.7 |
| 1543227 | fcalc | fragment_half | **0.771** | 0.696 | 0.34 | True | `vol_1000_3500` | 8 | 13.1 |
| 1543227 | fobs | auto | **0.163** | 0.743 | 0.57 | False | `vol_1000_3500` | 8 | 50.0 |
| 1543227 | fobs | partial_15 | **0.535** | 0.719 | 0.54 | False | `vol_1000_3500` | 8 | 12.6 |
| 1543227 | fobs | partial_30 | **0.730** | 0.757 | 0.39 | True | `vol_1000_3500` | 8 | 12.4 |
| 1543227 | fobs | fragment_half | **0.762** | 0.782 | 0.44 | True | `vol_1000_3500` | 8 | 23.6 |
| 1544230 | fcalc | auto | **0.234** | 0.780 | 0.57 | False | `vol_1000_3500` | 8 | 136.3 |
| 1544230 | fcalc | partial_15 | **0.650** | 0.789 | 0.50 | False | `vol_1000_3500` | 8 | 21.8 |
| 1544230 | fcalc | partial_30 | **0.822** | 0.753 | 0.33 | True | `vol_1000_3500` | 8 | 18.2 |
| 1544230 | fcalc | fragment_half | **0.670** | 0.692 | 0.46 | False | `vol_1000_3500` | 8 | 19.9 |
| 1544230 | fobs | auto | **0.191** | 0.734 | 0.59 | False | `vol_1000_3500` | 8 | 102.7 |
| 1544230 | fobs | partial_15 | **0.491** | 0.717 | 0.57 | False | `vol_1000_3500` | 8 | 29.1 |
| 1544230 | fobs | partial_30 | **0.707** | 0.743 | 0.45 | True | `vol_1000_3500` | 8 | 60.7 |
| 1544230 | fobs | fragment_half | **0.668** | 0.777 | 0.54 | False | `vol_1000_3500` | 8 | 67.8 |
| 1544651 | fcalc | auto | **0.219** | 0.782 | 0.57 | False | `vol_1000_3500` | 8 | 268.1 |
| 1544651 | fcalc | partial_15 | **0.664** | 0.788 | 0.45 | False | `vol_1000_3500` | 8 | 17.5 |
| 1544651 | fcalc | partial_30 | **0.825** | 0.750 | 0.29 | True | `vol_1000_3500` | 8 | 16.2 |
| 1544651 | fcalc | fragment_half | **0.759** | 0.699 | 0.39 | True | `vol_1000_3500` | 8 | 15.7 |
| 1544651 | fobs | auto | **0.156** | 0.744 | 0.56 | False | `vol_1000_3500` | 8 | 63.7 |
| 1544651 | fobs | partial_15 | **0.572** | 0.726 | 0.56 | False | `vol_1000_3500` | 8 | 19.9 |
| 1544651 | fobs | partial_30 | **0.723** | 0.747 | 0.44 | True | `vol_1000_3500` | 8 | 18.9 |
| 1544651 | fobs | fragment_half | **0.737** | 0.780 | 0.55 | False | `vol_1000_3500` | 8 | 18.6 |
| 1549607 | fcalc | auto | **0.148** | 0.787 | 0.54 | False | `vol_1000_3500` | 9 | 90.7 |
| 1549607 | fcalc | partial_15 | **0.625** | 0.785 | 0.46 | False | `vol_1000_3500` | 9 | 22.2 |
| 1549607 | fcalc | partial_30 | **0.823** | 0.768 | 0.29 | True | `vol_1000_3500` | 9 | 22.3 |
| 1549607 | fcalc | fragment_half | **0.696** | 0.698 | 0.49 | False | `vol_1000_3500` | 9 | 24.7 |
| 1549607 | fobs | auto | **0.120** | 0.774 | 0.53 | False | `vol_1000_3500` | 9 | 93.5 |
| 1549607 | fobs | partial_15 | **0.635** | 0.773 | 0.51 | False | `vol_1000_3500` | 9 | 16.9 |
| 1549607 | fobs | partial_30 | **0.815** | 0.785 | 0.34 | True | `vol_1000_3500` | 9 | 17.3 |
| 1549607 | fobs | fragment_half | **0.663** | 0.702 | 0.53 | False | `vol_1000_3500` | 9 | 18.4 |
| 1550274 | fcalc | auto | **0.219** | 0.788 | 0.54 | False | `vol_1000_3500` | 8 | 65.8 |
| 1550274 | fcalc | partial_15 | **0.636** | 0.787 | 0.46 | False | `vol_1000_3500` | 8 | 20.2 |
| 1550274 | fcalc | partial_30 | **0.796** | 0.762 | 0.35 | True | `vol_1000_3500` | 8 | 11.6 |
| 1550274 | fcalc | fragment_half | **0.714** | 0.706 | 0.45 | True | `vol_1000_3500` | 8 | 11.3 |
| 1550274 | fobs | auto | **0.125** | 0.708 | 0.61 | False | `vol_1000_3500` | 8 | 38.8 |
| 1550274 | fobs | partial_15 | **0.470** | 0.692 | 0.60 | False | `vol_1000_3500` | 8 | 12.0 |
| 1550274 | fobs | partial_30 | **0.712** | 0.712 | 0.57 | False | `vol_1000_3500` | 8 | 11.6 |
| 1550274 | fobs | fragment_half | **0.685** | 0.768 | 0.60 | False | `vol_1000_3500` | 8 | 11.9 |
| 2012000 | fcalc | auto | **0.213** | 0.756 | 0.45 | False | `vol_1000_3500` | 8 | 24.9 |
| 2012000 | fcalc | partial_15 | **0.563** | 0.787 | 0.45 | False | `vol_1000_3500` | 8 | 3.2 |
| 2012000 | fcalc | partial_30 | **0.762** | 0.769 | 0.26 | True | `vol_1000_3500` | 8 | 3.6 |
| 2012000 | fcalc | fragment_half | **0.677** | 0.743 | 0.48 | False | `vol_1000_3500` | 8 | 3.1 |
| 2012000 | fobs | auto | **0.186** | 0.755 | 0.52 | False | `vol_1000_3500` | 8 | 17.8 |
| 2012000 | fobs | partial_15 | **0.466** | 0.716 | 0.50 | False | `vol_1000_3500` | 8 | 2.5 |
| 2012000 | fobs | partial_30 | **0.644** | 0.743 | 0.42 | False | `vol_1000_3500` | 8 | 2.4 |
| 2012000 | fobs | fragment_half | **0.675** | 0.776 | 0.51 | False | `vol_1000_3500` | 8 | 2.8 |
| 2013000 | fcalc | auto | **0.680** | 0.703 | 0.46 | False | `vol_1000_3500` | 16 | 26.4 |
| 2013000 | fcalc | partial_15 | **0.676** | 0.752 | 0.51 | False | `vol_1000_3500` | 16 | 3.7 |
| 2013000 | fcalc | partial_30 | **0.811** | 0.692 | 0.36 | True | `vol_1000_3500` | 16 | 3.5 |
| 2013000 | fcalc | fragment_half | **0.753** | 0.676 | 0.40 | True | `vol_1000_3500` | 16 | 3.7 |
| 2013000 | fobs | auto | **0.656** | 0.659 | 0.53 | False | `vol_1000_3500` | 16 | 31.8 |
| 2013000 | fobs | partial_15 | **0.589** | 0.679 | 0.55 | False | `vol_1000_3500` | 16 | 5.2 |
| 2013000 | fobs | partial_30 | **0.766** | 0.597 | 0.52 | False | `vol_1000_3500` | 16 | 4.5 |
| 2013000 | fobs | fragment_half | **0.751** | 0.663 | 0.53 | False | `vol_1000_3500` | 16 | 4.3 |
| 2016430 | fcalc | auto | **0.105** | 0.786 | 0.54 | False | `vol_1000_3500` | 8 | 236.7 |
| 2016430 | fcalc | partial_15 | **0.622** | 0.776 | 0.52 | False | `vol_1000_3500` | 8 | 153.5 |
| 2016430 | fcalc | partial_30 | **0.814** | 0.797 | 0.44 | True | `vol_1000_3500` | 8 | 99.5 |
| 2016430 | fcalc | fragment_half | **0.722** | 0.752 | 0.50 | False | `vol_1000_3500` | 8 | 83.6 |
| 2016430 | fobs | auto | **0.167** | 0.684 | 0.57 | False | `vol_1000_3500` | 8 | 169.9 |
| 2016430 | fobs | partial_15 | **0.486** | 0.679 | 0.56 | False | `vol_1000_3500` | 8 | 114.3 |
| 2016430 | fobs | partial_30 | **0.692** | 0.685 | 0.52 | False | `vol_1000_3500` | 8 | 131.2 |
| 2016430 | fobs | fragment_half | **0.708** | 0.723 | 0.55 | False | `vol_1000_3500` | 8 | 86.7 |
| 2016452 | fcalc | auto | **0.694** | 0.791 | 0.35 | False | `vol_lt_1000` | 8 | 18.7 |
| 2016452 | fcalc | partial_15 | **0.654** | 0.783 | 0.42 | False | `vol_lt_1000` | 8 | 1.8 |
| 2016452 | fcalc | partial_30 | **0.798** | 0.769 | 0.33 | True | `vol_lt_1000` | 8 | 1.8 |
| 2016452 | fcalc | fragment_half | **0.743** | 0.727 | 0.51 | False | `vol_lt_1000` | 8 | 2.0 |
| 2016452 | fobs | auto | **0.256** | 0.708 | 0.63 | False | `vol_lt_1000` | 8 | 14.1 |
| 2016452 | fobs | partial_15 | **0.599** | 0.715 | 0.56 | False | `vol_lt_1000` | 8 | 1.3 |
| 2016452 | fobs | partial_30 | **0.717** | 0.729 | 0.50 | False | `vol_lt_1000` | 8 | 1.4 |
| 2016452 | fobs | fragment_half | **0.791** | 0.779 | 0.59 | False | `vol_lt_1000` | 8 | 1.4 |
| 2017775 | fcalc | auto | **0.048** | 0.778 | 0.54 | False | `vol_gt_3500` | 6 | 1629.3 |
| 2017775 | fcalc | partial_15 | **0.518** | 0.755 | 0.58 | False | `vol_gt_3500` | 6 | 1842.0 |
| 2017775 | fcalc | partial_30 | **0.756** | 0.782 | 0.56 | False | `vol_gt_3500` | 6 | 3138.7 |
| 2017775 | fcalc | fragment_half | **0.574** | 0.735 | 0.61 | False | `vol_gt_3500` | 6 | 1022.9 |
| 2017775 | fobs | auto | **0.115** | 0.687 | 0.51 | False | `vol_gt_3500` | 6 | 2065.9 |
| 2017775 | fobs | partial_15 | **0.422** | 0.689 | 0.52 | False | `vol_gt_3500` | 6 | 972.5 |
| 2017775 | fobs | partial_30 | **0.611** | 0.689 | 0.50 | False | `vol_gt_3500` | 6 | 1042.0 |
| 2017775 | fobs | fragment_half | **0.411** | 0.754 | 0.53 | False | `vol_gt_3500` | 6 | 565.7 |
| 2100301 | fcalc | auto | **0.431** | 0.790 | 0.54 | False | `vol_lt_1000` | 8 | 15.0 |
| 2100301 | fcalc | partial_15 | **0.589** | 0.791 | 0.55 | False | `vol_lt_1000` | 8 | 2.0 |
| 2100301 | fcalc | partial_30 | **0.801** | 0.800 | 0.43 | True | `vol_lt_1000` | 8 | 2.0 |
| 2100301 | fcalc | fragment_half | **0.784** | 0.747 | 0.52 | False | `vol_lt_1000` | 8 | 2.1 |
| 2100301 | fobs | auto | **0.221** | 0.706 | 0.68 | False | `vol_lt_1000` | 8 | 12.0 |
| 2100301 | fobs | partial_15 | **0.519** | 0.721 | 0.67 | False | `vol_lt_1000` | 8 | 1.5 |
| 2100301 | fobs | partial_30 | **0.713** | 0.720 | 0.65 | False | `vol_lt_1000` | 8 | 1.5 |
| 2100301 | fobs | fragment_half | **0.745** | 0.766 | 0.67 | False | `vol_lt_1000` | 8 | 1.7 |
| 2200000 | fcalc | auto | **0.295** | 0.731 | 0.46 | False | `vol_lt_1000` | 8 | 16.5 |
| 2200000 | fcalc | partial_15 | **0.566** | 0.780 | 0.47 | False | `vol_lt_1000` | 8 | 1.6 |
| 2200000 | fcalc | partial_30 | **0.755** | 0.767 | 0.32 | True | `vol_lt_1000` | 8 | 1.6 |
| 2200000 | fcalc | fragment_half | **0.700** | 0.753 | 0.41 | True | `vol_lt_1000` | 8 | 2.5 |
| 2200000 | fobs | auto | **0.279** | 0.780 | 0.50 | False | `vol_lt_1000` | 8 | 12.4 |
| 2200000 | fobs | partial_15 | **0.505** | 0.749 | 0.47 | False | `vol_lt_1000` | 8 | 1.1 |
| 2200000 | fobs | partial_30 | **0.661** | 0.775 | 0.46 | False | `vol_lt_1000` | 8 | 1.1 |
| 2200000 | fobs | fragment_half | **0.674** | 0.760 | 0.48 | False | `vol_lt_1000` | 8 | 1.3 |
| 2221836 | fcalc | auto | **0.146** | 0.791 | 0.62 | False | `vol_1000_3500` | 8 | 147.4 |
| 2221836 | fcalc | partial_15 | **0.644** | 0.796 | 0.61 | False | `vol_1000_3500` | 8 | 84.4 |
| 2221836 | fcalc | partial_30 | **0.793** | 0.784 | 0.58 | False | `vol_1000_3500` | 8 | 70.0 |
| 2221836 | fcalc | fragment_half | **0.720** | 0.720 | 0.57 | False | `vol_1000_3500` | 8 | 65.3 |
| 2221836 | fobs | auto | **0.172** | 0.685 | 0.61 | False | `vol_1000_3500` | 8 | 127.0 |
| 2221836 | fobs | partial_15 | **0.545** | 0.682 | 0.61 | False | `vol_1000_3500` | 8 | 95.4 |
| 2221836 | fobs | partial_30 | **0.714** | 0.702 | 0.57 | False | `vol_1000_3500` | 8 | 107.4 |
| 2221836 | fobs | fragment_half | **0.717** | 0.758 | 0.61 | False | `vol_1000_3500` | 8 | 73.2 |
| 2227862 | fcalc | auto | **0.151** | 0.784 | 0.58 | False | `vol_1000_3500` | 8 | 156.3 |
| 2227862 | fcalc | partial_15 | **0.658** | 0.792 | 0.51 | False | `vol_1000_3500` | 8 | 107.6 |
| 2227862 | fcalc | partial_30 | **0.832** | 0.769 | 0.41 | True | `vol_1000_3500` | 8 | 77.3 |
| 2227862 | fcalc | fragment_half | **0.789** | 0.719 | 0.49 | False | `vol_1000_3500` | 8 | 56.8 |
| 2227862 | fobs | auto | **0.151** | 0.678 | 0.58 | False | `vol_1000_3500` | 8 | 134.7 |
| 2227862 | fobs | partial_15 | **0.526** | 0.694 | 0.57 | False | `vol_1000_3500` | 8 | 87.3 |
| 2227862 | fobs | partial_30 | **0.710** | 0.696 | 0.52 | False | `vol_1000_3500` | 8 | 80.7 |
| 2227862 | fobs | fragment_half | **0.765** | 0.745 | 0.55 | False | `vol_1000_3500` | 8 | 58.5 |
| 2233297 | fcalc | auto | **0.255** | 0.790 | 0.59 | False | `vol_1000_3500` | 8 | 115.2 |
| 2233297 | fcalc | partial_15 | **0.658** | 0.798 | 0.55 | False | `vol_1000_3500` | 8 | 59.1 |
| 2233297 | fcalc | partial_30 | **0.805** | 0.782 | 0.50 | False | `vol_1000_3500` | 8 | 50.9 |
| 2233297 | fcalc | fragment_half | **0.751** | 0.719 | 0.53 | False | `vol_1000_3500` | 8 | 42.6 |
| 2233297 | fobs | auto | **0.237** | 0.751 | 0.60 | False | `vol_1000_3500` | 8 | 103.7 |
| 2233297 | fobs | partial_15 | **0.523** | 0.719 | 0.61 | False | `vol_1000_3500` | 8 | 69.8 |
| 2233297 | fobs | partial_30 | **0.714** | 0.757 | 0.53 | False | `vol_1000_3500` | 8 | 61.8 |
| 2233297 | fobs | fragment_half | **0.763** | 0.782 | 0.55 | False | `vol_1000_3500` | 8 | 46.5 |

## Summary by Vol band × run

- `vol_1000_3500/auto`: n=24 mean mapCC=**0.223** (median 0.185)
- `vol_1000_3500/fragment_half`: n=24 mean mapCC=**0.724** (median 0.723)
- `vol_1000_3500/partial_15`: n=24 mean mapCC=**0.587** (median 0.606)
- `vol_1000_3500/partial_30`: n=24 mean mapCC=**0.765** (median 0.780)
- `vol_gt_3500/auto`: n=2 mean mapCC=**0.081** (median 0.081)
- `vol_gt_3500/fragment_half`: n=2 mean mapCC=**0.493** (median 0.493)
- `vol_gt_3500/partial_15`: n=2 mean mapCC=**0.470** (median 0.470)
- `vol_gt_3500/partial_30`: n=2 mean mapCC=**0.684** (median 0.684)
- `vol_lt_1000/auto`: n=6 mean mapCC=**0.363** (median 0.287)
- `vol_lt_1000/fragment_half`: n=6 mean mapCC=**0.739** (median 0.744)
- `vol_lt_1000/partial_15`: n=6 mean mapCC=**0.572** (median 0.577)
- `vol_lt_1000/partial_30`: n=6 mean mapCC=**0.741** (median 0.736)

## Vol 1000–3500 Å³ focus (AI-PhaSeed hybrid-friendly band)

| Run/amp | n | mean mapCC | median |
|---------|---|------------|--------|
| `auto/fcalc` | 12 | **0.238** | 0.216 |
| `auto/fobs` | 12 | **0.209** | 0.169 |
| `fragment_half/fcalc` | 12 | **0.729** | 0.723 |
| `fragment_half/fobs` | 12 | **0.720** | 0.727 |
| `partial_15/fcalc` | 12 | **0.642** | 0.649 |
| `partial_15/fobs` | 12 | **0.531** | 0.530 |
| `partial_30/fcalc` | 12 | **0.809** | 0.812 |
| `partial_30/fobs` | 12 | **0.721** | 0.714 |

## Takeaways

- **auto** (ab initio) is typically weak; mapCC often ≪ 0.5 on hard cells.
- **partial_30** (oracle) is the Lane-B control for the ≥~30% strong-φ bar.
- **partial_15** often under-seeds vs that bar.
- **fragment_half** is the no-oracle path; on coherent half-models it should approach partial_30 mapCC (see also `cod_hard_path_validation.md`).
- Vol **1000–3500 Å³** is the Carrozzini / AI-PhaSeed hybrid-friendly band.
- Strict multi-criterion *solved* can fail on R1 under short budgets.
- Mid-band expansion (2026-09): ten additional light-atom CHNOF Fobs cells (~26 non-H). Original six rows are unchanged.
- Pooled Vol 1000–3500 (n=24): fragment_half mean mapCC 0.724, partial_30 0.765, auto 0.223. Fobs-only (n=12): fragment_half 0.720, partial_30 0.721, auto 0.209.
- Fobs `fragment_half` gates on this mid-band: mapCC ≥ 0.7 on 8/12, peak recovery ≥ 0.5 on 11/12, carbon-peak R1 ≤ 0.45 on 1/12; strict solves 1/12.
- `auto` phases the measured unique wedge without expanding it to P1, so its rows are not an ab initio test of the corrected charge flipping (see FOR_REVIEWERS C27).

Regenerate:
```bash
python scripts/run_cod_stratified_bench.py --dmin 1.0
```
