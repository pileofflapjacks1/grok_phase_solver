# COD stratified bench (v0.10 skeleton)

Local COD stratified skeleton. Not a 1505-structure Carrozzini panel. partial_30 is oracle control; auto is ab initio.

| Dataset | Run | mapCC | Vol band | max Z | HA |
|---------|-----|-------|----------|-------|----|
| 2016452 | auto | **0.439** | vol_lt_1000 | 8 | False |
| 2016452 | partial_30 | **0.782** | vol_lt_1000 | 8 | False |
| 2017775 | auto | **0.044** | vol_gt_3500 | 6 | False |
| 2017775 | partial_30 | **0.728** | vol_gt_3500 | 6 | False |
| 2100301 | auto | **0.300** | vol_lt_1000 | 8 | False |
| 2100301 | partial_30 | **0.769** | vol_lt_1000 | 8 | False |

## Summary by band/run

- `vol_gt_3500/auto`: n=1 mean mapCC=0.044
- `vol_gt_3500/partial_30`: n=1 mean mapCC=0.728
- `vol_lt_1000/auto`: n=2 mean mapCC=0.369
- `vol_lt_1000/partial_30`: n=2 mean mapCC=0.776
