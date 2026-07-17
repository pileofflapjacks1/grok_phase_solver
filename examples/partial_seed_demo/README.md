# Partial-φ demo (hard-ish synthetic)

Demonstrates the **hard-data path** that works when ab initio does not:
provide ~30% correct strong-|E| phases → AI-PhaSeed extension.

## Files

| File | Role |
|------|------|
| `demo_hard.hkl` / `demo_hard.ins` | Synthetic hard-ish cell (n≈12, d_min=1.5 Å) |
| `known_phases_30pct.csv` | Oracle top-30% strong\|E\| phases (degrees) |
| `known_phases_15pct.csv` | Weaker 15% seed (often insufficient) |
| `truth.npz` | Ground truth for developers (not needed for gps-solve) |

## Solve with partial phases

```bash
# From repo root
gps-solve --hkl examples/partial_seed_demo/demo_hard.hkl \
  --ins examples/partial_seed_demo/demo_hard.ins \
  --method partial_phaseed \
  --phase-seed-csv examples/partial_seed_demo/known_phases_30pct.csv \
  --out examples/partial_seed_demo/out_30

# Compare: ab initio auto (likely fails strict solve on hard)
gps-solve --hkl examples/partial_seed_demo/demo_hard.hkl \
  --ins examples/partial_seed_demo/demo_hard.ins \
  --method auto --out examples/partial_seed_demo/out_auto
```

## Decision rule

- **Easy / high-res** → `auto` (ensemble)
- **Hard ab initio fails** → partial-φ from HA/MAD/MR-lite/SHELXS fragment
- **After peaks** → refine with SHELXL (`trial.res` or full workflow with ShelX/shelxl)

See `docs/USER_GUIDE.md` § decision tree.
