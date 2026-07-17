# Partial-φ demo (hard-ish synthetic)

Demonstrates the **hard-data path** that works when ab initio does not:
provide a partial phase / fragment seed → AI-PhaSeed extension.

## Files

| File | Role |
|------|------|
| `demo_hard.hkl` / `demo_hard.ins` | Synthetic hard-ish cell (n≈12, d_min=1.5 Å) |
| `known_phases_30pct.csv` | Oracle top-30% strong\|E\| phases (degrees) |
| `known_phases_15pct.csv` | Weaker 15% seed (often insufficient) |
| `truth.npz` | Ground truth for developers (not needed for gps-solve) |

## 1. Solve with known phases (oracle)

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

## 2. Fragment path (SHELX-style .res → Fcalc seed)

If you have a partial model or SHELXS Q-peaks in a `.res` file:

```bash
# Build seed CSV only
gps-make-seed --hkl examples/partial_seed_demo/demo_hard.hkl \
  --ins examples/partial_seed_demo/demo_hard.ins \
  --from-res /path/to/fragment.res \
  -o examples/partial_seed_demo/seed_from_res.csv

# Or one-shot
gps-solve --hkl examples/partial_seed_demo/demo_hard.hkl \
  --ins examples/partial_seed_demo/demo_hard.ins \
  --method partial_phaseed \
  --phase-seed-res /path/to/fragment.res \
  --out examples/partial_seed_demo/out_frag
```

## 3. Recycle peaks from a failed ab initio run

```bash
gps-solve --hkl examples/partial_seed_demo/demo_hard.hkl \
  --ins examples/partial_seed_demo/demo_hard.ins \
  --method auto --out examples/partial_seed_demo/out_auto

gps-solve --hkl examples/partial_seed_demo/demo_hard.hkl \
  --ins examples/partial_seed_demo/demo_hard.ins \
  --method partial_phaseed \
  --seed-peaks-csv examples/partial_seed_demo/out_auto/peaks.csv \
  --seed-n-atoms 8 \
  --out examples/partial_seed_demo/out_from_peaks
```

## Decision rule

- **Easy / high-res** → `auto` (ensemble)
- **Hard ab initio fails** → partial-φ from HA / MAD / MR-lite / SHELXS fragment / peaks
- **After peaks** → refine with SHELXL (`trial.res`)

Check **`report.md` → Partial seed quality**: strong-|E| coverage vs the ~30% bar.

See `docs/USER_GUIDE.md` § decision tree.
