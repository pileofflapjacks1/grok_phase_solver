# Release notes — v0.13.1

**Theme:** COD Vol-band experimental panel on the release tag (patch over 0.13.0).

## What’s in 0.13.1

All of **v0.13.0** (GraphPhaseNet v11, CrystalX typing, XDXD research path), plus:

### COD Vol-band stratified panel
- Expanded `scripts/run_cod_stratified_bench.py`: **Fobs + Fcalc**; auto / partial_15 / partial_30 / fragment_half
- Six local COD sets spanning Vol **&lt;1000 / 1000–3500 / &gt;3500 Å³**
- New mid-band HKL+CIF: **2012000**, **2013000** (+ small control **2200000**)
- Scoreboard: `data/processed/cod_stratified_bench.{md,json}`
- Math note: `docs/math/cod_vol_band_panel.md`

### Headline numbers (mid-band Vol 1000–3500, mean mapCC, Fobs+Fcalc pooled)
| Run | mean mapCC |
|-----|------------|
| auto | ~0.27 |
| partial_15 | ~0.54 |
| partial_30 | ~0.70 |
| fragment_half | ~0.71 |

**Honest:** small local panel, not Carrozzini’s 1505-structure set. Partial information remains the practical hard path.

## Install

```bash
python -m pip install -U "grok-phase-solver>=0.13.1"
gps-solve --help

# regenerate panel
python scripts/run_cod_stratified_bench.py --dmin 1.0
```

## Full 0.13.0 feature list
See [RELEASE_NOTES_v0.13.0.md](RELEASE_NOTES_v0.13.0.md).
