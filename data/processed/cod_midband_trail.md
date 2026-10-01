# COD mid-band Fobs fragment trail (this run)

Scientist path on in-repo **2012000** experimental Fobs + heaviest-cluster
~½ non-H fragment. **Do not invent mapCC** — free FOM is from this run;
mapCC / R1 / strict are copied from the committed Vol-band scoreboard.

- Fragment CIF: `out_2012000_frag/2012000_half.cif` (13 of 27 non-H)
- Solve out: `out_2012000_frag`
- Seconds: 2.6

## This run (truth-free)

| Field | Value |
|-------|-------|
| method | `partial_phaseed` |
| free FOM (rank only) | **0.786** |
| n Q peaks written | 26 |
| next-action id | `olex2_handbuild` |
| space group | `P 1 21 1` |

`trial.res` is a Q-peak list. Open `olex2_handbuild.md` in the `--out`
folder: **File → Open** then **View → Work → Info** for Z / Z′.
Do not SHELXL the raw Q list.

## Committed scoreboard (copied, not recomputed)

Source: `data/processed/cod_stratified_bench.md`.

| Amp | Run | mapCC_OI | free FOM | R1 | strict |
|-----|-----|----------|----------|----|--------|
| fobs | fragment_half | **0.675** | 0.776 | 0.53 | **False** |

Strict success = mapCC_OI ≥ 0.7 AND peak recovery ≥ 0.5 AND R1 ≤ 0.45. This Fobs fragment_half row is not a strict solve (mapCC 0.675, R1 0.53). Living mid-band panel (12 structures, Fobs+Fcalc pooled n=24): fragment_half mean mapCC ~0.72, partial_30 ~0.74, auto ~0.16. Fobs-only n=12: fragment_half ~0.72, partial_30 ~0.71, auto ~0.18. The v0.13.1 freeze was the n=4 pilot (~0.71 vs auto ~0.27).

## gps-solve equivalent

```bash
gps-solve --hkl data/raw/cod/2012000.hkl \
  --predicted-model out_2012000_frag/2012000_half.cif \
  --method partial_phaseed --dmin 1.0 \
  --n-iter 90 --n-extend 26 --n-starts 1 \
  --prior-weight 0.52 --dm-ai-weight 0.42 --no-uncertainty \
  --out out_2012000_frag
```

Recipe: [`docs/examples/cod_midband_fragment_trail.md`](../../docs/examples/cod_midband_fragment_trail.md).
