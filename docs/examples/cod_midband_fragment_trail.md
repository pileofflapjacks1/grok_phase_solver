# Lab-style trail: COD mid-band Fobs + fragment seed

**Information source:** fragment  
**Panel:** local COD Vol-band (C25), not Carrozzini 1505.  
**Do not invent metrics** — numbers below are copied from the committed scoreboard
[`data/processed/cod_stratified_bench.md`](../../data/processed/cod_stratified_bench.md).

This checkout already has CIF + HKL for the stratified catalog under
`data/raw/cod/`. A new COD download is not required.

## Cell

| COD | Vol (Å³) | Band | max Z | n non-H | SG | HKL |
|-----|----------|------|-------|---------|----|-----|
| 2012000 | 1027 | `vol_1000_3500` | 8 | 27 | P1211 | yes |

(2016452 / 2100301 are in the same catalog but `vol_lt_1000`. Mid-band on this
panel is 2012000 and 2013000.)

## Commands (recipe)

Same path the validation scripts already run. Prefer the committed runner
over ad-hoc one-offs:

```bash
# Stratified Vol-band panel (auto / partial_15 / partial_30 / fragment_half)
python scripts/run_cod_stratified_bench.py --ids 2012000

# Older hard-path Fobs validation (oracle 15/30 + fragment)
python scripts/run_cod_hard_path_validation.py
```

Scientist-facing equivalent (after the bench has written a fragment CIF / seed):

```bash
# Fragment / predicted-model seed on experimental Fobs
gps-make-seed --hkl data/raw/cod/2012000.hkl --from-cif data/raw/cod/2012000.cif \
  -o /tmp/2012000_frag_seed.csv
gps-solve --hkl data/raw/cod/2012000.hkl \
  --predicted-model /tmp/2012000_half.cif \
  --method partial_phaseed --out ./out_2012000_frag

# Or: --phase-seed-res fragment.res  /  --phase-seed-csv seed.csv
```

`--retry-with-peaks` is **peaks-as-carbon, not a fragment.**

## Committed scoreboard (2012000 Fobs)

From `data/processed/cod_stratified_bench.md` (d_min = 1.0 Å):

| Amp | Run | mapCC_OI | free FOM | R1 | strict success |
|-----|-----|----------|----------|----|----------------|
| fobs | auto | 0.181 | 0.693 | 0.54 | False |
| fobs | partial_15 | 0.466 | 0.716 | 0.51 | False |
| fobs | partial_30 | 0.644 | 0.743 | 0.44 | False |
| fobs | **fragment_half** | **0.675** | 0.776 | 0.53 | **False** |

Peak recovery is not a column on that markdown table; do not invent it.
Strict success remains mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5 **and**
R1 ≤ 0.45. fragment_half on this Fobs row is **not** a strict solve
(mapCC 0.675, R1 0.53). Mid-band **mean** fragment_half mapCC across the
pooled Fobs+Fcalc rows is **~0.71** vs auto **~0.27** (C25).

Fcalc control on the same cell: fragment_half mapCC 0.677 (R1 0.50, False);
partial_30 mapCC 0.714 (R1 0.36, True).

## Output paths (when you re-run)

- `data/processed/cod_stratified_bench.{json,md}`
- per-run gps-solve folders under `--out` (`report.md`, `trial.res`, `density.map`)

`trial.res` is Q-peaks for Olex2 hand-build, not a SHELXL starting model,
unless an explicit research typing flag is on.

## SHELXL next action

Inspect `trial.res` / `density_slice.png` in Olex2, assign elements from
chemistry, then refine with SHELXL. Free FOM (0.776 on this Fobs fragment_half
row) ranks the trial only; it does not prove the structure. R1 on the
unrefined peak model is 0.53, so this is still a **hand-build / enlarge-fragment**
step, not a polished SHELXL finish.
