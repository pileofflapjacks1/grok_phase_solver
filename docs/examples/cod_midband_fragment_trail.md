# Lab-style trail: COD mid-band Fobs + fragment seed → Olex2

**Information source:** fragment  
**Panel:** local COD Vol-band (C25), not Carrozzini 1505.  
**Do not invent metrics.** mapCC / R1 / strict below are copied from
[`data/processed/cod_stratified_bench.md`](../../data/processed/cod_stratified_bench.md).
This-run free FOM lives in
[`data/processed/cod_midband_trail.md`](../../data/processed/cod_midband_trail.md)
after `scripts/run_cod_midband_trail.py`.

This checkout already has CIF + HKL under `data/raw/cod/`. A new COD
download is not required.

`trial.res` is a **Q-peak list** for Olex2 hand-build. It is **not** a
SHELXL starting model. `--retry-with-peaks` is **peaks-as-carbon, not a
fragment.**

## Cell

| COD | Vol (Å³) | Band | max Z | n non-H | SG | HKL |
|-----|----------|------|-------|---------|----|-----|
| 2012000 | 1027 | `vol_1000_3500` | 8 | 27 | P 1 21 1 | yes |

(2016452 / 2100301 / 2200000 are `vol_lt_1000`; 2017775 is `vol_gt_3500`.
The living mid-band is 12 structures. This trail stays on 2012000.)

## Commands (working)

Preferred — writes the heaviest-cluster ~½ non-H CIF (same selection as the
Vol-band bench), runs `partial_phaseed`, and exports `trial.res` +
`olex2_handbuild.md`:

```bash
python scripts/run_cod_midband_trail.py --out ./out_2012000_frag
```

Equivalent `gps-solve` after the runner has written the half-CIF (or after
`--dry-run`):

```bash
python scripts/run_cod_midband_trail.py --dry-run --out ./out_2012000_frag

gps-solve --hkl data/raw/cod/2012000.hkl \
  --predicted-model ./out_2012000_frag/2012000_half.cif \
  --method partial_phaseed --dmin 1.0 \
  --n-iter 90 --n-extend 26 --n-starts 1 \
  --prior-weight 0.52 --dm-ai-weight 0.42 --no-uncertainty \
  --out ./out_2012000_frag
```

The CIF-style HKL already carries cell + SG (`P 1 21 1`,
7.5950, 9.9731, 13.7266, 90, 98.967, 90). `--ins` is not required.

`--predicted-model` on the **full** deposited CIF is a larger seed than
this trail; the product path here is the **half-model** fragment.

Stratified panel (oracle + fragment, writes mapCC — not the scientist path):

```bash
python scripts/run_cod_stratified_bench.py --ids 2012000
```

## Olex2 hand-build (after gps-solve)

1. In the `--out` folder, open **`olex2_handbuild.md`**.
2. Olex2: **File → Open** `trial.res`. Header should read
   `TITL gps-solve hand-build peaks (not a SHELXL start)`.
3. **View → Work → Info** for formula / **Z** / **Z′** / polymeric packing.
   There is no big polymeric banner. `UNIT 1` is a dummy.
4. Peak-pick / assign C, N, O from chemistry. Delete junk Q peaks.
   **Do not SHELXL the raw Q list.**
5. After a chemically plausible molecule is built:

```bash
cp built.res work.ins
cp data/raw/cod/2012000.hkl work.hkl
ShelX/shelxl work
```

Free FOM ranks trials. Chemical sense + SHELXL R1 decide.

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
(mapCC 0.675, R1 0.53). On the living 12-structure mid-band
(Fobs+Fcalc pooled, n=24) fragment_half mean mapCC is **~0.72**,
partial_30 **~0.74**, and auto **~0.16**. Fobs-only (n=12): fragment_half
**~0.72**, partial_30 **~0.71**, auto **~0.18**. The v0.13.1 freeze was
the n=4 pilot (~0.71 vs auto ~0.27). This 2012000 row is unchanged.

Fcalc control on the same cell: fragment_half mapCC 0.677 (R1 0.50, False);
partial_30 mapCC 0.714 (R1 0.36, True).

This trail does **not** recompute those mapCC numbers. Compare this-run
free FOM in `data/processed/cod_midband_trail.md` against the 0.776
scoreboard FOM as a ranking check only.

## Output paths

- Runner `--out` (default `./out_2012000_frag/`): `trial.res`,
  `olex2_handbuild.md`, `report.md`, `density.map`, `2012000_half.cif`
- `data/processed/cod_midband_trail.{json,md}` — this-run free FOM + copied
  scoreboard
