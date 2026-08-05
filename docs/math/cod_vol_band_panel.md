# COD Vol-band experimental panel

## Purpose

Stratified evaluation of ab initio vs partial-φ / fragment seeding on **local
COD** entries, binned by unit-cell volume:

| Band | Role |
|------|------|
| Vol &lt; 1000 Å³ | Small-cell / high-res-ish controls |
| **Vol 1000–3500 Å³** | Carrozzini / AI-PhaSeed hybrid-friendly band |
| Vol &gt; 3500 Å³ | Large / hard ab initio control |

## How to run

```bash
# ensure CIF+HKL under data/raw/cod/ (see data/cod.py COD_SAMPLE_IDS)
python scripts/run_cod_stratified_bench.py --dmin 1.0
```

Writes `data/processed/cod_stratified_bench.{json,md}`.

## Methods

| Run | Meaning |
|-----|---------|
| `auto` | Ab initio path (ensemble / prior / CF) |
| `partial_15` | Oracle 15% strong-\|E\| phases |
| `partial_30` | Oracle 30% strong-\|E\| phases (Lane B bar) |
| `fragment_half` | ~½ non-H ASU (heaviest cluster) + full Fcalc soft prior |

Each dataset is run with **Fcalc** control and **Fobs** (when HKL present).
mapCC uses deposited-model Fcalc phases as proxy truth.

## Honest limits

- Small local set — **not** a 1505-COD Carrozzini replication.
- Short budgets: strict multi-criterion “solved” may fail on R1 even when mapCC is high.
- fragment_half uses deposited-structure fragment knowledge (realistic for MR-like / predicted-model workflows, not pure ab initio).
