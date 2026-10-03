# COD Vol-band experimental panel

**Paper:** Figure 6 of `docs/arxiv_draft.md` · claim **C25** · software freeze **v0.13.1**.

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

## Living scoreboard (2026-09)

`data/processed/cod_stratified_bench.md` is the current panel: 16 local COD
structures (12 in Vol 1000–3500 Å³), d_min = 1.0 Å, 128 OK rows.

Pooled Fobs+Fcalc mid-band means (n=24): fragment_half mapCC **~0.72**,
partial_30 **~0.74**, auto **~0.16**. Fobs-only (n=12): fragment_half
**~0.72**, partial_30 **~0.71**, auto **~0.18**. fragment_half does not
beat partial_30 on the pooled table. No Fobs fragment_half row is a strict
solve. COD **1544230** Fobs fragment_half scores mapCC 0.668, peak recovery
1.000, and carbon-peak R1 0.562 after the shared d_min slack. It is not a
strict solve.

Figure 6 and `docs/arxiv_draft.md` stay the v0.13.1 freeze: the earlier n=4
mid-band pilot (fragment_half ~0.71 vs auto ~0.27). That table is not the
living scoreboard.

## R1 gate (Fobs fragment_half)

`data/processed/r1_gate_diagnostic.md` splits the 12 mid-band Fobs
`fragment_half` rows by gate. mapCC ≥ 0.7 on 8/12. Peak recovery ≥ 0.5 on
12/12. Carbon-peak R1 ≤ 0.45 on 0/12. Assigning deposited element types to
those same peaks leaves every side R1 above 0.45 (minimum 0.477). `solved`
is still the carbon-peak definition.

COD 1544230 stays in the mapCC mean. The bench and the solver share
`d_min_keep_mask` (d ≥ d_min − 1e-9). Fobs fragment_half mapCC moved from
0.669 to 0.668 when that cell was rescored; carbon-peak R1 is 0.562 and
peak recovery is 1.000. Other panel cells have no reflection in the slack,
so their rows were not rerun.
