# SHELXD head-to-head

## Goal

Compare **gps-solve** classical and hybrid methods against the academic standard
small-molecule dual-space tool **SHELXD** (Sheldrick), on shared synthetic and
COD Fcalc controls, with honest metrics.

## What is / is not included

| Component | Status |
|-----------|--------|
| Real `shelxd` binary | **External** — not redistributed (academic license) |
| SHELX `.ins` / `.hkl` writers | ✅ `io/shelx_write.py` |
| Subprocess runner + `.res` parser | ✅ `solvers/shelxd_runner.py` |
| Educational dual-space baseline | ✅ `solvers/dual_space.py` (SHELXD-**inspired**, not SHELXD) |
| Scoreboard script | ✅ `scripts/run_shelxd_h2h.py` |

## Install SHELXD (optional)

1. Register and download from [https://shelx.uni-goettingen.de/](https://shelx.uni-goettingen.de/)
2. Place `shelxd` on your `PATH`, or:

```bash
export SHELXD=/path/to/shelxd
# optional:
export SHELX_BIN=/path/to/shelx/bin
```

3. Re-run:

```bash
python scripts/run_shelxd_h2h.py
# → data/processed/shelxd_h2h.{json,md}
```

Without the binary, the script still runs all **in-repo** methods plus
`dual_space`; `shelxd` rows are omitted (or error if forced).

## Dual-space baseline (in-repo)

Algorithm sketch (multi-start):

1. Random phases (or prior)
2. Density FFT → pick top $N$ peaks
3. $F_\mathrm{calc}$ from equal-atom model at peaks
4. Blend phases; re-impose $|F_\mathrm{obs}|$
5. Score starts by free FOM − partial $R$; light CF polish

This captures the **spirit** of Sheldrick dual-space recycling for transparent
benchmarking. It is **not** a drop-in SHELXD replacement (no E-selection
engine, Patterson seeding fidelity, space-group machinery, or CFOM ranking of
the commercial/academic binary).

## Metrics

On each case (same structure for every method):

- mapCC_OI (origin-invariant)
- peak recovery, $R_1$, strict `solved` (`SuccessThresholds`)
- free FOM (truth-free)
- wall time

## CLI methods

```bash
gps-solve --hkl data.hkl --ins data.ins --method dual_space --out out/
gps-solve --hkl data.hkl --ins data.ins --method shelxd --out out/   # needs binary
gps-solve --hkl data.hkl --ins data.ins --method shelxd_or_dual --out out/
```

## Scope statement

This head-to-head supports **honest baselines** for an open research framework.
It does not claim that gps-solve supersedes SHELXD/SHELXT for production small-
molecule structure solution.
