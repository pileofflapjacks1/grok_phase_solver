# SHELXS head-to-head

## Goal

Compare **gps-solve** classical/hybrid methods against **SHELXS** (Sheldrick
direct-methods structure solution) on shared synthetic cases with fair mapCC /
strict success metrics.

## Binary (not redistributed)

Place the academic macOS/Linux binary at:

```text
ShelX/shelxs
```

or:

```bash
export SHELXS=/path/to/shelxs
```

Download: [https://shelx.uni-goettingen.de/](https://shelx.uni-goettingen.de/)

```bash
# macOS: allow downloaded binary to run
xattr -dr com.apple.quarantine ShelX
chmod +x ShelX/shelxs
```

`ShelX/` is **gitignored** (license: do not push proprietary binaries).

## Pipeline

1. Write fixed-format HKLF-4 `.hkl` (`3i4,2f8.2`, intensities scaled)
2. Write minimal `.ins` (`LATT -1`, `SFAC C`, `UNIT n`, `TREF ntry`)
3. Run `shelxs job`
4. Parse Q-peaks from `.res` → equal-atom $F_\mathrm{calc}$ phases
5. Score mapCC_OI / peak recovery / strict solved vs truth

Code: `solvers/shelxs_runner.py`, `scripts/run_shelxs_h2h.py`

## Run

```bash
python scripts/run_shelxs_h2h.py
# → data/processed/shelxs_h2h.{json,md}

gps-solve --hkl data.hkl --ins data.ins --method shelxs --out out/
```

## SHELXS vs SHELXD

| Tool | Role |
|------|------|
| **SHELXS** | Classic direct methods / multi-trial tangent (this H2H) |
| **SHELXD** | Dual-space (optional separate runner; not required) |
| **SHELXE** | Density modification / phase extension |
| **SHELXL** | Least-squares refinement (after `trial.res`) |

## Scope

SHELXS is the gold-standard **small-molecule ab initio** baseline. This harness
does not claim gps-solve replaces SHELXS on easy organics; it measures when
open methods are competitive and when they fail (hard region).
