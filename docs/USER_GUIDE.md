# User guide — solve your crystal structure

**Goal:** clone this repo, install once, feed experimental diffraction data, get phased structure factors and a density map to inspect / refine elsewhere.

---

## 1. Install (once)

```bash
git clone https://github.com/pileofflapjacks1/grok_phase_solver.git
cd grok_phase_solver
python -m pip install -e .
```

Optional (PhAI neural phasing for many P2₁/c organics):

```bash
python -m pip install -e ".[ml]"
# + download weights — see third_party/phai/README.md
```

Check:

```bash
gps-solve --help
```

---

## 2. What you need from the experiment

| Item | Required? | Notes |
|------|-----------|--------|
| Reflection file | **Yes** | SHELX `.hkl` (HKLF 4), COD-style HKL CIF, or `.mtz` |
| Unit cell | **Yes** | From `.ins` `CELL` line, or `--cell a,b,c,α,β,γ` |
| Space group | Strongly recommended | From `.ins` / known SG, e.g. `--sg "P 1 21/c 1"` |
| Composition | Optional | Helps interpretation of peaks (not strictly required for CF) |

Typical small-molecule pair from Olex2/SHELXTL:

```text
mycrystal.hkl
mycrystal.ins   # must contain CELL (and ideally SYMM/LATT)
```

---

## 3. One-command solve

### Recommended (SHELX workflow)

```bash
gps-solve --hkl mycrystal.hkl --ins mycrystal.ins --out ./solve_out
```

### Without .ins (cell + SG on the command line)

```bash
gps-solve \
  --hkl mycrystal.hkl \
  --cell 9.748,8.890,7.566,90,112.74,90 \
  --sg "P 1 21/c 1" \
  --out ./solve_out
```

### Method choices

| `--method` | When to use |
|------------|-------------|
| `auto` (default) | Smart pick: AI-PhaSeed if P2₁/c+PhAI; ensemble at high res; else CF |
| `charge_flipping` | Robust classical default |
| `ensemble` | Multistart CF+RAAR ranked by free FOM (good high-res) |
| `phai_phaseed` | **AI-PhaSeed**: PhAI seed → phase extension → free-FOM polish |
| `phai+cf_cond` | PhAI + CF only if free FOM accepts polish |
| `phai` / `phai+cf` | PhAI alone or unconditional CF polish (needs weights) |
| `hard_p1_phaseed` | Domain-matched hard-P1 prior + AI-PhaSeed (synthetic-trained) |
| `strong_prior_phaseed` | Triplet-graph GNN prior + AI-PhaSeed (hard multi-SG) |
| `raar` / `recycle` / `hio` | Projection / positivity variants |
| `direct_methods` | Educational multi-start triplets (not SHELXD-strength) |

Examples:

```bash
gps-solve --hkl data.hkl --ins data.ins --method auto --n-iter 150 --out out_auto
gps-solve --hkl data.hkl --ins data.ins --method phai_phaseed --out out_phaseed
gps-solve --hkl data.hkl --ins data.ins --method ensemble --n-starts 4 --out out_ens
```

`auto` policy (simplified): P2₁/c + PhAI weights → `phai_phaseed`; high resolution → `ensemble`; else `charge_flipping`. Free-FOM composite is written into diagnostics.

---

## 4. What you get (`--out` folder)

| File | Use |
|------|-----|
| **`report.md`** | Summary, free FOM, warnings, next steps |
| **`phases.csv`** | h, k, l, \|F\|, phase (°), A, B |
| **`structure_factors.F`** | Complex F for custom tools |
| **`density.npz`** | Electron density grid + cell |
| **`density_slice.png`** | Quick visual check |
| **`peaks.csv` / `peaks.xyz`** | Strongest density maxima (trial atoms) |
| **`trial.res`** | SHELXL-style trial model (Q/C peaks) for Olex2 |
| **`solve_summary.json`** | Machine-readable log |

---

## 5. After gps-solve (important)

This tool **phases** data and suggests **density peaks**. It does **not** replace:

- **SHELXL** / **Olex2** refinement  
- **SHELXT/SHELXD** as the industrial small-molecule solver  
- **Phenix** / experimental phasing for proteins  

**Typical path:**

1. Open `density_slice.png` and `peaks.csv`.  
2. Load **`trial.res`** in Olex2 / ShelXle (or `peaks.xyz` in PyMOL).  
3. Assign C/N/O/… from chemistry and residual maps.  
4. Refine against your intensities with SHELXL (`ACTA`, anisotropic ADPs, H-atoms, etc.).

If the map is uninterpretable: check cell/SG, try `--method ensemble` or `phai_phaseed`, improve resolution/completeness, or use classical SHELXD / experimental phasing.

---

## 6. Demo (no lab data required)

```bash
# From repo root — uses bundled sample-style data
gps-solve \
  --hkl examples/demo_solve/demo.hkl \
  --ins examples/demo_solve/demo.ins \
  --method charge_flipping \
  --n-iter 80 \
  --out examples/demo_solve/out
```

Then open `examples/demo_solve/out/report.md`.

---

## 7. Python API

```python
from grok_phase_solver.pipeline import solve_structure, export_solution
from grok_phase_solver.pipeline.solve import SolveConfig

result = solve_structure(
    "mycrystal.hkl",
    ins_path="mycrystal.ins",
    config=SolveConfig(method="charge_flipping", n_iter=150),
)
export_solution(result, "solve_out")
print(result.method, len(result.peaks))
```

---

## 8. Limitations (please read)

- Best for **small-molecule** crystals, ideally **d_min ≲ 1.2 Å**.  
- **Not** a general protein ab initio solver.  
- Success is **not guaranteed**; always validate with refinement R-factors.  
- Origin/enantiomorph of the map may need fixing in refinement.  
- PhAI path is specialized (P2₁/c, fixed reciprocal grid); CF is the general default.

---

## 9. Getting help

- Math / methods: `docs/math/`  
- Scoreboard (method comparison): `data/processed/scoreboard.md`  
- Issues: GitHub repository issues  

When filing a bug, attach (if allowed) cell, SG, resolution, completeness, and whether CF/PhAI was used — not necessarily proprietary coordinates.
