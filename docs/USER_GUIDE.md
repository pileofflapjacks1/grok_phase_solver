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
| **`auto` (default)** | **Easy/high-res → ensemble**; hard → graph/hard-P1 prior or CF; P2₁/c+PhAI → AI-PhaSeed |
| `ensemble` | Best open ab initio on easy data (CF+RAAR multistart, free-FOM pick) |
| `charge_flipping` | Fast classical baseline |
| **`partial_phaseed`** | **Hard-data path** when you have partial phases (HA/MAD/MR-lite) — see below |
| `shelxs` / `shelxs+shelxe` | External academic SHELXS (± SHELXE density mod); needs `ShelX/` binaries |
| `strong_prior_phaseed` / `hard_p1_phaseed` | Learned priors + AI-PhaSeed (hard synthetic domain) |
| `phai_phaseed` / `phai+cf_cond` | PhAI seed hybrids (needs weights) |
| `raar` / `recycle` / `hio` / `direct_methods` | Projection / educational DM |

Examples:

```bash
gps-solve --hkl data.hkl --ins data.ins --method auto --n-iter 150 --out out_auto
gps-solve --hkl data.hkl --ins data.ins --method ensemble --n-starts 4 --out out_ens
gps-solve --hkl data.hkl --ins data.ins --method shelxs+shelxe --out out_shelx
```

### Decision tree (what to run)

```text
Have partial phases (HA / MAD / MR / SHELXS fragment)?
   YES →  --method partial_phaseed --phase-seed-csv known.csv
   NO  →  resolution good (d ≲ 1.15 Å)?
            YES → auto (ensemble)
            NO  → auto (prior/CF) — if map fails, get partial φ or try shelxs
Finish → trial.res → SHELXL / Olex2
```

| Situation | Command |
|-----------|---------|
| Default | `gps-solve --hkl … --ins … --method auto` |
| Easy / high-res | `auto` or `ensemble` |
| Hard, pure ab initio | `auto` (expect struggle; see free FOM) |
| **Hard + known φ / HA** | `partial_phaseed` + `--phase-seed-csv` |
| External classical | `shelxs` or `shelxs+shelxe` |
| After any solve | `trial.res` → **SHELXL** |

### Partial-φ hard path (recommended when ab initio fails)

Oracle benchmarks: **≥ ~30% correct strong \|E\| phases (≲20° error)** → hard cells can strict-solve via AI-PhaSeed. Full ab initio priors still sit ~20% within 20°.

```bash
# CSV: h,k,l,phase_deg  (strong reflections you know)
gps-solve --hkl data.hkl --ins data.ins \
  --method partial_phaseed \
  --phase-seed-csv known_phases.csv \
  --out ./out_partial
```

**Packaged demo** (synthetic hard-ish + 30% oracle seed):

```bash
gps-solve --hkl examples/partial_seed_demo/demo_hard.hkl \
  --ins examples/partial_seed_demo/demo_hard.ins \
  --method partial_phaseed \
  --phase-seed-csv examples/partial_seed_demo/known_phases_30pct.csv \
  --out examples/partial_seed_demo/out_30
```

### SHELXS → SHELXE → SHELXL (local academic binaries)

Place binaries in `ShelX/` (gitignored — do not push):

```bash
# Solve with SHELXS, density-mod with SHELXE
gps-solve --hkl data.hkl --ins data.ins --method shelxs+shelxe --out out_sx

# Refine trial model with SHELXL (outside gps-solve)
cp out_sx/trial.res work.ins
cp data.hkl work.hkl
ShelX/shelxl work
```

macOS: `xattr -dr com.apple.quarantine ShelX && chmod +x ShelX/*`

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
