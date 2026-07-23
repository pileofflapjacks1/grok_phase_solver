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

Optional **GUI** (Streamlit web UI):

```bash
python -m pip install -e ".[gui]"
gps-gui
# opens http://localhost:8501 — upload HKL/INS, run phasing, download trial.res
```

Check:

```bash
gps-solve --help
gps-gui --help   # launches Streamlit; use Ctrl+C to stop
```

---

## 1b. Graphical interface (optional)

The GUI is a thin front end on the same `gps-solve` pipeline:

| Feature | Notes |
|---------|--------|
| Upload HKL + INS | Or type cell / space group |
| Method menu | `auto`, `ensemble`, `partial_phaseed`, PhAI, SHELXS, … |
| Seed uploads | Phase CSV, fragment `.res`, `peaks.csv` |
| Packaged demos | Easy ensemble; hard + 30% φ; hard + fragment |
| Results | Free FOM, density slice, peaks table, full `report.md` |
| Downloads | `trial.res`, CSV, zip of the export folder |

```bash
python -m pip install -e ".[gui]"
gps-gui
# or:  python -m grok_phase_solver.gui
# or:  python scripts/run_gui.py
```

Headless / CI still use the CLI. The GUI does not replace SHELXL refinement.

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
| `diffusion_hybrid` / `diffusion_phaseed` | **Experimental** Langevin phase completion (v0.5) |
| Predicted model seed | `--predicted-model model.cif` → partial_phaseed (AF/OpenFold3/Boltz) |
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
| **Hard + known φ / fragment / HA** | `partial_phaseed` + seed source (below) |
| External classical | `shelxs` or `shelxs+shelxe` |
| After any solve | `trial.res` → **SHELXL** |

### Partial-φ hard path (recommended when ab initio fails)

Oracle benchmarks: **≥ ~30% correct strong \|E\| phases (≲20° error)** → hard cells can strict-solve via AI-PhaSeed. Full ab initio priors still sit ~**21%** within 20° (even after scale-up).

**Any seed source works** — if you pass one of these with `--method auto`, gps-solve switches to `partial_phaseed`:

| Seed source | Flag | Typical origin |
|-------------|------|----------------|
| Known phases | `--phase-seed-csv known.csv` | MAD/MR / manual φ (`h,k,l,phase_deg`) |
| SHELXS / fragment model | `--phase-seed-res model.res` | Q-peaks or partial atoms → Fcalc |
| Density peaks | `--seed-peaks-csv peaks.csv` | Prior gps-solve peaks as light atoms |
| Explicit fragment | `--seed-atoms-csv atoms.csv` | `x,y,z,element` fractional |
| Structure-prediction model | `gps-make-seed --from-cif model.cif` **or** `--predicted-model model.cif` | AF/OpenFold3/Boltz/RF CIF → Fcalc seed (+ SG expand) |
| Isomorphous HA | `--native-hkl` + `--derivative-hkl` | Difference Patterson → HA sites |
| Single-dataset HA | `--patterson-ha` | Weak Patterson heuristic (HA present) |

```bash
# 1) Known phases
gps-solve --hkl data.hkl --ins data.ins \
  --method partial_phaseed --phase-seed-csv known_phases.csv --out ./out_partial

# 2) SHELXS fragment / trial.res → Fcalc seed (no manual CSV)
gps-solve --hkl data.hkl --ins data.ins \
  --method partial_phaseed --phase-seed-res shelxs_job.res --out ./out_frag

# 3) Build seed CSV offline, then solve
gps-make-seed --hkl data.hkl --ins data.ins --from-res model.res -o seed.csv
gps-solve --hkl data.hkl --ins data.ins \
  --method partial_phaseed --phase-seed-csv seed.csv --out ./out_partial

# 4) Isomorphous pair (HA)
gps-solve --hkl der.hkl --ins data.ins --method ha_phaseed \
  --native-hkl nat.hkl --derivative-hkl der.hkl --ha-element Br --out ./out_ha
```

`report.md` includes a **Partial seed quality** section (strong-|E| coverage vs the 30% bar, free FOM of the raw seed, next-step hints). Size is truth-free; correctness still requires chemistry / refinement.

**Packaged demo** (synthetic hard-ish + 30% oracle seed):

```bash
gps-solve --hkl examples/partial_seed_demo/demo_hard.hkl \
  --ins examples/partial_seed_demo/demo_hard.ins \
  --method partial_phaseed \
  --phase-seed-csv examples/partial_seed_demo/known_phases_30pct.csv \
  --out examples/partial_seed_demo/out_30

# Fragment path: Fcalc from a partial truth model written as .res (see demo README)
python scripts/run_partial_seed_demo.py
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

If the map is uninterpretable: check cell/SG, try `--method ensemble` or `phai_phaseed`, improve resolution/completeness, add a predicted-model or fragment seed (`--predicted-model`), or use classical SHELXD / experimental phasing. Report.md includes free-FOM bootstrap and optional multistart phase uncertainty (v0.5).

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
