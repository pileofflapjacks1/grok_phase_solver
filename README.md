# grok_phase_solver

**Open physics / AI phasing assistant for X-ray crystallography**  
Version **0.3.0** · Python ≥ 3.10 · MIT

Recover phases $\varphi(hkl)$ from measured amplitudes $|F(hkl)|$, write density maps and a SHELXL-ready trial model, then refine elsewhere (Olex2 / SHELXL).

$$
\rho(\mathbf{r})
=
\frac{1}{V}\sum_{\mathbf{h}}
|F(\mathbf{h})|\,e^{i\varphi(\mathbf{h})}\,e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}
$$

| | |
|--|--|
| **Audience** | Experimental crystallographers + method developers |
| **Best at** | Small molecules, good resolution; transparent classical + hybrid pipelines |
| **Not** | A general protein ab initio solver or a drop-in SHELXL replacement |
| **Docs** | [User guide](docs/USER_GUIDE.md) · [**For reviewers**](docs/FOR_REVIEWERS.md) · [**Paper pack**](docs/paper/README.md) · [arXiv draft](docs/arxiv_draft.md) · [Math notes](docs/math/) · [Changelog](CHANGELOG.md) · [TODO](TODO.md) |
| **Repo** | https://github.com/pileofflapjacks1/grok_phase_solver |

---

## 1. Five-minute start (scientist path)

### Install (recommended: PyPI)

```bash
python -m pip install grok-phase-solver
gps-solve --help

# Optional browser UI
python -m pip install "grok-phase-solver[gui]"
gps-gui
# → http://localhost:8501  (upload HKL/INS, download trial.res)
```

On macOS, use `python3` or Anaconda if `python` is not found:

```bash
source ~/anaconda3/bin/activate   # if you use Anaconda
python -m pip install grok-phase-solver
```

### Install from source (developers)

```bash
git clone https://github.com/pileofflapjacks1/grok_phase_solver.git
cd grok_phase_solver
python -m pip install -e ".[gui]"
gps-solve --help
```

Optional PhAI weights (not on PyPI): see [`third_party/phai/README.md`](third_party/phai/README.md) and `pip install -e ".[ml]"`.  
Release notes: [`docs/RELEASE.md`](docs/RELEASE.md) · [v0.3.0 notes](docs/RELEASE_NOTES_v0.3.0.md) · [Paper PDF](docs/paper/arxiv_draft.pdf)

### Phase your data

```bash
# Recommended: SHELX-style pair
gps-solve --hkl mycrystal.hkl --ins mycrystal.ins --out ./solve_out

# Without .ins
gps-solve --hkl mycrystal.hkl \
  --cell 9.75,8.89,7.57,90,112.7,90 \
  --sg "P 1 21/c 1" \
  --out ./solve_out
```

### What you get

| File | Purpose |
|------|---------|
| `report.md` | Method used, free FOM, decision hints, SHELXL steps |
| `density_slice.png` | Quick map check |
| `peaks.csv` | Strong density maxima (trial atoms) |
| **`trial.res`** | Load in **Olex2 / ShelXle → assign elements → SHELXL** |
| `phases.csv` | $h,k,l,\|F\|$, phase (°) |

**This tool phases.** It does **not** replace least-squares refinement. Always check chemical sense and refinement R-factors.

### Demos (no lab data)

```bash
# Easy: multistart ensemble (best open ab initio on easy cells in our benchmarks)
gps-solve --hkl examples/demo_solve/demo.hkl --ins examples/demo_solve/demo.ins \
  --method ensemble --n-iter 100 --out examples/demo_solve/out

# Hard-ish + partial phases (path that works when pure ab initio fails)
python scripts/run_partial_seed_demo.py
```

---

## 2. Which method should I use?

```text
Have partial info (φ / fragment / HA)?
   YES →  partial_phaseed + seed source:
            --phase-seed-csv | --phase-seed-res | --seed-peaks-csv
            | --native-hkl + --derivative-hkl | gps-make-seed …
   NO  →  resolution good (d ≲ 1.15 Å)?
            YES → --method auto   (→ ensemble)
            NO  → --method auto   (→ prior/CF; may fail — get partial φ or try shelxs)
Finish → trial.res → SHELXL / Olex2
```

| Situation | Flag |
|-----------|------|
| Default / unsure | `--method auto` |
| Easy / high resolution | `auto` or `ensemble` |
| Hard, pure ab initio | `auto` (honest: often unsolved) |
| **Hard + known φ / fragment / HA** | **`partial_phaseed` + seed source** (CSV / `.res` / peaks / HA) |
| Build seed only | `gps-make-seed --from-res model.res -o seed.csv` |
| External classical solve | `shelxs` or `shelxs+shelxe` (local academic binaries in `ShelX/`) |
| P2₁/c + PhAI weights | `phai_phaseed` |

Full table and flags: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

---

## 3. What this project is (and is not)

### It is
- An **open, modular** implementation of classical phasing (charge flipping, HIO, RAAR, DiffMap, direct methods, Patterson)
- A **scientist CLI** (`gps-solve`) with free figures of merit and strict success metrics
- A **hybrid research stack**: AI-PhaSeed, GraphPhaseNet prior, partial-φ API, Wilson domain matching
- **Head-to-head** harnesses vs academic **SHELXS** (and optional SHELXD/E if you install them)
- **Documented math** and reproducible scoreboards under `data/processed/`

### It is not
- A claim that the crystallographic phase problem is “solved”
- A general **protein** ab initio solver
- A replacement for **SHELXL** refinement or industrial small-molecule pipelines (SHELXT/SHELXS + Olex2)
- A redistributor of **SHELX** or **PhAI** binaries/weights (obtain under their licenses)

**Honest hard-region result:** pure ab initio (CF, ensemble, GraphPhaseNet, even SHELXS on our hard panel) remains **~0% strict success** for harder synthetic cells ($n\gtrsim 12$, $d_{\min}\gtrsim 1.5\,\text{Å}$). With **≥ ~30% correct strong-|E| phases** (≲20° error), AI-PhaSeed **can** strict-solve those cells — so the open path for hard data is **partial information**, not more polish alone.

---

## 4. Key scientific findings (start here if reviewing the work)

**One-pager for referees:** [`docs/FOR_REVIEWERS.md`](docs/FOR_REVIEWERS.md) (claims C1–C8, non-claims, reproduce steps).

Strict success = mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5 **and** R1 ≤ 0.45 ([`metrics/success.py`](src/grok_phase_solver/metrics/success.py)).

| Finding | Where |
|---------|--------|
| **Solvability cliff** — classical success collapses with more atoms / worse resolution | [`solvability_diagram.md`](data/processed/solvability_diagram.md) |
| **Ensemble** best open ab initio on **easy** cells (vs CF / SHELXS in-repo H2H) | [`shelxs_h2h.md`](data/processed/shelxs_h2h.md) |
| **Hard ab initio** ~0% strict for CF, priors, dual-space, SHELXS (our panels) | same + [`strong_prior.md`](data/processed/strong_prior.md) |
| **Partial-φ bar** — ≥30% strong φ within 20° → hard strict solves | [`partial_seed_benchmark.md`](data/processed/partial_seed_benchmark.md), [math](docs/math/partial_seed.md) |
| Graph prior strong-seed quality ~**21%** ≤20° (below 30% bar) even after Lane A **v4 XL** (1200 structs, residual GNN) | [`strong_prior.md`](data/processed/strong_prior.md) |
| Free FOM v2.1 — positivity residual $R_+$; reduces false “solved” gates | [math](docs/math/free_fom.md) |
| Failure taxonomy A/B/C — hard failures are basin + degeneracy, not FOM inversion | [math](docs/math/failure_taxonomy.md) |
| Wilson gap synth→exp can be largely closed by amplitude matching | [math](docs/math/wilson_domain_gap.md) |
| Fair PhAI + conditional CF can solve COD 2016452 Fcalc @ 0.9 Å | [`cod_hybrid_benchmark.md`](data/processed/cod_hybrid_benchmark.md) |

Pedagogy: [Cowtan ELS notes](docs/cowtan_phase_problem_notes.md) · [Phase problem overview](docs/math/phase_problem_overview.md) · [Uniqueness & non-claims](docs/math/uniqueness_and_bounds.md)

---

## 5. Methods at a glance

### What `auto` does
1. **Easy / high-res** ($d_{\min} \le 1.15\,\text{Å}$) → **ensemble** (CF+RAAR multistart, free-FOM pick)  
2. **P2₁/c + PhAI weights** → AI-PhaSeed  
3. **Harder res** → GraphPhaseNet / hard-P1 prior if available, else charge flipping  
4. Partial phases are **never auto-detected** — pass `--phase-seed-csv` yourself  

### Core algorithms (in-repo)

| Family | Examples | Modules |
|--------|----------|---------|
| Classical ab initio | Charge flipping, HIO, DM, Patterson | `solvers/charge_flipping.py`, `hio.py`, `direct_methods.py`, `patterson.py` |
| Projections | RAAR, DiffMap, ER | `solvers/iterative_retrieval.py` |
| Ranking / polish | Free FOM, multistart ensemble, conditional hybrid | `free_fom.py`, `ensemble.py`, `conditional_hybrid.py` |
| Hybrids | AI-PhaSeed, partial-φ, graph prior | `ai_phaseed.py`, `partial_seed.py`, `models/strong_prior.py` |
| External SHELX | SHELXS solve, SHELXE density mod | `shelxs_runner.py`, `shelxe_runner.py` (binaries **not** in git) |

### External SHELX (optional)

Place academic binaries in `ShelX/` (gitignored — do not commit):

```bash
xattr -dr com.apple.quarantine ShelX   # macOS if needed
chmod +x ShelX/shelxs ShelX/shelxe ShelX/shelxl

gps-solve --hkl data.hkl --ins data.ins --method shelxs+shelxe --out out_sx
# Refine: cp out_sx/trial.res work.ins && cp data.hkl work.hkl && ShelX/shelxl work
```

Download: [https://shelx.uni-goettingen.de/](https://shelx.uni-goettingen.de/)

---

## 6. Library API (short)

```python
from grok_phase_solver.pipeline.solve import SolveConfig, solve_structure
from grok_phase_solver.pipeline.export import export_solution

result = solve_structure(
    "mycrystal.hkl",
    ins_path="mycrystal.ins",
    config=SolveConfig(method="auto", n_iter=120, n_starts=3),
)
export_solution(result, "solve_out")
print(result.method, result.diagnostics.get("free_fom_composite"))
```

```python
# Multistart ensemble (no ground truth needed for ranking)
from grok_phase_solver.solvers.ensemble import ensemble_cf_raar
phases, rho, info = ensemble_cf_raar(hkl, amplitudes, cell, n_starts=5, n_iter=120)

# Partial phases → extension
from grok_phase_solver.solvers.partial_seed import partial_phaseed_solve, load_phase_seed_csv
seed, mask, meta = load_phase_seed_csv("known.csv", hkl)
phases, rho, info = partial_phaseed_solve(hkl, amplitudes, cell, seed, mask=mask)
```

---

## 7. Reproduce research reports

```bash
python -m pip install -e ".[dev]"
pytest -q

# Core scoreboards (subset)
python scripts/run_shelxs_h2h.py              # vs local SHELXS if installed
python scripts/run_partial_seed_benchmark.py  # hard-cliff oracle curves
python scripts/run_wilson_domain_gap.py
python scripts/train_strong_prior.py --scale-xl --wilson-match
python scripts/run_experimental_scoreboard.py
python scripts/run_failure_taxonomy.py
```

| Report | Output |
|--------|--------|
| SHELXS H2H | [`data/processed/shelxs_h2h.md`](data/processed/shelxs_h2h.md) |
| Partial-φ curves | [`data/processed/partial_seed_benchmark.md`](data/processed/partial_seed_benchmark.md) |
| Strong prior v3 | [`data/processed/strong_prior.md`](data/processed/strong_prior.md) |
| Wilson gap | [`data/processed/wilson_domain_gap.md`](data/processed/wilson_domain_gap.md) |
| Failure taxonomy | [`data/processed/failure_taxonomy.md`](data/processed/failure_taxonomy.md) |
| Free-FOM calibration | [`data/processed/free_fom_calibration.md`](data/processed/free_fom_calibration.md) |
| Solvability diagram | [`data/processed/solvability_diagram.md`](data/processed/solvability_diagram.md) |
| Experimental HKL | [`data/processed/experimental_scoreboard.md`](data/processed/experimental_scoreboard.md) |
| Full script list | [`scripts/`](scripts/) · [`TODO.md`](TODO.md) |

---

## 8. Repository map

```text
src/grok_phase_solver/
  pipeline/   # gps-solve: solve, peaks, export (trial.res)
  solvers/    # CF, ensemble, free FOM, AI-PhaSeed, partial_seed, SHELXS/E runners
  physics/    # Fcalc, density FFT, Patterson, form factors
  metrics/    # mapCC_OI, R1, success thresholds, strong-seed bar
  models/     # GraphPhaseNet, hard_p1, PhAI fair packing
  data/       # synthetic, Wilson match, MIR/MAD/MR simulators
  io/         # HKL, INS, CIF, MTZ, SHELX writers
docs/         # USER_GUIDE + math/ + Cowtan notes
examples/     # demo_solve (easy), partial_seed_demo (hard + partial φ)
scripts/      # benchmarks and training
data/processed/  # committed scoreboard .md/.json
notebooks/    # pedagogical 01–03
tests/        # pytest (100+ tests)
ShelX/        # YOUR local SHELX binaries only (gitignored)
```

---

## 9. Sample data (COD)

| COD ID | Role |
|--------|------|
| [2016452](https://www.crystallography.net/cod/2016452.html) | PhAI / hybrid Fcalc control (P2₁/c) |
| [2100301](https://www.crystallography.net/cod/2100301.html) | Small organic baseline |
| [2017775](https://www.crystallography.net/cod/2017775.html) | Larger cell + experimental HKL in-repo |

```bash
gps-download-cod          # helper for COD samples
```

---

## 10. Documentation index

| For… | Read |
|------|------|
| **Reviewing / refereeing** | [**FOR_REVIEWERS**](docs/FOR_REVIEWERS.md) (one-pager) |
| Running on lab data | [**USER_GUIDE**](docs/USER_GUIDE.md) |
| Browser GUI | `gps-gui` after `pip install -e ".[gui]"` |
| Partial-φ theory + API | [docs/math/partial_seed.md](docs/math/partial_seed.md) |
| Free FOM | [docs/math/free_fom.md](docs/math/free_fom.md) |
| Failure modes A/B/C | [docs/math/failure_taxonomy.md](docs/math/failure_taxonomy.md) |
| Projections (RAAR/DiffMap) | [docs/math/iterative_projections.md](docs/math/iterative_projections.md) |
| Graph prior | [docs/math/strong_prior.md](docs/math/strong_prior.md) |
| SHELXS H2H notes | [docs/math/shelxs_h2h.md](docs/math/shelxs_h2h.md) |
| Wilson matching | [docs/math/wilson_domain_gap.md](docs/math/wilson_domain_gap.md) |
| Uniqueness / non-claims | [docs/math/uniqueness_and_bounds.md](docs/math/uniqueness_and_bounds.md) |
| Cowtan overview | [docs/cowtan_phase_problem_notes.md](docs/cowtan_phase_problem_notes.md) |
| Paper / arXiv draft | [docs/paper/README.md](docs/paper/README.md) · [arxiv_draft.md](docs/arxiv_draft.md) · [figures](docs/figures/paper_figure_captions.md) |
| Notebooks | [01](notebooks/01_math_and_baseline.md) · [02](notebooks/02_patterson_and_triplets.md) · [03](notebooks/03_uniqueness_parseval_friedel.md) |

---

## 11. Tests & packaging

```bash
pytest -q
python -m build && python -m twine check dist/grok_phase_solver-0.3.0*
# PyPI (needs API token): python -m twine upload dist/grok_phase_solver-0.3.0*
```

PyPI upload is optional (`twine upload dist/*` with your API token). GitHub source install is fully supported.

---

## 12. Citation

If you use this framework, please cite the **upstream methods** you rely on (and this repository if useful):

```bibtex
@article{Larsen2024PhAI,
  author  = {Larsen, Anders S. and Rekis, Toms and Madsen, Anders {\O}.},
  title   = {PhAI: A deep-learning approach to solve the crystallographic phase problem},
  journal = {Science},
  year    = {2024},
  doi     = {10.1126/science.adn2777}
}
@incollection{Cowtan2001Phase,
  author    = {Cowtan, Kevin},
  title     = {Phase Problem in X-ray Crystallography, and Its Solution},
  booktitle = {Encyclopedia of Life Sciences},
  year      = {2001}
}
```

SHELX (Sheldrick): use under academic terms from [shelx.uni-goettingen.de](https://shelx.uni-goettingen.de/).  
COD: [crystallography.net/cod](https://www.crystallography.net/cod/).  
gemmi: [gemmi.readthedocs.io](https://gemmi.readthedocs.io/).

---

## 13. License & contributing

**MIT** — see [LICENSE](LICENSE). Third-party data/models keep their own terms. **Do not commit** SHELX binaries or PhAI weights.

Contributions: prefer mathematical correctness and physics fallbacks over marketing claims. Workflow: plan → code → test → document limits → commit.

```text
Plan → Code → Test → Analyze math → Refine → Commit
```

---

## 14. Contact / status

- **Checklist:** [TODO.md](TODO.md)  
- **Version notes:** [CHANGELOG.md](CHANGELOG.md)  
- **Issues / PRs:** GitHub repository above  

**Bottom line for a new reader:** clone → `pip install -e .` → run `gps-solve` on `examples/` → read `report.md` → refine in SHELXL. For hard structures, bring partial phases. For the science, start with §4 and the linked scoreboards.
