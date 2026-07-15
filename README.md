# grok_phase_solver

**AI-driven / physics-based phasing assistant for X-ray crystallography**

Recover phases $\varphi(hkl)$ from experimental amplitudes $|F(hkl)|$ using first-principles physics, classical algorithms (Patterson, direct methods, charge flipping, RAAR, Difference Map), experimental-phasing hybrids (MIR/MAD/MR), multistart ensembles, conditional neural–classical hybrids, and optional [PhAI](https://doi.org/10.1126/science.adn2777)-style neural phasing.

$$
\rho(\mathbf{r})
=
\frac{1}{V}\sum_{\mathbf{h}}
|F(\mathbf{h})|\,e^{i\varphi(\mathbf{h})}\,e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}
$$

---

## For experimentalists — solve from your data

**Full guide:** [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)

### 1. Install

```bash
git clone https://github.com/pileofflapjacks1/grok_phase_solver.git
cd grok_phase_solver
python -m pip install -e .
```

### 2. Run on your experiment

```bash
# SHELX-style pair (recommended)
gps-solve --hkl mycrystal.hkl --ins mycrystal.ins --out ./solve_out

# Or pass cell + space group explicitly
gps-solve --hkl mycrystal.hkl \
  --cell 9.75,8.89,7.57,90,112.7,90 \
  --sg "P 1 21/c 1" \
  --out ./solve_out
```

### 3. Open the results

| File | What it is |
|------|------------|
| `solve_out/report.md` | Summary + free FOM + next steps |
| `solve_out/density_slice.png` | Quick map view |
| `solve_out/peaks.csv` | Strong density maxima (trial atoms) |
| `solve_out/trial.res` | SHELXL-style trial model for Olex2 |
| `solve_out/phases.csv` | Phased structure factors |

Then refine in **SHELXL / Olex2** as usual. This tool helps **phase** the data; it does not replace refinement.

`auto` picks among charge flipping, multistart ensemble, PhAI+AI-PhaSeed, and free-FOM–gated hybrids based on space group, resolution, and available weights.

### Demo (no lab data)

```bash
gps-solve --hkl examples/demo_solve/demo.hkl --ins examples/demo_solve/demo.ins \
  --method charge_flipping --n-iter 100 --out examples/demo_solve/out
```

**Scope:** strongest for **small molecules** at good resolution. Not a general protein ab initio solver. Always validate with refinement R-factors.

---

## Cowtan overview (integrated)

Pedagogical backbone: **Kevin Cowtan**, *Phase Problem in X-ray Crystallography, and Its Solution* (Encyclopedia of Life Sciences, 2001). Full notes: [`docs/cowtan_phase_problem_notes.md`](docs/cowtan_phase_problem_notes.md).

| Classical approach | When it works | In this repo |
|--------------------|---------------|--------------|
| **Patterson** | Few atoms / heavy atoms; map of interatomic vectors from $|F|^2$ | `physics/patterson.py`, `solvers/patterson.py` |
| **Direct methods** | Atomic resolution; triplet invariants $\varphi_h+\varphi_k+\varphi_{-h-k}\approx 0$ | `solvers/direct_methods.py` (κ-weighted) |
| **MIR** | Isomorphous heavy-atom derivatives | `data/experimental_phasing.simulate_mir` |
| **MAD** | Anomalous scatterers, multi-λ (e.g. SeMet) | `simulate_mad` |
| **MR** | Homologous model | `simulate_mr` |
| **Phase improvement** | Solvent flatten, NCS, iterate | density mod, HIO positivity |
| **Charge flipping / HIO** | Ab initio density constraints | `solvers/charge_flipping.py`, `hio.py` |
| **RAAR / DiffMap / ER** | Modern projection algorithms | `solvers/iterative_retrieval.py` |

Derivations (Patterson identity, Cochran triplets): [`notebooks/02_patterson_and_triplets.md`](notebooks/02_patterson_and_triplets.md).  
Hybrid AI test matrix: [`docs/hybrid_ai_tests.md`](docs/hybrid_ai_tests.md).  
Projection math: [`docs/math/iterative_projections.md`](docs/math/iterative_projections.md).

---

## Status

See **[`TODO.md`](TODO.md)** for the full phase checklist.

| Phase | Focus | State |
|-------|--------|--------|
| **1** | I/O, CF/HIO, COD samples, metrics, math docs | ✅ |
| **1b** | Cowtan classical (Patterson, DM, MIR Blow–Crick, Δ-Patterson) | ✅ |
| **2** | Fragment synth, Wilson gap, PhaseMLP, hybrid benchmark | ✅ core / 🚧 scale |
| **3** | Hybrid seed+polish, RAAR/DiffMap, ensemble, recycle net, uniqueness | ✅ core / 🚧 research |
| **4** | Plots, arXiv skeleton, CI, solvability + PhAI scoreboards | ✅ core / 🚧 external tools |

**Honest limit:** This framework implements *correct classical mathematics* and hybrid testbeds. It does **not** claim a general solution of the phase problem for proteins. Hard synthetic cells ($n \ge 12$, $d_{\min} \ge 1.5\,\text{Å}$) remain largely unsolved under strict success criteria.

| Component | State |
|-----------|--------|
| CIF / HKL / MTZ I/O (gemmi + pure CIF fallback) | ✅ |
| Structure factors + density FFT | ✅ |
| Patterson map + peak pick | ✅ |
| Direct methods (E-values, κ-triplets, tangent multi-start) | ✅ |
| Charge flipping / HIO / ER | ✅ |
| RAAR, Difference Map (+ charge-flip $P_S$, β retune) | ✅ |
| Multistart ensemble (CF+RAAR, free-FOM pick) | ✅ |
| Free FOM + conditional hybrid polish | ✅ |
| MIR / MAD / MR simulators + Blow–Crick | ✅ |
| Strict success metrics (mapCC_OI, peak recovery, R1) | ✅ |
| Fragment synthetic shards + PhaseMLP | ✅ |
| Physics-recycle net (hard-region training) | ✅ |
| Fair PhAI packing + conditional PhAI hybrids | ✅ |
| `gps-solve` end-user pipeline | ✅ |
| PhAI weights | 🔌 local / ERDA (not redistributed) |
| Large equivariant production nets | 🚧 |

---

## Solvers & research stack

### Classical ab initio

| Solver | Module | Notes |
|--------|--------|--------|
| Charge flipping | `solvers/charge_flipping.py` | Oszlányi–Sütő baseline |
| HIO | `solvers/hio.py` | Fienup hybrid input–output |
| Direct methods | `solvers/direct_methods.py` | E-values, Cochran κ, tangent |
| Patterson | `solvers/patterson.py` | Interatomic-vector baseline |
| Density modification | `solvers/density_modification.py` | Solvent flatten / positivity |

### Iterative projections (frontier)

| Algorithm | Module | Notes |
|-----------|--------|--------|
| RAAR | `solvers/iterative_retrieval.py` | Luke 2005; β + positivity or charge-flip |
| Difference Map | same | Elser 2003; grid-search retune for β, $P_S$, δσ |
| Error reduction (ER) | same | Alternating $P_S P_M$ |
| Free FOM v2 | `solvers/free_fom.py` | Positivity residual $R_+$, atomicity, calibrated gate ([math note](docs/math/free_fom.md)) |
| Multistart ensemble | `solvers/ensemble.py` | CF+RAAR starts → free-FOM pick |
| Conditional hybrid | `solvers/conditional_hybrid.py` | Polish seed only if free FOM improves |
| **AI-PhaSeed** | `solvers/ai_phaseed.py` | AI seed → strong-\|E\| extension → free-FOM polish |
| **Hard-P1 prior** | `models/hard_p1_prior.py` | Domain-matched PhaseMLP (OI loss) → AI-PhaSeed |
| **Strong graph prior** | `models/graph_phase_net.py`, `strong_prior.py` | Scaled triplet-GNN (curriculum + triplet aux) → AI-PhaSeed |
| **Dual-space / SHELXD** | `solvers/dual_space.py`, `shelxd_runner.py` | Educational dual-space + optional external SHELXD H2H |
| **Partial-φ / fragment seed** | `solvers/partial_seed.py` | Known phases or partial model → AI-PhaSeed extension |
| Phase recycle | `solvers/phase_recycle.py` | Fourier modulus projection loop |
| Physics-recycle net | `solvers/recycle_net.py` | PhaseMLP inside recycle (hard cells) |

**DiffMap retune recommendation** (synthetic free-FOM grid): **β ≈ 0.5**, **real_proj = charge_flip**, **delta_sigma = 1.0**. See [`data/processed/diffmap_retune.md`](data/processed/diffmap_retune.md).

### Hybrid / neural

| Path | Role |
|------|------|
| PhaseMLP | Supervised phase seed (synthetic) |
| PhAI fair packing | Official-style merge + `/max|F|` prep |
| PhAI + CF/RAAR (conditional) | Free-FOM gate; avoids destroying a good neural prior |
| Hybrid seed+polish | External phases → CF / HIO / DM |

---

## Install (developers)

```bash
cd grok_phase_solver
python -m pip install -e ".[dev]"
# Optional ML stack (PhAI):
# python -m pip install -e ".[ml]"   # + weights: third_party/phai/README.md
```

Requires Python ≥ 3.10, NumPy, SciPy, Matplotlib, [gemmi](https://gemmi.readthedocs.io/).

## Developer quick start

```python
from grok_phase_solver.pipeline import solve_structure, export_solution
from grok_phase_solver.pipeline.solve import SolveConfig

result = solve_structure(
    "mycrystal.hkl",
    ins_path="mycrystal.ins",
    config=SolveConfig(method="charge_flipping", n_iter=150),
)
export_solution(result, "solve_out")
```

### Ensemble + conditional hybrid (library API)

```python
from grok_phase_solver.solvers import (
    ensemble_cf_raar,
    conditional_polish,
    raar_solve,
    difference_map_solve,
    retune_difference_map,
    recycle_net_solve,
    load_recycle_net,
    phai_phaseed_solve,
    ai_phaseed_solve,
)

# Multistart CF + RAAR; pick by free FOM (no ground truth needed)
phases, rho, info = ensemble_cf_raar(hkl, amplitudes, cell, n_starts=5, n_iter=120)

# Seed → polish only if free FOM improves
phases, rho, info = conditional_polish(
    hkl, amplitudes, cell, phases_seed, polish="raar", n_iter=80
)

# AI-PhaSeed: PhAI fair → strong-|E| extension → free-FOM–gated CF
phases, rho, info = phai_phaseed_solve(
    hkl, amplitudes, cell, n_extend=15, polish="charge_flipping", n_starts=2
)

# Domain-matched hard-P1 prior + AI-PhaSeed (synthetic hard P1)
from grok_phase_solver.models.hard_p1_prior import hard_p1_phaseed_solve
phases, rho, info = hard_p1_phaseed_solve(hkl, amplitudes, cell)


# DiffMap with retuned charge-flip projector
phases, rho, hist = difference_map_solve(
    hkl, amplitudes, cell, beta=0.5, real_proj="charge_flip", delta_sigma=1.0
)
```

### CLIs & scripts

```bash
gps-solve --help
gps-download-cod
gps-baseline --synthetic --n-atoms 8 --method charge_flipping --dmin 1.2

# Research benchmarks
python scripts/run_scoreboard.py
python scripts/run_solvability_diagram.py
python scripts/run_fair_phai_benchmark.py
python scripts/run_frontier_benchmark.py
python scripts/run_ensemble_benchmark.py
python scripts/run_diffmap_retune.py
python scripts/train_recycle_net.py
python scripts/run_cod_hybrid_benchmark.py
python scripts/train_phase_mlp.py
```

## Repository layout

```text
src/grok_phase_solver/
  io/           # CIF, HKL, MTZ, ReflectionTable
  physics/      # form factors, Fcalc, density, Patterson, reciprocal
  solvers/      # CF, HIO, DM, RAAR, DiffMap, ensemble, hybrid, recycle_net
  metrics/      # phase error, map CC, FSC, R, strict SuccessThresholds
  data/         # COD, synthetic v1/v2, MIR/MAD/MR, Wilson
  models/       # PhAI fair/runner, PhaseMLP, physics losses
  pipeline/     # gps-solve: solve + peaks + export
docs/           # USER_GUIDE, math, Cowtan notes, roadmap
notebooks/      # 01 baseline, 02 Patterson & triplets, 03 uniqueness
scripts/        # scoreboard + frontier + ensemble + COD hybrid + training
data/raw/cod/   # COD samples (+ experimental HKL where available)
data/processed/ # benchmark JSON/MD (force-tracked reports)
tests/
third_party/phai/  # weights docs (model.pth gitignored)
```

## PhAI integration

https://erda.ku.dk/archives/e3be15d017d8c4fe81402da833e26894/published-archive.html  

See [`third_party/phai/README.md`](third_party/phai/README.md) for weight placement (`third_party/phai/weights/PhAI_model.pth`).

Fair packing protocol (reindex monoclinic + merge + `/max|F|`) lives in `models/phai_fair.py` — this is required for apples-to-apples PhAI comparisons.

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

## Sample data (COD)

| COD ID | System | Role |
|--------|--------|------|
| [2016452](https://www.crystallography.net/cod/2016452.html) | PhAI demo (P2₁/c) | Fair PhAI + hybrid benchmarks |
| [2100301](https://www.crystallography.net/cod/2100301.html) | C₇H₅NO₄, P2₁/c | Small organic baseline |
| [2017775](https://www.crystallography.net/cod/2017775.html) | Roxithromycin | Larger + experimental HKL |

## Scoreboard & research results

Strict success = **mapCC_OI ≥ 0.7** + **peak recovery ≥ 0.5** + **R1 ≤ 0.45** (`metrics/success.py`).

| Report | Command | Output |
|--------|---------|--------|
| Multi-method scoreboard | `python scripts/run_scoreboard.py` | [`scoreboard.md`](data/processed/scoreboard.md) |
| Solvability phase diagram | `python scripts/run_solvability_diagram.py` | [`solvability_diagram.md`](data/processed/solvability_diagram.md) |
| Fair PhAI benchmark | `python scripts/run_fair_phai_benchmark.py` | [`fair_phai_benchmark.md`](data/processed/fair_phai_benchmark.md) |
| Frontier (RAAR / DiffMap / κ-DM) | `python scripts/run_frontier_benchmark.py` | [`frontier_benchmark.md`](data/processed/frontier_benchmark.md) |
| Multistart ensemble CF+RAAR | `python scripts/run_ensemble_benchmark.py` | [`ensemble_benchmark.md`](data/processed/ensemble_benchmark.md) |
| DiffMap retune (β, $P_S$, δσ) | `python scripts/run_diffmap_retune.py` | [`diffmap_retune.md`](data/processed/diffmap_retune.md) |
| Physics-recycle net (hard cells) | `python scripts/train_recycle_net.py` | [`recycle_net.md`](data/processed/recycle_net.md) |
| COD 2016452 PhAI hybrids | `python scripts/run_cod_hybrid_benchmark.py` | [`cod_hybrid_benchmark.md`](data/processed/cod_hybrid_benchmark.md) |
| Free-FOM calibration | `python scripts/calibrate_free_fom.py` | [`free_fom_calibration.md`](data/processed/free_fom_calibration.md) |
| Failure taxonomy (A/B/C) | `python scripts/run_failure_taxonomy.py` | [`failure_taxonomy.md`](data/processed/failure_taxonomy.md) |
| PhAI-seeded taxonomy | `python scripts/run_phai_taxonomy.py` | [`phai_taxonomy.md`](data/processed/phai_taxonomy.md) |
| AI-PhaSeed benchmark | `python scripts/run_ai_phaseed_benchmark.py` | [`ai_phaseed_benchmark.md`](data/processed/ai_phaseed_benchmark.md) |
| Hard-P1 domain prior | `python scripts/train_hard_p1_prior.py` | [`hard_p1_prior.md`](data/processed/hard_p1_prior.md) |
| Experimental HKL scoreboard | `python scripts/run_experimental_scoreboard.py` | [`experimental_scoreboard.md`](data/processed/experimental_scoreboard.md) |
| Strong GraphPhaseNet prior | `python scripts/train_strong_prior.py --scale` | [`strong_prior.md`](data/processed/strong_prior.md) |
| SHELXD head-to-head | `python scripts/run_shelxd_h2h.py` | [`shelxd_h2h.md`](data/processed/shelxd_h2h.md), [`docs/math/shelxd_h2h.md`](docs/math/shelxd_h2h.md) |
| Partial-φ hard-cliff curves | `python scripts/run_partial_seed_benchmark.py` | [`partial_seed_benchmark.md`](data/processed/partial_seed_benchmark.md), [`docs/math/partial_seed.md`](docs/math/partial_seed.md) |
| Wilson domain-gap (close gap) | `python scripts/run_wilson_domain_gap.py` | [`wilson_domain_gap.md`](data/processed/wilson_domain_gap.md), [`docs/math/wilson_domain_gap.md`](docs/math/wilson_domain_gap.md) |
| Math write-ups | — | [`docs/math/free_fom.md`](docs/math/free_fom.md), [`docs/math/failure_taxonomy.md`](docs/math/failure_taxonomy.md), [`docs/math/solvability_and_phai.md`](docs/math/solvability_and_phai.md), [`docs/math/iterative_projections.md`](docs/math/iterative_projections.md) |

### Headlines (reproducible reports)

- **Solvability cliff:** classical methods work on small $N$ / high resolution; success collapses for large $N$ / low $d_{\min}$.
- **Fair PhAI:** with official-style prep, PhAI mapCC exceeds CF on COD 2016452 Fcalc; **`phai+CF` solves** 2016452 @ 0.9 Å under strict criteria (mapCC ≈ 0.87).
- **Conditional polish:** free-FOM gate **accepts** helpful CF polish at high res; **rejects** RAAR when it would destroy a good PhAI prior (important at 1.2–2.0 Å).
- **Free FOM v2:** fixed vacuous post-modulus $R$; uses positivity residual $R_+$, kurtosis/peakiness, conservative gate (composite↑ **and** $R_+$ not worse). See [`docs/math/free_fom.md`](docs/math/free_fom.md).
- **Failure taxonomy:** with free-FOM v2.1 (anti-false-atomicity), calibration **FOM inversion → 0%**; hard-region failures shift toward **B+C** (basin + degeneracy). PhAI seeding on synthetic P1 hard cells helps modestly (see `phai_taxonomy.md`). Docs: [`failure_taxonomy.md`](docs/math/failure_taxonomy.md).
- **Ensemble:** multistart CF+RAAR improves some easy cases; hard region remains **0%** strict success (honest ceiling for pure classical multistart).
- **DiffMap retune:** charge-flip $P_S$ + β≈0.5 beats default positivity DiffMap on free FOM; still trails CF on truth mapCC in many cells.
- **Recycle net:** PhaseMLP trained on hard synthetic cells; physics recycle enforces \|F\| consistency — supervised prior, not a claimed general solver.

## Documentation

- [User guide (`gps-solve`)](docs/USER_GUIDE.md)
- [Math overview](docs/math/phase_problem_overview.md)
- [Uniqueness & non-claims](docs/math/uniqueness_and_bounds.md)
- [Baseline failure modes](docs/math/baseline_failure_modes.md)
- [Iterative projections (RAAR / DiffMap)](docs/math/iterative_projections.md)
- [Solvability & fair PhAI](docs/math/solvability_and_phai.md)
- [Cowtan notes](docs/cowtan_phase_problem_notes.md)
- [Hybrid AI tests](docs/hybrid_ai_tests.md)
- [Roadmap](docs/roadmap.md)
- [Notebook 01](notebooks/01_math_and_baseline.md) · [02](notebooks/02_patterson_and_triplets.md) · [03](notebooks/03_uniqueness_parseval_friedel.md)

## Tests

```bash
pytest -q
```

Covers I/O/physics, classical methods, iterative frontier (RAAR/DiffMap/free FOM), ensemble, recycle net, success criteria, and user pipeline.

## License

MIT — see [LICENSE](LICENSE).  
Third-party data/models retain original licenses. Cowtan ELS article cited, not redistributed. PhAI weights are **not** redistributed; obtain per upstream terms.

## Contributing

**Plan → Code → Test → Analyze math → Refine → Commit.**  
Every ML component keeps a physics fallback. Prefer mathematical correctness over marketing claims — do not assert a general solution of the crystallographic phase problem.
