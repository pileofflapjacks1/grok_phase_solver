# grok_phase_solver

**AI-driven / physics-based phasing assistant for X-ray crystallography**

Recover phases \(\varphi(hkl)\) from experimental amplitudes \(|F(hkl)|\) using first-principles physics, classical algorithms (Patterson, direct methods, charge flipping), experimental-phasing hybrids (MIR/MAD/MR), and optional [PhAI](https://doi.org/10.1126/science.adn2777)-style neural phasing.

\[
\rho(\mathbf{r})
=
\frac{1}{V}\sum_{\mathbf{h}}
|F(\mathbf{h})|\,e^{i\varphi(\mathbf{h})}\,e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}
\]

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
| `solve_out/report.md` | Summary + next steps |
| `solve_out/density_slice.png` | Quick map view |
| `solve_out/peaks.csv` | Strong density maxima (trial atoms) |
| `solve_out/phases.csv` | Phased structure factors |

Then refine in **SHELXL / Olex2** as usual. This tool helps **phase** the data; it does not replace refinement.

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
| **Patterson** | Few atoms / heavy atoms; map of interatomic vectors from \(\|F\|^2\) | `physics/patterson.py`, `solvers/patterson.py` |
| **Direct methods** | Atomic resolution; triplet invariants \(\varphi_h+\varphi_k+\varphi_{-h-k}\approx 0\) | `solvers/direct_methods.py` |
| **MIR** | Isomorphous heavy-atom derivatives | `data/experimental_phasing.simulate_mir` |
| **MAD** | Anomalous scatterers, multi-λ (e.g. SeMet) | `simulate_mad` |
| **MR** | Homologous model | `simulate_mr` |
| **Phase improvement** | Solvent flatten, NCS, iterate | HIO positivity; Phase 3 envelopes |
| **Charge flipping / HIO** | Ab initio density constraints | `solvers/charge_flipping.py`, `hio.py` |

Derivations (Patterson identity, Cochran triplets): [`notebooks/02_patterson_and_triplets.md`](notebooks/02_patterson_and_triplets.md).  
Hybrid AI test matrix: [`docs/hybrid_ai_tests.md`](docs/hybrid_ai_tests.md).

## Status

See **[`TODO.md`](TODO.md)** for the full phase checklist.

| Phase | Focus | State |
|-------|--------|--------|
| **1** | I/O, CF/HIO, COD samples, metrics, math docs | ✅ |
| **1b** | Cowtan classical (Patterson, DM, MIR Blow–Crick, Δ-Patterson) | ✅ |
| **2** | Fragment synth, Wilson gap, PhaseMLP, hybrid benchmark | ✅ core / 🚧 scale |
| **3** | Hybrid seed+polish, solvent flatten, uniqueness docs | ✅ core / 🚧 research |
| **4** | Plots, arXiv skeleton, CI | ✅ core / 🚧 external tools |

**Honest limit:** This framework implements *correct classical mathematics* and hybrid testbeds. It does **not** claim a general solution of the phase problem for proteins.

| Component | State |
|-----------|--------|
| CIF / HKL I/O (gemmi) | ✅ |
| Structure factors + density FFT | ✅ |
| Patterson map + peak pick | ✅ |
| Direct methods (E-values, triplets, tangent) | ✅ |
| Charge flipping / HIO | ✅ |
| MIR / MAD / MR simulators | ✅ |
| Metrics (origin-inv. map CC, MPE, FSC, R) | ✅ |
| Fragment synthetic shards (Phase 2) | ✅ |
| Physics-informed losses (NumPy) | ✅ |
| PhAI weights | 🔌 ERDA archive interface |
| Full torch training loop | 🚧 |

### Baseline highlights (Phase 1)

- Synthetic P1 (~6 atoms): charge-flipping **origin-invariant map CC ≈ 0.87**
- COD 2100301 @ 0.9 Å: CF **map CC ≈ 0.83**; weaker at 1.2–2.0 Å (documents classical failure → need priors / PhAI)

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

### Other CLIs

```bash
gps-solve --help
gps-download-cod
gps-baseline --synthetic --n-atoms 8 --method charge_flipping --dmin 1.2
python scripts/run_scoreboard.py
```

## Repository layout

```text
src/grok_phase_solver/
  io/           # CIF, HKL, ReflectionTable
  physics/      # form factors, Fcalc, density, Patterson, reciprocal
  solvers/      # CF, HIO, direct methods, Patterson, baseline API
  metrics/      # phase error, map CC, FSC, R-factor
  data/         # COD, synthetic v1/v2, MIR/MAD/MR
  models/       # PhAI interface, physics losses
docs/           # math, Cowtan notes, hybrid AI tests, roadmap
notebooks/      # 01 baseline, 02 Patterson & triplets
data/raw/cod/   # COD 2100301, 2017775 (+ experimental HKL)
tests/
```

## PhAI integration

https://erda.ku.dk/archives/e3be15d017d8c4fe81402da833e26894/published-archive.html  

See `third_party/phai/README.md`.

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
| [2100301](https://www.crystallography.net/cod/2100301.html) | C₇H₅NO₄, P2₁/c | Small organic |
| [2017775](https://www.crystallography.net/cod/2017775.html) | Roxithromycin | Larger + experimental HKL |

## Scoreboard & solvability (research)

| Report | Command | Output |
|--------|---------|--------|
| Multi-method scoreboard | `python scripts/run_scoreboard.py` | [`data/processed/scoreboard.md`](data/processed/scoreboard.md) |
| **Solvability phase diagram** | `python scripts/run_solvability_diagram.py` | [`data/processed/solvability_diagram.md`](data/processed/solvability_diagram.md) |
| **Fair PhAI benchmark** | `python scripts/run_fair_phai_benchmark.py` | [`data/processed/fair_phai_benchmark.md`](data/processed/fair_phai_benchmark.md) |
| Math summary | — | [`docs/math/solvability_and_phai.md`](docs/math/solvability_and_phai.md) |

**Headline (strict success = mapCC≥0.7 + peak recovery≥0.5 + R1≤0.45):**

- Synthetic frontier: CF success ~28% overall; collapses for large \(N\) / low resolution.  
- Fair PhAI (official merge + /max \|F\|): **PhAI mapCC > CF** on COD 2016452 Fcalc; **`phai_fair+CF` solves** 2016452 @ 0.9 Å (mapCC 0.87).  
- PhAI weights: local `third_party/phai/weights/PhAI_model.pth` (gitignored; see `third_party/phai/README.md`).

## Documentation

- [Math overview](docs/math/phase_problem_overview.md)
- [Baseline failure modes](docs/math/baseline_failure_modes.md)
- [Cowtan notes](docs/cowtan_phase_problem_notes.md)
- [Hybrid AI tests](docs/hybrid_ai_tests.md)
- [Roadmap](docs/roadmap.md)
- [Notebook 01](notebooks/01_math_and_baseline.md) · [Notebook 02](notebooks/02_patterson_and_triplets.md)

## Tests

```bash
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).  
Third-party data/models retain original licenses. Cowtan ELS article cited, not redistributed.

## Contributing

**Plan → Code → Test → Analyze math → Refine → Commit.**  
Every ML component keeps a physics fallback.
