# grok_phase_solver

**AI-driven general solver for the X-ray crystallography phase problem**

Recover phases \(\varphi(hkl)\) from experimental amplitudes \(|F(hkl)|\) using first-principles physics, classical algorithms (Patterson, direct methods, charge flipping), experimental-phasing hybrids (MIR/MAD/MR), and (Phase 2+) deep learning—extending [PhAI](https://doi.org/10.1126/science.adn2777) toward a general open-source tool.

\[
\rho(\mathbf{r})
=
\frac{1}{V}\sum_{\mathbf{h}}
|F(\mathbf{h})|\,e^{i\varphi(\mathbf{h})}\,e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}
\]

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

## Install

```bash
cd grok_phase_solver
python -m pip install -e ".[dev]"
# Optional ML stack:
# python -m pip install -e ".[ml]"
```

Requires Python ≥ 3.10, NumPy, SciPy, Matplotlib, [gemmi](https://gemmi.readthedocs.io/).

## Quick start

```python
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.solvers.baseline import run_physics_baseline
from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.data.experimental_phasing import simulate_mir, mir_phase_indication

st = generate_random_organic(n_atoms=8, seed=0)
for method in ["random", "patterson", "direct_methods", "charge_flipping"]:
    print(run_physics_baseline(st, method=method, d_min=1.2, n_iter=80, verbose=False).summary())

# MIR hybrid features
mir = simulate_mir(st, heavy_element="AU", n_heavy=1, d_min=1.5)
phase_est, fom = mir_phase_indication(mir.F_native, mir.F_derivative, mir.F_heavy)
```

### CLI

```bash
gps-download-cod
gps-baseline --synthetic --n-atoms 8 --method direct_methods --dmin 1.2
gps-baseline --cif data/raw/cod/2100301.cif --method charge_flipping
python scripts/run_phase1_baseline.py
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
