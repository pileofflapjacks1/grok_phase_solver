# grok_phase_solver

**AI-driven general solver for the X-ray crystallography phase problem**

Recover phases \(\varphi(hkl)\) from experimental amplitudes \(|F(hkl)|\) using first-principles physics, classical iterative algorithms, and (Phase 2+) deep learning—extending ideas from [PhAI](https://doi.org/10.1126/science.adn2777) toward a general open-source tool for small molecules to macromolecules.

\[
\rho(\mathbf{r})
=
\frac{1}{V}\sum_{\mathbf{h}}
|F(\mathbf{h})|\,e^{i\varphi(\mathbf{h})}\,e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}
\]

## Status

**Phase 1 (current):** data pipeline, physics baselines, metrics, COD samples, PhAI integration hooks, math documentation.

| Component | State |
|-----------|--------|
| CIF / HKL I/O (gemmi) | ✅ |
| Structure factors + density FFT | ✅ |
| Charge flipping / HIO | ✅ |
| Metrics (MPE, map CC, FSC, R) | ✅ |
| COD download + samples | ✅ |
| Synthetic data (minimal) | ✅ |
| PhAI weights | 🔌 interface only (ERDA archive) |
| Neural training | 🚧 Phase 2 |

## Install

```bash
cd grok_phase_solver
python -m pip install -e ".[dev]"
# Optional ML stack (Phase 2):
# python -m pip install -e ".[ml]"
```

Requires Python ≥ 3.10, NumPy, SciPy, Matplotlib, [gemmi](https://gemmi.readthedocs.io/).

## Quick start

```python
from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.solvers.baseline import run_physics_baseline
from grok_phase_solver.data.synthetic import generate_random_organic

# Synthetic ab initio test
st = generate_random_organic(n_atoms=8, seed=0)
result = run_physics_baseline(st, method="charge_flipping", d_min=1.2, n_iter=100)
print(result.summary())

# From COD CIF
st = load_cif("data/raw/cod/2100301.cif")
result = run_physics_baseline(st, method="charge_flipping", d_min=1.2)
print(result.summary())
```

### CLI

```bash
# Download curated COD samples (CIF + HKL when available)
gps-download-cod

# Run a baseline
gps-baseline --synthetic --n-atoms 8 --method charge_flipping --dmin 1.2
gps-baseline --cif data/raw/cod/2100301.cif --method charge_flipping

# Full Phase-1 demonstration
python scripts/run_phase1_baseline.py
```

## Repository layout

```text
src/grok_phase_solver/
  io/           # CIF, HKL, ReflectionTable
  physics/      # form factors, Fcalc, density, reciprocal geometry
  solvers/      # charge flipping, HIO, baseline pipeline
  metrics/      # phase error, map CC, FSC, R-factor
  data/         # COD download, synthetic generation
  models/       # PhAI interface (weights optional)
data/
  raw/cod/      # sample CIFs / HKL
  processed/    # baseline JSON outputs
docs/
  math/         # phase problem overview & proofs
  roadmap.md
notebooks/      # math + baseline walkthrough
third_party/phai/  # where to place official PhAI weights
tests/
scripts/
```

## Physics baselines

1. **Charge flipping** (Oszlányi–Sütő) — flip low density; reimpose \(|F|\)  
2. **HIO** (Fienup) — hybrid input–output with positivity  
3. **Random phases** — null baseline for metrics  

Every neural component will keep a physics fallback (project principle).

## PhAI integration

Official code and weights:  
https://erda.ku.dk/archives/e3be15d017d8c4fe81402da833e26894/published-archive.html  

See `third_party/phai/README.md` and `grok_phase_solver.models.phai_interface`.

```bibtex
@article{Larsen2024PhAI,
  author  = {Larsen, Anders S. and Rekis, Toms and Madsen, Anders {\O}.},
  title   = {PhAI: A deep-learning approach to solve the crystallographic phase problem},
  journal = {Science},
  year    = {2024},
  doi     = {10.1126/science.adn2777}
}
```

## Sample data (COD)

| COD ID | System | Role |
|--------|--------|------|
| [2100301](https://www.crystallography.net/cod/2100301.html) | C₇H₅NO₄, P2₁/c | Small organic, PhAI-relevant SG |
| [2017775](https://www.crystallography.net/cod/2017775.html) | Roxithromycin, P2₁2₁2₁ | Larger molecule + experimental HKL |

Data from the [Crystallography Open Database](https://www.crystallography.net/).

## Documentation

- [Math overview](docs/math/phase_problem_overview.md) — Fourier analysis, constraints, algorithms, metrics  
- [Roadmap](docs/roadmap.md) — Phases 1–4  
- [Notebook 01](notebooks/01_math_and_baseline.md) — worked pipeline  

## Tests

```bash
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).  
Third-party data/models retain their original licenses and citation requirements.

## Contributing

Iterate: **Plan → Code → Test → Analyze math → Refine → Commit.**  
Open issues/PRs for new solvers, space groups, or loss functions with physics derivations.
