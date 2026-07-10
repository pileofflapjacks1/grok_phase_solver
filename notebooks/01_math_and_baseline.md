# Notebook 01 — Math foundations & Phase-1 baseline

This is a script-style notebook (run cells top-to-bottom in Jupyter after
`pip install -e ".[dev]"` from the repo root).

## 0. Setup

```python
import sys
from pathlib import Path
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from grok_phase_solver.io.cif import load_cif
from grok_phase_solver.data.synthetic import generate_random_organic, simulate_diffraction
from grok_phase_solver.physics.structure_factors import compute_structure_factors
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import generate_hkl, d_spacing
from grok_phase_solver.solvers.baseline import run_physics_baseline, structure_to_fcalc
from grok_phase_solver.metrics.phase_error import mean_phase_error_origin_invariant
```

## 1. Core equation

Electron density from phased structure factors:

$$
\rho(\mathbf{r}) = \frac{1}{V}\sum_{\mathbf{h}} |F(\mathbf{h})| e^{i\varphi(\mathbf{h})} e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}
$$

Phases \(\varphi\) are unknown from the experiment. We recover them under positivity and atomicity.

## 2. Synthetic Fcalc and density

```python
st = generate_random_organic(n_atoms=8, seed=0)
print(st.summary())
data = structure_to_fcalc(st, d_min=1.2)
print("Nrefl", len(data["amplitudes"]))
rho = density_from_structure_factors(
    data["hkl"], data["F"], st.cell, d_min=1.2, sampling=4.0
)
print("density shape", rho.shape, "max", rho.max(), "min", rho.min())

# Central slice
plt.figure(figsize=(5, 4))
plt.imshow(rho[:, :, rho.shape[2]//2].T, origin="lower", cmap="magma")
plt.title("True density slice (synthetic)")
plt.colorbar(label="ρ")
plt.show()
```

## 3. Parseval check (discrete)

```python
# Rough energy comparison: sum |F|^2 vs real-space variance * volume scaling
F = data["F"]
E_F = np.sum(np.abs(F)**2)
E_rho = np.sum(rho**2) * st.volume / rho.size
print("Σ|F|²", E_F)
print("∫ρ² proxy", E_rho)
print("ratio", E_rho / (E_F + 1e-16))
```

## 4. Baseline phase retrieval

```python
for method in ["random", "charge_flipping", "hio"]:
    res = run_physics_baseline(st, method=method, d_min=1.2, n_iter=100, seed=0)
    print(res.summary())
```

## 5. COD structure

```python
cif = ROOT / "data/raw/cod/2100301.cif"
if cif.exists():
    st2 = load_cif(cif)
    print(st2.summary())
    res = run_physics_baseline(st2, method="charge_flipping", d_min=1.2, n_iter=100)
    print(res.summary())
else:
    print("Download COD sample first: python -m grok_phase_solver.cli download or scripts/run_phase1_baseline.py")
```

## 6. Failure mode: low resolution

At low resolution, atomicity weakens and iterative methods fail more often.

```python
if cif.exists():
    for dmin in [0.9, 1.2, 1.5, 2.0, 2.5]:
        res = run_physics_baseline(
            st2, method="charge_flipping", d_min=dmin, n_iter=80, verbose=False
        )
        print(res.summary())
```

See `docs/math/phase_problem_overview.md` for full derivations.
