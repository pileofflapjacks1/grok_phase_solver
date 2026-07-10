# Notebook 03 — Parseval, Friedel, uniqueness limits

```python
import sys
from pathlib import Path
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.physics.parseval import parseval_check, friedel_check
from grok_phase_solver.physics.patterson import patterson_from_amplitudes, autocorrelation_density
from grok_phase_solver.physics.density import density_from_structure_factors
```

## Parseval diagnostic

```python
st = generate_random_organic(n_atoms=6, seed=0)
data = structure_to_fcalc(st, d_min=1.0)
rep = parseval_check(data["hkl"], data["F"], st.cell)
print(rep)
# relative_error can be O(0.1–1) with incomplete reciprocal sampling — document, don't hide
```

## Friedel law for real density

```python
print(friedel_check(data["hkl"], data["F"]))
# max_err should be ~ numerical noise
```

## Trivial associate: origin shift does not change |F|

```python
t = np.array([0.1, 0.2, 0.3])
phase_shift = 2 * np.pi * (data["hkl"].astype(float) @ t)
F2 = data["F"] * np.exp(-1j * phase_shift)
assert np.allclose(np.abs(F2), np.abs(data["F"]))
rho1 = density_from_structure_factors(data["hkl"], data["F"], st.cell)
rho2 = density_from_structure_factors(data["hkl"], F2, st.cell, shape=rho1.shape)
# maps differ by translation — same structure
print(rho1.max(), rho2.max())
```

See `docs/math/uniqueness_and_bounds.md` for theorems and non-claims.
