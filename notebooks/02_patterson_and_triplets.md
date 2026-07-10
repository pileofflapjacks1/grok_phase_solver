# Notebook 02 — Patterson function & triplet invariants

**Reference:** Kevin Cowtan, *Phase Problem in X-ray Crystallography, and Its Solution*
(Encyclopedia of Life Sciences, 2001). Local PDF: user-provided `phaseproblem.pdf`.

Run from repo root after `pip install -e ".[dev]"` (or set `sys.path` as below).

---

## 0. Setup

```python
import sys
from pathlib import Path
ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.solvers.baseline import structure_to_fcalc
from grok_phase_solver.physics.patterson import (
    patterson_from_amplitudes,
    autocorrelation_density,
    find_patterson_peaks,
    interatomic_vectors_from_atoms,
)
from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.solvers.direct_methods import (
    normalize_E,
    build_triplets,
    direct_methods_solve,
    figure_of_merit_triplets,
)
from grok_phase_solver.metrics.phase_error import wrap_phase
```

---

## 1. The phase problem (Cowtan)

Each structure factor is a wave \(F(\mathbf{h})=|F|e^{i\varphi}\). Experiment measures
**amplitudes** only. Electron density needs phases:

$$
\rho(\mathbf{r})=\frac{1}{V}\sum_{\mathbf{h}}|F(\mathbf{h})|e^{i\varphi(\mathbf{h})}e^{-2\pi i\mathbf{h}\cdot\mathbf{r}}.
$$

Constructive vs destructive interference of waves from atoms determines spot intensities
(Cowtan Fig. 1). Lattice periodicity → discrete reciprocal lattice of spots (Fig. 2).

---

## 2. Derivation of the Patterson function

### 2.1 Definition from intensities

Set all phases to zero and Fourier-transform \(|F|^2\):

$$
P(\mathbf{u})=\frac{1}{V}\sum_{\mathbf{h}}|F(\mathbf{h})|^2\,e^{-2\pi i\mathbf{h}\cdot\mathbf{u}}.
$$

### 2.2 Autocorrelation identity

Start from \(\rho\) and its Fourier coefficients \(F(\mathbf{h})\). The autocorrelation is

$$
P(\mathbf{u})=\int_V\rho(\mathbf{r})\,\rho(\mathbf{r}+\mathbf{u})\,d\mathbf{r}.
$$

Substitute the Fourier series for \(\rho\), integrate term-by-term, use orthogonality of
plane waves, and obtain \(P\leftrightarrow |F|^2\) (Wiener–Khinchin / Parseval).  
**Phases cancel:** only \(|F|^2\) appears.

### 2.3 Interatomic vectors

For atomic \(\rho(\mathbf{r})=\sum_j n_j\,\delta(\mathbf{r}-\mathbf{r}_j)\) (idealized),

$$
P(\mathbf{u})\propto\sum_{i,j}n_i n_j\,\delta\bigl(\mathbf{u}-(\mathbf{r}_i-\mathbf{r}_j)\bigr).
$$

Peaks at **every** interatomic vector \(\mathbf{r}_i-\mathbf{r}_j\), including the origin peak
(\(i=j\)). Number of non-origin vectors: \(N(N-1)\) → overlaps kill interpretability for
\(N\gtrsim 20\)–\(50\) unless heavy atoms dominate (Cowtan).

### 2.4 Numerical check: \(P(|F|)\) vs \(\mathrm{autocorr}(\rho)\)

```python
st = generate_random_organic(n_atoms=5, seed=1)
data = structure_to_fcalc(st, d_min=1.0)
hkl, F, amp = data["hkl"], data["F"], data["amplitudes"]

P = patterson_from_amplitudes(hkl, amp, st.cell, remove_origin=False)
rho = density_from_structure_factors(hkl, F, st.cell, shape=P.shape)
P_auto = autocorrelation_density(rho)

# Correlate maps
def cc(a, b):
    a, b = a.ravel(), b.ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-16))

print("CC(Patterson, autocorr ρ) =", cc(P, P_auto))

z = P.shape[2] // 2
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(P[:, :, z].T, origin="lower", cmap="viridis")
ax[0].set_title("P(u) from |F|²")
ax[1].imshow(P_auto[:, :, z].T, origin="lower", cmap="viridis")
ax[1].set_title("autocorr(ρ)")
plt.tight_layout(); plt.show()

peaks = find_patterson_peaks(P, n_peaks=10)
print("Top Patterson peaks (fractional):")
for p in peaks[:5]:
    print(f"  {p.rank}: {p.fract}, height={p.height:.3g}")
```

---

## 3. Triplet (three-phase) invariants

### 3.1 Idea (Cowtan / Cochran 1952)

Positivity + atomicity: if one strong reflection is phased, atoms prefer certain
regions; other strong reflections must reinforce those peaks. The strongest relation
involves **three** reflections with Miller indices summing to zero:

$$
\mathbf{h}+\mathbf{k}+(-\mathbf{h}-\mathbf{k})=\mathbf{0}.
$$

### 3.2 Structure invariant

$$
\Phi_{\mathbf{hk}}=\varphi(\mathbf{h})+\varphi(\mathbf{k})+\varphi(-\mathbf{h}-\mathbf{k}).
$$

Origin shift \(\varphi'(\mathbf{h})=\varphi(\mathbf{h})-2\pi\mathbf{h}\cdot\mathbf{t}\) leaves \(\Phi\) unchanged
(invariant). For strong normalized amplitudes \(E\), Cochran:

$$
P(\Phi)\propto \exp\bigl(\kappa\cos\Phi\bigr),\qquad
\kappa \approx 2\,N^{-1/2}|E_{\mathbf{h}}E_{\mathbf{k}}E_{\mathbf{h}+\mathbf{k}}|,
$$

so \(\Phi\approx 0\) (mod \(2\pi\)) with reliability growing in \(|EEE|\).

### 3.3 Tangent formula (sketch)

Rearrangement of many triplet estimates:

$$
\tan\varphi_{\mathbf{h}}\approx
\frac{\sum_{\mathbf{k}}|E_{\mathbf{k}}E_{\mathbf{h}-\mathbf{k}}|\sin(\varphi_{\mathbf{k}}+\varphi_{\mathbf{h}-\mathbf{k}})}
{\sum_{\mathbf{k}}|E_{\mathbf{k}}E_{\mathbf{h}-\mathbf{k}}|\cos(\varphi_{\mathbf{k}}+\varphi_{\mathbf{h}-\mathbf{k}})}.
$$

Multi-solution: randomize starting phases, iterate, rank by FOM \(\langle\cos\Phi\rangle\).

### 3.4 Numerical experiment

```python
E = normalize_E(hkl, amp, st.cell)
strong_idx, E_s, triplets = build_triplets(hkl, E, e_min=1.0, max_reflections=80)
print(f"Strong reflections: {len(strong_idx)}, triplets: {len(triplets)}")

# True invariants
phi = data["phases"]
true_phi = []
for t in triplets[:200]:
    ih, ik, im = strong_idx[t.i_h], strong_idx[t.i_k], strong_idx[t.i_hpk]
    true_phi.append(phi[ih] + phi[ik] - phi[im])
true_phi = wrap_phase(np.array(true_phi))
print("Mean |true Φ| (deg) for top triplets:", np.rad2deg(np.mean(np.abs(true_phi))))

plt.hist(np.rad2deg(true_phi), bins=36, range=(-180, 180))
plt.xlabel("True triplet invariant Φ (deg)")
plt.ylabel("count")
plt.title("Cochran: strong triplets concentrate near 0")
plt.show()

dm = direct_methods_solve(hkl, amp, st.cell, n_atoms_approx=5, n_trials=40, seed=0, verbose=True)
print("Best triplet FOM:", dm.history["best_fom"])
```

---

## 4. When classical methods fail (Cowtan summary → hybrid AI)

| Method | Works when | Fails when | AI hybrid angle |
|--------|------------|------------|-----------------|
| Patterson | Few / heavy atoms | \(N\gtrsim 50\), weak HA | Learn peak→coordinate; HA substructure nets |
| Direct methods | Atomic res., \(N\lesssim 10^2\)–\(10^3\) | Low res. (~2–3 Å proteins) | Triplet graph nets; E-value transformers |
| MIR | Isomorphous HA derivatives | Non-isomorphism | Predict Δφ from multi-crystal \|F\| channels |
| MAD | Anomalous scatterers, precision | Tiny Bijvoet diffs | Multi-λ inputs; SeMet site models |
| MR | Homologous model | Remote homology | AlphaFold pose + AI phase polish |
| Phase improvement | Partial phases exist | Wrong hand/envelope | Solvent flatten as NN residual |

See `docs/hybrid_ai_tests.md` for concrete test protocols.

---

## 5. Classical baseline one-liners

```python
from grok_phase_solver.solvers.baseline import run_physics_baseline

for method in ["random", "patterson", "direct_methods", "charge_flipping"]:
    res = run_physics_baseline(st, method=method, d_min=1.0, n_iter=80, verbose=False)
    print(res.summary(), res.notes)
```
