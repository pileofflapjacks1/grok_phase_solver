# Mathematical Foundations of the Crystallographic Phase Problem

**grok_phase_solver** — Phase 1 math overview  
Last updated: 2026-07-10

---

## 1. The forward problem

A crystal is a 3-periodic electron density \(\rho(\mathbf{r})\) with unit-cell volume \(V\). The complex **structure factors** are the Fourier coefficients

\[
F(\mathbf{h})
=
\int_V \rho(\mathbf{r})\, e^{2\pi i \mathbf{h}\cdot\mathbf{r}}\,d\mathbf{r}
=
\bigl|F(\mathbf{h})\bigr|\, e^{i\varphi(\mathbf{h})},
\]

where \(\mathbf{h}=(h,k,l)\in\mathbb{Z}^3\) indexes the reciprocal lattice.

For an atomic model with \(N\) atoms at fractional coordinates \(\mathbf{r}_j\),

\[
F(\mathbf{h})
=
\sum_{j=1}^{N}
\mathrm{occ}_j\,
f_j(s)\,
e^{-B_j s^2}\,
e^{2\pi i \mathbf{h}\cdot\mathbf{r}_j},
\qquad
s=\frac{\sin\theta}{\lambda}=\frac{1}{2d_{hkl}}.
\]

Here \(f_j(s)\) is the atomic form factor and \(B_j\) the isotropic Debye–Waller factor.

## 2. The inverse problem (phase problem)

X-ray diffraction (kinematic theory) measures intensities \(I(\mathbf{h})\propto |F(\mathbf{h})|^2\). Phases \(\varphi(\mathbf{h})\) are lost. The inverse Fourier transform

\[
\rho(\mathbf{r})
=
\frac{1}{V}\sum_{\mathbf{h}}
|F(\mathbf{h})|\,
e^{i\varphi(\mathbf{h})}\,
e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}
\]

cannot be evaluated without \(\varphi\).

### 2.1 Ambiguities

Even with perfect amplitudes, solutions are unique only up to:

1. **Origin shift** \(\mathbf{t}\): \(\varphi'(\mathbf{h})=\varphi(\mathbf{h})-2\pi\mathbf{h}\cdot\mathbf{t}\)
2. **Enantiomorph** (non-centrosymmetric): \(\varphi'(\mathbf{h})=-\varphi(\mathbf{h})\) (inversion)
3. **Homometric structures** (rare discrete ambiguities)

Metrics in this package report **origin/enantiomorph-invariant** mean phase error when comparing to ground truth.

## 3. Physical constraints that make recovery possible

| Constraint | Mathematical form | Role |
|------------|-------------------|------|
| Positivity | \(\rho(\mathbf{r})\ge 0\) (X-ray electrons) | Convex set in real space |
| Atomicity | \(\rho\) is sparse sum of atomic peaks | Sayre / sparsity prior |
| Known composition | \(f_j\), approximate \(Z\), \(B\) | Forward model |
| Symmetry | Space group \(G\); systematic absences | Reduces free phases |
| Parseval | \(\frac{1}{V}\int\rho^2 = \frac{1}{V^2}\sum |F|^2\) | Energy consistency |
| Oversampling | finite support / solvent | Phase retrieval theory (CDI) |

### 3.1 Sayre's equation (atomicity)

For equal atoms at atomic resolution,

\[
F(\mathbf{h}) \approx \theta(s)\sum_{\mathbf{k}} F(\mathbf{k})\,F(\mathbf{h}-\mathbf{k}),
\]

which underlies classical **direct methods** (triplet phase relations).

### 3.2 Patterson function (phase-free)

\[
P(\mathbf{u})
=
\frac{1}{V}\sum_{\mathbf{h}} |F(\mathbf{h})|^2 e^{-2\pi i \mathbf{h}\cdot\mathbf{u}}
=
\int \rho(\mathbf{r})\,\rho(\mathbf{r}+\mathbf{u})\,d\mathbf{r}
\]

is the autocorrelation of \(\rho\). It is computable from intensities alone and encodes interatomic vectors.

## 4. Iterative projection algorithms

### 4.1 Fourier modulus projection \(\mathcal{P}_F\)

Given density estimate \(\rho\), replace Fourier moduli with observations:

\[
\mathcal{P}_F\rho
\;\longleftrightarrow\;
F'(\mathbf{h})
=
|F_{\mathrm{obs}}(\mathbf{h})|\,
\frac{\hat\rho(\mathbf{h})}{|\hat\rho(\mathbf{h})|}.
\]

### 4.2 Charge flipping \(\mathcal{P}_{\mathrm{CF}}\) (Oszlányi–Sütő)

\[
(\mathcal{P}_{\mathrm{CF}}\rho)(\mathbf{r})
=
\begin{cases}
-\rho(\mathbf{r}) & \rho(\mathbf{r}) < \delta,\\
\rho(\mathbf{r}) & \text{otherwise.}
\end{cases}
\]

Iterate \(\rho_{n+1} = \mathcal{P}_F\mathcal{P}_{\mathrm{CF}}\rho_n\). Implemented in `solvers/charge_flipping.py`.

### 4.3 Hybrid Input–Output (HIO, Fienup)

With support \(S\) (or positivity set),

\[
\rho_{n+1}(\mathbf{r})
=
\begin{cases}
(\mathcal{P}_F\rho_n)(\mathbf{r}) & \mathbf{r}\in S,\\
\rho_n(\mathbf{r})-\beta\,(\mathcal{P}_F\rho_n)(\mathbf{r}) & \mathbf{r}\notin S.
\end{cases}
\]

Implemented in `solvers/hio.py`.

### 4.4 Convergence remarks

These algorithms are **not** globally convergent in general (non-convex modulus constraint). Success depends on atomicity, resolution, completeness, and initialization. Failures are expected for:

- very low resolution (\(d_{\min}\gtrsim 2.5\)–\(3\,\text{Å}\) for small molecules without strong priors)
- severe incompleteness / missing wedges
- pseudo-centrosymmetry and twinning
- macromolecules without envelope / MR priors

Documenting these failure modes mathematically is a project goal.

## 5. Neural phase retrieval (PhAI and beyond)

PhAI (Larsen et al., *Science* 2024) learns an approximate posterior

\[
\hat\varphi
=
f_\theta\bigl(|F|;\,\text{SG},\,d_{\min},\,\ldots\bigr)
\]

from millions of synthetic \((|F|,\varphi)\) pairs, with **phase recycling** enforcing \(\mathcal{P}_F\) consistency.

Physics-informed losses we target in Phase 2:

\[
\begin{aligned}
\mathcal{L}_{\mathrm{phase}}
&=
\sum_{\mathbf{h}} w_{\mathbf{h}}
\bigl(1-\cos(\hat\varphi_{\mathbf{h}}-\varphi_{\mathbf{h}})\bigr),\\
\mathcal{L}_{+}
&=
\int \bigl(\min(\hat\rho,0)\bigr)^2\,d\mathbf{r},\\
\mathcal{L}_{\mathrm{atom}}
&=
-\int |\nabla^2\hat\rho|\,d\mathbf{r}
\quad\text{(peak sharpness / Laplacian)},\\
\mathcal{L}_{\mathrm{F}}
&=
\bigl\||\mathcal{F}\hat\rho|-|F_{\mathrm{obs}}|\bigr\|^2.
\end{aligned}
\]

## 6. Quality metrics

| Metric | Definition | Package |
|--------|------------|---------|
| Mean phase error (MPE) | \(\langle|\mathrm{wrap}(\hat\varphi-\varphi)|\rangle\) | `metrics.phase_error` |
| Origin-invariant MPE | \(\min_{t,\pm}\mathrm{MPE}\) | same |
| Map CC | Pearson corr. of \(\hat\rho,\rho_{\mathrm{true}}\) | `metrics.map_cc` |
| FSC | Shell-wise Fourier correlation | `metrics.map_cc` |
| \(R\)-factor | \(\sum\bigl||F_o|-k|F_c|\bigr|/\sum|F_o|\) | `metrics.rfactor` |

### 6.1 Fourier shell correlation (map quality)

\[
\mathrm{FSC}(s)
=
\frac{
\sum_{\mathbf{h}\in s}\mathrm{Re}\bigl(F_1 F_2^*\bigr)
}{
\sqrt{\sum|F_1|^2\sum|F_2|^2}
}.
\]

## 7. Uniqueness sketch (oversampling / support)

In coherent diffraction imaging, if \(\rho\) is supported on a set \(S\) whose convex hull is known and the Fourier transform is sampled more finely than the Nyquist rate of the autocorrelation (factor \(\ge 2\) linear oversampling in 2D/3D under mild conditions), uniqueness results of Hayes, Barakat, and others apply up to trivial associates. Crystals are **critically sampled** on the reciprocal lattice; uniqueness then relies on **atomicity** rather than support oversampling—hence direct methods and ML priors.

## 8. Parseval identity (implementation check)

\[
\frac{1}{V}\int_V \rho(\mathbf{r})^2\,d\mathbf{r}
=
\frac{1}{V^2}\sum_{\mathbf{h}}|F(\mathbf{h})|^2.
\]

Discrete FFT conventions in `physics/density.py` are chosen so that this holds approximately on fine grids; tests monitor relative energy error.

## 9. Related project notes

- [AI-PhaSeed](ai_phaseed.md) — Carrozzini 2025 hybrid (modified tangent, Class 0/1 seeds, EDM path)
- [Partial seeds](partial_seed.md) — hard-data φ / fragment / HA path
- [Free FOM v2.1](free_fom.md) · [Failure taxonomy](failure_taxonomy.md)
- [Melgalvis synthetics](synthetic_melgalvis.md) · [Strong prior](strong_prior.md)

## 10. Roadmap of theorems / notes to expand

1. Closed-form triplet phase expectations under random atom models  
2. Bounds relating Patterson peak resolution to phase error  
3. Convergence of charge flipping for sparse non-negative signals (connections to GESPAR / compressed sensing)  
4. Domain-gap metrics between synthetic and COD experimental \(|F|\) distributions  
5. Formal connection between free-FOM proxies and Carrozzini MPE_seed / CORR_seed under origin ambiguity  

---

*This document is the living theoretical companion to the code. Notebooks under `notebooks/` will add worked numerical proofs.*
