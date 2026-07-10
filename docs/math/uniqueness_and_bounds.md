# Uniqueness, oversampling, and honest limits

## 1. What is rigorously true

### 1.1 Fourier inversion with phases known

If complex structure factors \(F(\mathbf{h})\) are known for all \(\mathbf{h}\in\mathbb{Z}^3\) (or sufficiently many),

\[
\rho(\mathbf{r})=\frac{1}{V}\sum_{\mathbf{h}}F(\mathbf{h})\,e^{-2\pi i\mathbf{h}\cdot\mathbf{r}}
\]

recovers \(\rho\) exactly (as a tempered distribution / band-limited function). This is **not** the phase problem.

### 1.2 Parseval / Plancherel

\[
\frac{1}{V}\int_V|\rho(\mathbf{r})|^2\,d\mathbf{r}
=
\frac{1}{V^2}\sum_{\mathbf{h}}|F(\mathbf{h})|^2.
\]

Implemented diagnostics: `physics.parseval.parseval_check`. Relative error grows when reflections are incomplete or the FFT grid undersamples reciprocal space.

### 1.3 Friedel law (no anomalous scattering)

For real \(\rho\): \(F(-\mathbf{h})=\overline{F(\mathbf{h})}\).  
Anomalous scattering (MAD) **breaks** equality of magnitudes \(|F(\mathbf{h})|\neq|F(-\mathbf{h})|\) near edges.

### 1.4 Patterson uniqueness failure

\(P=\rho\star\rho\) is determined by \(|F|^2\). Distinct \(\rho\) (homometric sets) can share \(P\).  
Number of interatomic vectors scales as \(N(N-1)\) → peak overlap (Cowtan).

### 1.5 Triplet concentration (equal-atom random model)

Cochran: for normalized \(E\),

\[
P(\Phi)\propto e^{\kappa\cos\Phi},\quad
\kappa\sim 2 N^{-1/2}|E_{\mathbf{h}}E_{\mathbf{k}}E_{\mathbf{h}+\mathbf{k}}|.
\]

This is a **probabilistic** relation, not an identity. It weakens as \(N\) grows and as resolution drops (fewer large \(E\)).

### 1.6 CDI support oversampling (not crystals)

For compact support and Fourier magnitudes on a finer-than-Nyquist grid, uniqueness results exist up to trivial associates (translation, conjugate inversion) under mild conditions (Hayes et al.).  
**Crystals sample critically** on \(\mathbb{Z}^3\). Atomicity replaces support oversampling.

## 2. Trivial associates (always free)

1. Origin: \(\varphi'(\mathbf{h})=\varphi(\mathbf{h})-2\pi\mathbf{h}\cdot\mathbf{t}\)  
2. Enantiomorph: \(\varphi'=-\varphi\) (non-centrosymmetric)  
3. Global phase for continuous diffraction (not discrete crystal SF in standard setting)

Metrics **must** be origin/enantiomorph invariant for ab initio comparisons.

## 3. Algorithmic status (truth table)

| Method | Guarantees | Failure mode |
|--------|------------|--------------|
| Patterson | Correct interatomic vector map from \(\|F\|^2\) | Ambiguous inversion to coordinates when \(N\) large |
| Direct methods | Heuristic multi-solution | Low res / large \(N\); local minima |
| Charge flipping | Empirical success small molecules | Non-convex; not globally convergent |
| HIO | Empirical CDI success | Needs support / positivity; stagnation |
| MIR/MAD | Information-theoretic phase info from experiments | Non-iso, noise, weak anomalous |
| Neural nets | Learn \(p(\varphi\mid\|F\|)\) on training measure | Domain gap; no general uniqueness proof |

## 4. What this repository does **not** claim

- We have **not** solved the phase problem for arbitrary macromolecules.
- Our PhaseMLP is a **supervised synthetic toy**; overfit on train structures does not imply experimental transfer.
- PhAI weights are external; without them we do not reproduce Science 2024 numbers here.
- Blow–Crick implementation is the classical Gaussian lack-of-closure model, not SHARP.

## 5. Correct path forward

1. Keep physics projections (\(|F|\) reimposition) in every hybrid loop.  
2. Measure origin-invariant map CC + FSC.  
3. Expand training measure toward experimental COD/PDB distributions (Wilson domain-gap).  
4. Treat MIR/MAD/MR multi-channel data as extra information, not magic.  
5. Publish negative results (when CF/DM/NN fail) with the same care as successes.
