# Iterative projection algorithms for phase retrieval

## Setup

Seek density \(\rho\) such that the Fourier moduli match observations:

\[
|(\mathcal{F}\rho)(\mathbf{h})| = |F_{\mathrm{obs}}(\mathbf{h})|.
\]

Let \(P_M\) be the **modulus projector** (replace \(|F|\) by \(|F_{\mathrm{obs}}|\), keep phase) and \(P_S\) a **real-space projector** (positivity \(P_+\) or charge flip \(P_{\mathrm{CF}}\)).

Reflectors: \(R_S = 2P_S - I\), \(R_M = 2P_M - I\).

## Algorithms

### Error reduction (ER)
\[
x_{n+1} = P_S P_M x_n.
\]
Can stagnate at non-global fixed points.

### Hybrid Input–Output (HIO, Fienup 1982)
Feedback outside the support/positivity set (see `hio.py`).

### RAAR (Luke 2005)
\[
x_{n+1}
=
\frac{\beta}{2}(R_S R_M + I)\,x_n
+
(1-\beta)\,P_M x_n,
\qquad \beta\in(0,1].
\]
Often more robust than plain HIO for non-convex feasibility.

### Difference Map (Elser 2003)
With \(\gamma_S = 1/\beta\), \(\gamma_M = -1/\beta\):
\[
\begin{aligned}
f_M &= (1+\gamma_M)P_M - \gamma_M I,\\
f_S &= (1+\gamma_S)P_S - \gamma_S I,\\
x_{n+1} &= x_n + \beta\bigl(P_S f_M - P_M f_S\bigr)x_n.
\end{aligned}
\]

### Charge flipping (Oszlányi–Sütő)
Real-space map \(P_{\mathrm{CF}}\rho = \mathrm{sign flip\ when\ }\rho<\delta\), alternated with \(P_M\).

## Implementation notes

- Iterate in **real space** (density grid); \(P_M\) via FFT with package conventions in `solvers/projectors.py`.
- Always end with a modulus projection so reported phases match \(|F_{\mathrm{obs}}|\).
- Convergence is **not guaranteed**; use multistart and free FOMs (`free_fom.py`).

## Direct methods upgrade

Cochran reliability for equal atoms:
\[
\kappa(\mathbf{h},\mathbf{k}) = 2 N^{-1/2}\,|E_{\mathbf{h}} E_{\mathbf{k}} E_{\mathbf{h}+\mathbf{k}}|.
\]
Tangent formula weights use \(\kappa\) and \(I_1(\kappa)/I_0(\kappa) = \mathbb{E}[\cos\Phi]\).

## Conditional hybrid

Apply classical polish to a neural/classical seed only if the **truth-free composite FOM** increases (`conditional_hybrid.py`). Prevents CF from destroying a good PhAI prior (observed in fair PhAI benchmarks).
