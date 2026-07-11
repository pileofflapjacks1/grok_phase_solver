# Free figures of merit (truth-free phase ranking)

## Why free FOMs?

In real experiments (and conditional hybrid decisions) we do **not** have ground-truth phases. Ranking trials or accepting a classical polish after a neural seed requires diagnostics that depend only on \(|F_{\mathrm{obs}}|\) and a candidate \(\varphi\).

## Critical constraint: modulus is already enforced

Given observed amplitudes, any phased map

\[
\rho = \mathcal{F}^{-1}\!\big(|F_{\mathrm{obs}}|\,e^{i\varphi}\big)
\]

already satisfies the Fourier modulus constraint (up to numerics and Friedel completion). Therefore

\[
R\big(|\mathcal{F}\rho|,\,|F_{\mathrm{obs}}|\big) \approx 0
\]

**for every phase set**. A residual measured *after* modulus re-projection is **vacuous**.

### Bug fixed in free-FOM v2

v1 computed:

```text
ρ → positivity → FFT → project_modulus → R(·, |F_obs|)   # always ≈ 0
```

v2 computes the **positivity residual**:

\[
R_+
=
R\big(\,|\mathcal{F}\,\max(\rho,0)|,\; |F_{\mathrm{obs}}|\,\big).
\]

Clipping negatives changes moduli. Atomic, nearly positive maps keep small \(R_+\); random phases produce large negative density whose clipping wrecks agreement with \(|F_{\mathrm{obs}}|\).

This is the density-side analogue of classical direct-methods FOMs that measure **self-consistency of a physical model** not already hard-enforced in the iterate.

## Component diagnostics

| Symbol | Definition | Good map |
|--------|------------|----------|
| \(R_+\) | positivity residual (above) | low |
| excess kurtosis | \(\langle z^4\rangle - 3\) of \(\rho\) | high (peaked) |
| peakiness | \(\max\rho/\sigma\) and mass in top 5% voxels | high |
| skewness | \(\langle z^3\rangle\) | positive |
| pos_frac | fraction \(\rho\ge 0\) | moderate (weak weight) |
| shell \(R_+\) | \(R_+\) in resolution shells | low, esp. high-res |
| \(R_{\mathrm{Sayre}}\) | \(1-\langle\cos(\varphi_h+\varphi_k-\varphi_{h+k})\rangle_w\) on strong \(|F|\) | lower |

### Why positivity fraction is down-weighted

Fcalc maps at incomplete resolution are **not** purely positive; charge-flipping solutions intentionally keep sign flips. Over-weighting \(\mathrm{pos\_frac}\) ranked CF above true phases in early diagnostics. Atomicity + \(R_+\) are more discriminative.

## Composite score

Each raw metric \(m\) is mapped to a score \(s(m)\in(0,1]\) (logistic / \(1/(1+R)\)). The composite is a weighted sum (default weights in `DEFAULT_WEIGHTS`):

\[
C
=
\sum_k w_k\, s_k
\quad
(+\;\text{light shell / Sayre blend}).
\]

Higher \(C\) is better. Weights are calibrated to maximize rank agreement with origin-invariant mapCC on synthetic + COD Fcalc (`scripts/calibrate_free_fom.py`).

## Conditional polish gate

Accept polish only if:

1. \(C_{\mathrm{after}} \ge C_{\mathrm{before}} + \delta\) (default \(\delta=0.02\))
2. \(R_{+,\mathrm{after}} \le R_{+,\mathrm{before}} + \varepsilon\) (default \(\varepsilon=0.03\))
3. **Rewrite trust-region:** if phase displacement
   \[
   D_\varphi = \langle 1 - \cos(\varphi_{\mathrm{after}}-\varphi_{\mathrm{before}})\rangle_{|F|}
   \]
   exceeds \(D_{\mathrm{thr}}\) (default 0.5), require a **substantial** residual drop
   \(R_{+,\mathrm{before}} - R_{+,\mathrm{after}} \ge \eta\) (default \(\eta=0.08\)).

Rules 2–3 block the COD 2016452 pathology: CF can raise composite and lower \(R_+\)
*slightly* while **destroying** a good PhAI prior at 1.2–2.0 Å. Helpful polish
(PhAI+CF @ 0.9 Å) improves \(R_+\) by ~0.12 and is accepted; harmful polishes
improve \(R_+\) only by ~0.01–0.06 under large \(D_\varphi\) and are rejected.

## Limitations (honest)

- Free FOMs are **proxies**, not oracles. Origin ambiguity, enantiomorph, and incomplete models limit density statistics.
- At very low resolution, atomicity signals weaken; neural priors may dominate.
- Always validate with refinement R-factors on experimental data.
- Sayre / triplet terms assume approximate atomic equal-atom structure.

## Code

- Implementation: `solvers/free_fom.py`
- Gate: `should_accept_polish` → used by `conditional_hybrid.py`, ensemble ranking
- Calibration: `scripts/calibrate_free_fom.py` → `data/processed/free_fom_calibration.md`

## References

- Fienup, J. R. (1982). *Appl. Opt.* 21, 2758 — residual after support/positivity.
- Oszlányi & Sütő (2004). Charge flipping — dynamic \(\delta\), weak-phase perturbation.
- Sayre, D. (1952). *Acta Cryst.* 5, 60 — squaring method / atomicity.
- Classical direct-methods FOMs (ABSFOM, etc.) — reciprocal-space consistency analogues.
