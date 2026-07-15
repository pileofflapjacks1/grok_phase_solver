# Closing the Wilson domain gap

## Problem

Synthetic Fcalc for hard cells has different **Wilson slopes**, shell intensities,
and amplitude tails than experimental Fobs. Training GraphPhaseNet / PhaseMLP on
raw Fcalc therefore sees a different input distribution than gps-solve will see
on real data.

Empirically (COD 2017775 Fobs, d ≥ 1.2 Å):

| Cohort | mean domain_gap_score (raw) |
|--------|------------------------------|
| easy synthetic | ~2 |
| hard synthetic | ~9–10 |

The score is dominated by **Wilson slope mismatch** (overall B), then moments.

## Solution

`data/wilson_match.py` transforms **amplitudes only** (phases stay Fcalc truth):

1. **Wilson slope match** — overall Debye–Waller: $|F|' = |F|\exp(-B_\mathrm{extra}s^2)$
   chosen so relative Wilson slope matches the reference.
2. **Shell mean match** — per-resolution-shell $\langle I\rangle$ scaling to reference profile.
3. **Quantile match** — rank-preserving histogram match of $|F|$ magnitudes.
4. **Measurement noise** — small relative Gaussian noise (experimental-like).

Template file: `data/processed/wilson_ref_template.npz` (built from experimental HKL).

## Usage

```bash
# Measure gap + build template + report ablations
python scripts/run_wilson_domain_gap.py

# Train with matched |F|
python scripts/train_strong_prior.py --scale --wilson-match
```

```python
from grok_phase_solver.data.wilson_match import close_wilson_gap, load_reference_template
from grok_phase_solver.models.strong_prior import train_strong_prior, iter_hard_multsg_samples

ref = load_reference_template()
model, meta = train_strong_prior(n_structures=100, wilson_match=True)

# Or per sample:
matched, report = close_wilson_gap(synth_pack, ref)
```

Also: `simulate_diffraction_wilson_matched` in `data/synthetic.py`.

## What this does *not* fix

- Chemical realism of random organics (fragment libraries help separately).
- Completeness / anisotropy / twinning of a specific experiment.
- The **seed quality cliff** for pure ab initio (still need ~30% good strong φ).

Wilson matching improves **domain transfer of amplitude features**; it does not
by itself invent correct phases.

## Metrics

`domain_gap_score` ≈ Wilson `slope_diff` + intensity quantile L1 + 0.25·moment gap.
After full matching, hard-cohort mean gap should drop by a large fraction
(see `data/processed/wilson_domain_gap.md` for the latest numbers).
