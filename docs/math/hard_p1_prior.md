# Domain-matched hard-P1 phase prior

## Why

PhAI is trained largely for COD-like / \(P2_1/c\) statistics. On **random
synthetic P1 hard cells** (\(n \ge 12\), \(d_{\min} \ge 1.5\,\text{Å}\)) it
shows a **domain gap**: seed mapCC ~0.2–0.4 and AI-PhaSeed does not cross the
solvability cliff.

This module trains a **PhaseMLP only on hard (+bridge) P1 synthetic data** so
the AI-PhaSeed seed matches the evaluation domain.

## Training (origin-invariant)

Absolute phases are only defined up to origin (and enantiomorph). Naive
supervised \(\varphi\) MSE collapses to ~90° MPE (random).

**Protocol:**

1. Sample P1 organics: hard \(n \in [12,20]\), \(d_{\min} \in [1.5,2.0]\);
   every 4th sample is a **bridge** (\(n \in [8,12)\), \(d_{\min} \in [1.2,1.5)\)).
2. Global feature mean/std over all training reflections.
3. Each SGD step: among discrete origin shifts of \(\varphi_{\mathrm{true}}\)
   (and enantiomorph), pick the target that best matches the current network
   output; backprop to that target.
4. Up-weight strong reflections (top ~60% by \(|F|\)) — these form the PhaSeed set.
5. Inference: network phases → free-FOM origin/enantiomorph search → AI-PhaSeed.

## Results (this repo, ~60 structures, hidden 128)

| Metric | Naive absolute train | Origin-invariant train |
|--------|----------------------|-------------------------|
| Hold-out MPE (OI) | ~91° | **~64°** |
| Prior-only mapCC | ~0.3–0.4 | **~0.46–0.53** |
| hard_p1 + AI-PhaSeed mean mapCC | ~0.41 | **~0.52** |
| CF mean mapCC (same hold-out) | ~0.50 | ~0.50 |
| Strict solved (hold-out) | 0/8 | 0/8 |

**Interpretation:** domain matching + origin-invariant loss yields a **weak but
real** prior (clearly better than random + PhaSeed ~0.35 mapCC; competitive
with CF on mapCC). A small per-reflection MLP is **not** enough to solve hard
P1 under strict SuccessThresholds — that still needs a larger architecture
and/or more information (see failure taxonomy B+C).

## API

```python
from grok_phase_solver.models.hard_p1_prior import (
    train_hard_p1_prior,
    hard_p1_phaseed_solve,
    predict_phases_hard_p1,
    load_hard_p1_prior,
)

model, meta = train_hard_p1_prior(n_structures=80)
phases, rho, info = hard_p1_phaseed_solve(hkl, amp, cell, model=model)
```

```bash
python scripts/train_hard_p1_prior.py
# → data/processed/hard_p1_prior.{npz,json,md}
```

## Relation to goal

| Step | Status |
|------|--------|
| Free FOM not ranking false atomicity over truth | Done (v2.1) |
| AI-PhaSeed extension algorithm | Done (oracle/partial seeds solve hard) |
| Domain-matched prior for hard P1 | **Done (weak MLP prior)** |
| Strong in-domain prior (PhAI-scale / equivariant) | Open |
| Experimental multi-SG scoreboard | Open |
