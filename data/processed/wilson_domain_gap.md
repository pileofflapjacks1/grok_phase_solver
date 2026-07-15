# Wilson domain-gap report (close-the-gap)

Compare **synthetic** |F| to **experimental** COD, then apply Wilson-slope + shell-mean + quantile matching (`data/wilson_match.py`).

- Synthetic structures: **20** (easy≈10, hard≈10)
- Experimental reference: **COD_2017775_exp_dmin1.2**
- Training template saved: `data/processed/wilson_ref_template.npz`

## Raw gap (before matching)

| Cohort | mean gap | std | min | max |
|--------|----------|-----|-----|-----|
| easy | 2.185 | 1.069 | 0.820 | 3.695 |
| hard | 9.499 | 2.901 | 3.779 | 13.236 |
| all | 5.842 | 4.261 | 0.820 | 13.236 |

## After full matching (slope + shells + quantiles + noise)

| Cohort | mean gap before | mean gap after | reduction | frac ↓ |
|--------|-----------------|----------------|-----------|--------|
| easy | 2.185 | 2.251 | -0.066 | -0% |
| hard | 9.499 | 2.839 | 6.660 | 64% |
| all | 5.842 | 2.503 | 3.340 | 33% |

## Ablations (hard cohort)

| Recipe | mean gap after | reduction frac |
|--------|----------------|----------------|
| `slope_only` | 3.301 | 58% |
| `slope_shells` | 3.018 | 61% |
| `full` | 2.839 | 64% |

## Example hard structure

- Gap 13.236 → **2.822** (79% reduction)
- Wilson B: 26.3 → target ~14.3 (after match B_a=15.7)

Hard-vs-hard self gap (sanity): **4.640**

## How to use in training

```bash
# rebuild template + report
python scripts/run_wilson_domain_gap.py

# train GraphPhaseNet with matched |F|
python scripts/train_strong_prior.py --scale  # set wilson_match=True in API
```

```python
from grok_phase_solver.data.wilson_match import close_wilson_gap, load_reference_template
from grok_phase_solver.models.strong_prior import train_strong_prior

model, meta = train_strong_prior(n_structures=100, wilson_match=True)
```

Phases always come from Fcalc truth; only **amplitudes** are matched. This keeps labels correct while aligning input statistics to experiment.

JSON: `data/processed/wilson_domain_gap.json`
