# Wilson domain-gap report (close-the-gap)

Compare **synthetic** |F| to **experimental** COD, then apply Wilson-slope + shell-mean + quantile matching (`data/wilson_match.py`).

- Synthetic structures: **8** (easy≈4, hard≈4)
- Experimental reference: **COD_2017775_exp_dmin1.2**
- Training template saved: `data/processed/wilson_ref_template.npz`

## Raw gap (before matching)

| Cohort | mean gap | std | min | max |
|--------|----------|-----|-----|-----|
| easy | 2.136 | 1.035 | 1.125 | 3.575 |
| hard | 11.743 | 11.832 | 3.917 | 32.170 |
| all | 6.939 | 9.675 | 1.125 | 32.170 |

## After full matching (slope + shells + quantiles + noise)

| Cohort | mean gap before | mean gap after | reduction | frac ↓ |
|--------|-----------------|----------------|-----------|--------|
| easy | 2.136 | 2.286 | -0.151 | 7% |
| hard | 11.743 | 2.532 | 9.211 | 60% |
| all | 6.939 | 2.264 | 4.675 | 37% |

## Ablations (hard cohort)

| Recipe | mean gap after | reduction frac |
|--------|----------------|----------------|
| `slope_only` | 2.504 | 58% |
| `slope_shells` | 2.787 | 53% |
| `full` | 2.532 | 60% |

## Example hard structure

- Gap 32.170 → **2.963** (91% reduction)
- Wilson B: 44.0 → target ~14.3 (after match B_a=13.4)

Hard-vs-hard self gap (sanity): **25.755**

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
