# Wilson domain-gap report

Compare **synthetic** |F| statistics to **experimental** COD reflections.
Lower `domain_gap_score` ⇒ closer match (Wilson slope + intensity quantiles + moments).

- Synthetic structures: **20** (easy≈10, hard≈10)
- Experimental reference: **COD_2017775_exp_dmin1.2**

## Synthetic vs experimental

| Cohort | mean gap | std | min | max |
|--------|----------|-----|-----|-----|
| easy | 2.185 | 1.069 | 0.820 | 3.695 |
| hard | 9.499 | 2.901 | 3.779 | 13.236 |
| all | 5.842 | 4.261 | 0.820 | 13.236 |

## Synthetic vs COD Fcalc control (2016452 @ 0.9 Å)

- hard mean gap: **31.384**
- easy mean gap: **20.829**

Hard-vs-hard self gap (sanity): **4.640**

## Notes

- Large gap to experimental Fobs is expected (completeness, B-factor, disorder, measurement noise).
- Use this score when reweighting or filtering synthetic training shards.
- Code: `data/wilson.py` — `domain_gap_report`, `mean_domain_gap_vs_experiment`.

JSON: `data/processed/wilson_domain_gap.json`
