# GraphPhaseNet v9 (v0.11)

v9 extends the residual GNN + κ-gated multipath stack with **d_in=26** features
and Melgalvis **large-cell / HA** curricula.

## Features (feature_version=8)

| Block | d | Cues |
|-------|---|------|
| v4 | 10 | E, s, amp, hkl, deg, E² |
| v5 | +4 | shell rank, E·deg, local E, shell |F| |
| v6 | +4 | HA E-tail, low-res, E·low-res, κ-centrality |
| v7 | +4 | hop2 E, edge √(EE), Wilson E residual, centro-HA |
| **v8** | **+4 → 26** | **log Vol, shell E std, κ×E, low-res·rank·HA** |

Edges: stronger multipath κ×E reweight (κ power ≥1.55, self-loop ≥0.12).

## Train

```bash
# laptop smoke
python scripts/run_strong_prior_v9.py --quick --melgalvis-preset large

# HA-focused
python scripts/run_strong_prior_v9.py --pilot --melgalvis-preset ha

# cluster
python scripts/run_strong_prior_v9.py --scale-xl --melgalvis-preset large \
  --continue-from data/processed/strong_prior_v9.npz
```

## Reporting

Hold-out reports **frac≤20°**, seedOK, strong MPE, strict solve rate, plus
stratification by HA / max Z and **Vol bands** (lt1000 / 1000–3500 / gt3500).

## Honest limits

- Laptop pilots remain in the ~20–25% frac≤20° band unless scale-xl clears more.
- **30% oracle bar is not claimed** unless the committed scoreboard says YES.
- Official GraPhAI weights are external only (`graphai_external.md`).
- Hard strict solves without partial information remain rare.
