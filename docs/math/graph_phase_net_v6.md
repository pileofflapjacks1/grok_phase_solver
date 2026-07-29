# GraphPhaseNet v6 (GraPhAI HA-aware features)

## Motivation

Melgalvis & Rekis (JACS / Acta 2026 lineage) show that **physics-informed
diffraction graphs** (strong-|E| nodes, κ-weighted triplets) carry useful
structure for centrosymmetric and heavy-atom (Z≥19) organics. v5 already used
shell-rank / local-E / κ-gated edges. **v6** adds four HA/low-res cues without
claiming a general phase-problem solution.

## Node features (`d_in=18`)

| Index | Feature | Role |
|------:|---------|------|
| 0–9 | v4 base | E, s, amp, hkl, deg, E² |
| 10–13 | v5 | shell_rank, log1p(E·deg), local_E, shell |F| |
| 14 | `ha_E_tail` | soft strong-|E| tail (HA-sensitive outliers) |
| 15 | `low_res_w` | 1 − s_norm (low-s weight) |
| 16 | `E·low_res` | couples strong E with low resolution |
| 17 | `κ_centrality` | incident triplet-κ mass |

## Edges

Same triplet graph as v5, with slightly stronger κ power-law reweight (1.35)
and residual self-loops (0.08) for message-passing stability.

## Training

```bash
# laptop pilot
python scripts/run_strong_prior_v6.py --pilot --melgalvis-preset cod

# cluster scale
python scripts/run_strong_prior_v6.py --scale-xl --melgalvis-preset cod \
  --continue-from data/processed/strong_prior_v6.npz
```

Scoreboard: `data/processed/strong_prior_v6.{npz,json,md}`

## Honest limits

- Synthetic hold-out frac≤20° may remain near the ~20–25% plateau on laptop
  pilots; **30% oracle bar is not assumed cleared** unless the scoreboard says so.
- Hard ab initio strict solves without partial-φ remain rare.
- Physics fallback: ensemble / CF / partial_phaseed always available.
