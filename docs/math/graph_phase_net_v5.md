# GraphPhaseNet v5 (strong prior)

## Motivation

Melgalvis & Rekis (2026) style synthetic crystals + diffraction-graph ML
emphasize Wilson-aware features and physics edges (triplets / κ). v5 enriches
the existing residual GraphPhaseNet without abandoning NumPy-first training.

## Node features

| Version | d_in | Features |
|---------|------|----------|
| v4 | 10 | E, s, s², \|h\|, amp, hkl norms, deg, E² |
| **v5** | **14** | v4 + shell_rank, log1p(E·deg), local_E_mean, \|F\|/⟨\|F|⟩_shell |

Edges: undirected triplet pairs with **κ-gated** reweight (clip 0.25–4× median).

## Training

```bash
# Pilot / laptop (v0.7 hard curriculum)
python scripts/run_strong_prior_v5.py --n-structures 200 \
  --melgalvis-preset hard --low-res-frac 0.2

# Larger (cluster)
python scripts/run_strong_prior_v5.py --scale --melgalvis-preset cod
python scripts/run_strong_prior_v5.py --scale-xl --melgalvis-preset hard
```

Defaults: Melgalvis generator, hard/COD presets, Wilson match, strong-|E| loss,
Adam residual GNN, v5.1 κ power-law edges.

## Scoreboard

See `data/processed/strong_prior_v5.md` for hold-out frac≤20° and seedOK.

**Honest:** pure ab initio hard cells still rarely strict-solve; partial-φ
remains the practical hard-data path. Legacy plateau was ~21–22% frac≤20°.

## Code

- `models/graph_phase_net.py` — features + message passing
- `models/strong_prior.py` — train / predict / PhaSeed wrap
- `scripts/run_strong_prior_v5.py` — CLI
