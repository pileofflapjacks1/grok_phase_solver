# GraphPhaseNet v7 (GraPhAI multipath + Carrozzini bins)

## Motivation

Melgalvis & Rekis **GraPhAI** (JACS 2026) shows efficient diffraction graphs
with physics-weighted edges succeed for many centrosymmetric / HA cases.
Carrozzini **AI-PhaSeed** (2025) shows discretizing phases into few bins
stabilizes seed selection. v7 combines both ideas without claiming parity
with published GraPhAI weights or general ab initio solution.

## Node features (`d_in=22`)

v6 (18) plus:

| Feature | Role |
|---------|------|
| hop2_local_E | 2-hop mean \|E\| |
| edge_E_geom | mean √(E_i E_j) on incident pairs |
| wilson_E_shell | E / shell-mean E |
| centro_HA_cue | ha_E_tail · low_res · shell_rank |

## Edges

κ power-law reweight × geometric mean of \|E\| on the triplet (multipath
reliability). Self-loop 0.10 for residual MP stability.

## Loss

- OI (cos, sin) MSE with strong-|E| weights + within-20° boost  
- Triplet Cochran invariant aux  
- **Carrozzini bin CE** (`bin_weight`, 4 bins or centro 0/π auto)

## Training

```bash
python scripts/run_strong_prior_v7.py --pilot --melgalvis-preset acta2026
python scripts/run_strong_prior_v7.py --scale-xl --melgalvis-preset cod \
  --continue-from data/processed/strong_prior_v7.npz
```

Scoreboard: `data/processed/strong_prior_v7.{npz,json,md}`

## Official GraPhAI weights

Official GraPhAI (Zenodo) is **not redistributed**. If you download weights
locally, treat them as an external reference model: compare seed metrics on
shared hold-outs; do not claim identity with GraphPhaseNet v7.

## Honest limits

- Laptop pilots may remain near mid-20% frac≤20°; **30% bar not assumed**.
- Strict hard ab initio still rare without partial-φ / HA.
- Physics fallbacks (ensemble, CF, partial_phaseed) always available.
