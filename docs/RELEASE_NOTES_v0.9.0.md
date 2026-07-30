# v0.9.0 — grok-phase-solver

**GraphPhaseNet v7 (GraPhAI multipath + Carrozzini bins), synthetic packing, seed-quality cues.** MIT.

## Install

```bash
pip install -U grok-phase-solver==0.9.0
# or from source
pip install -e ".[gui,dev]"
```

## Scientific / engineering gains

| Area | Change |
|------|--------|
| Graph prior | **v7** d_in=22 multipath features; κ×E edges; Carrozzini bin CE training |
| Training | `run_strong_prior_v7.py` (quick/pilot/scale/scale-xl); presets `cod` / `hard` / `acta2026` |
| AI-PhaSeed | Quadrant discretization alias; multi-seed **bin agreement** boost |
| Seed quality | Features: bin entropy, mean \|cos φ\|, top-10% E; heuristic uses them |
| Synthetics | Multi-fragment packing; `actas2026_config` curriculum |
| DM | Solvent estimate uses volume/N_atom prior |
| Docs | `graph_phase_net_v7.md`, `graphai_external.md` (no weight redistribution) |

## Honest limits

- Laptop pilots may stay near mid-20% frac≤20°; **30% oracle bar not assumed cleared**.
- Hard pure ab initio strict solves remain rare; partial-φ / fragment / HA is the practical hard path.
- GraPhAI official weights are **external** (Zenodo); GraphPhaseNet is idea-aligned, not a reimplementation claim.
- Diffusion / SE(3) remain research-only, off default `auto`.

## How to test

```bash
pytest -q
python scripts/run_strong_prior_v7.py --quick --melgalvis-preset acta2026
python scripts/run_cod_hard_path_validation.py
gps-solve --help
```

## Cluster scale

```bash
python scripts/run_strong_prior_v7.py --scale-xl --melgalvis-preset cod \
  --continue-from data/processed/strong_prior_v7.npz
```
