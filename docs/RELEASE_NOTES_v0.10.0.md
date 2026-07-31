# v0.10.0 — grok-phase-solver

**HA curriculum + stratified prior reporting, Hybrid Difference Map (research), AI-PhaSeed filters.** MIT.

## Install

```bash
pip install -U grok-phase-solver==0.10.0
# or from source
pip install -e ".[gui,dev]"
```

## Scientific / engineering gains

| Area | Change |
|------|--------|
| Graph prior | v8 curriculum (`ha` preset), stratified Z/HA hold-out tables |
| Iterative | Experimental **HDM** (`--method hdm`): DiffMap protein + HIO solvent |
| AI-PhaSeed | Bin-quality seed filter; |E| floor; Vol-band seed fraction |
| Data | `ha_heavy_config`; COD stratified bench skeleton |
| Docs | `hybrid_difference_map.md`, `graph_phase_net_v8.md` |

## Honest limits

- Graph prior still below **30% ≤20°** bar on laptop pilots.
- **HDM is research-only** — not default `auto`.
- Hard pure ab initio still rare; use partial-φ / fragment / HA.
- No external GraPhAI/PhAI/SHELX redistribution.

## How to test

```bash
pytest -q
python scripts/run_strong_prior_v8.py --quick --melgalvis-preset ha
python scripts/run_cod_stratified_bench.py
gps-solve --hkl examples/demo_solve/demo.hkl --ins examples/demo_solve/demo.ins \
  --method hdm --n-iter 40 -o /tmp/hdm_demo
```

## Cluster scale

```bash
python scripts/run_strong_prior_v8.py --scale-xl --melgalvis-preset ha \
  --continue-from data/processed/strong_prior_v8.npz
```
