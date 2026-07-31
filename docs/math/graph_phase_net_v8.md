# GraphPhaseNet v8 curriculum / stratified reporting (v0.10)

v8 uses the **same d_in=22 architecture and bin CE as v7**, with training
and evaluation improvements:

1. **HA-heavy Melgalvis preset** (`--melgalvis-preset ha`) for Z≥19-friendly
   curricula (GraPhAI success regime).
2. **Wilson match default** retained for new training runs.
3. **Stratified hold-out** by HA-bearing / max Z / organic light
   (`metrics/stratified_prior.py`).
4. **scale-xl** path documented for 5k–10k cluster runs.

## Train

```bash
# laptop pilot with HA curriculum
python scripts/run_strong_prior_v8.py --quick --melgalvis-preset ha

# cluster
python scripts/run_strong_prior_v8.py --scale-xl --melgalvis-preset ha \
  --continue-from data/processed/strong_prior_v8.npz
```

## Honest limits

- Architecture alone has not cleared the **30% ≤20°** bar on laptop pilots.
- Stratified HA cohorts may look better than all-organic hard P1 — report both.
- Official GraPhAI weights remain external (see `graphai_external.md`).
