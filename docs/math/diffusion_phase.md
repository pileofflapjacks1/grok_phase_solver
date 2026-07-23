# Diffusion-style phase completion (experimental)

## Status

**Research / experimental (v0.5).** Physics-first Langevin reverse process on the
phase circle with positivity + modulus projection. Optional neural score network
is **not** shipped. Inspired by score-based inverse problems and recent
diffusion-for-diffraction work (PXRDnet, XRDSol concepts) — **not** a reimplementation
and **not** claimed to match their metrics.

Prefer production paths:

- Easy: `ensemble`
- Hard with partial info: `partial_phaseed` / `gps-make-seed`
- AI-PhaSeed: `phai_phaseed` + optional `--ai-dm-hybrid`

## Algorithm

```text
φ ← seed (or random on unknown reflections)
for t = T … 1:   # σ_t high → low
    ρ ← IFFT(|F| e^{iφ});  ρ ← positivity (+ optional solvent)
    φ_data ← arg(FFT(ρ)) with |F| reimposed; blend toward seed
    φ ← circular blend(φ, φ_data; data_weight)
    φ ← φ + N(0, σ_t²)   # wrap to (−π, π]
final clean physics step; optional free-FOM–gated CF polish
```

Implementation: `models/diffusion_phase.py`

| API | Role |
|-----|------|
| `reverse_diffusion_phases` | core annealed reverse process |
| `diffusion_hybrid_solve` | multistart + free-FOM polish |
| `conditional_diffusion_complete` | seed-conditioned wrapper (v0.4 compat) |

## Physics checks

- Modulus projection each step (data fidelity)
- Positivity (atomicity proxy)
- Seed re-imposition weight (partial-φ consistency)
- Free FOM v2.1 gates final polish

## CLI

```bash
gps-solve --hkl data.hkl --ins data.ins --method diffusion_hybrid --n-diffusion-steps 20
gps-solve --hkl data.hkl --ins data.ins --diffusion --predicted-model model.cif
# or: --method diffusion_phaseed with a seed source
```

## Honest limits

- No trained equivariant diffusion weights (XRDSol-style) in this package.
- Hard ab initio seed bar (~30% strong ≤20°) is **unchanged**.
- Multistart agreement can be jointly wrong — see `metrics/uncertainty.py`.

## References

1. Score-based generative models for inverse problems (Song et al. lineage).
2. PXRDnet (Nature Materials 2025) — powder / map completion (conceptual).
3. XRDSol (Nature Comm. 2026) — equivariant diffusion phasing (conceptual).
4. Project: `docs/math/ai_phaseed.md`, `free_fom.md`, `partial_seed.md`.
