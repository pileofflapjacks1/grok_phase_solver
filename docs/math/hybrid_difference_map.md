# Hybrid Difference Map (HDM-style) — research path (v0.10)

## Motivation

2026 literature on Hybrid Difference Maps blends **Difference Map** relaxation
in ordered (protein) density with **HIO-style** feedback in solvent. This can
stabilize envelope-aware phase retrieval when solvent fraction is significant.

## Implementation

`solvers/iterative_retrieval.hybrid_difference_map_solve`

1. Estimate solvent fraction (histogram + optional Vol/N prior; protein_mode).
2. Build solvent mask (low local density).
3. Each iteration:
   - DiffMap update → candidate `x_dm`
   - HIO feedback outside protein support → candidate `x_hio`
   - Blend: protein voxels from DM, solvent voxels from HIO
4. Final modulus projection → phases + density

## CLI / pipeline

```bash
gps-solve --hkl data.hkl --ins data.ins --method hdm --n-iter 150 -o ./out
```

**Not** selected by `auto`. Marked `research_only` in diagnostics.

## Fallbacks

- Small-molecule production: `ensemble`, `charge_flipping`, `raar`
- Hard data: `partial_phaseed` + seeds
- Classical DiffMap: `difference_map_solve` (no solvent split)

## Honesty

Experimental. Not a claim of general macromolecular ab initio solution or
parity with published HDM software. Document residual free-FOM / mapCC only.
