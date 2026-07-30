# External GraPhAI reference (optional)

**Paper:** Melgalvis & Rekis, JACS 2026 — graph neural net for phase retrieval
on centrosymmetric structures with Z≥19, reported high success to ~2 Å on their
panel. Code/data on Zenodo (user download).

**This package:** GraphPhaseNet v6/v7 implements *compatible ideas* (diffraction
graph, κ/physics edges, HA/low-res features, Melgalvis-style synthetics). It does
**not** ship GraPhAI weights or claim numerical parity.

## Suggested local benchmark workflow

1. Download GraPhAI release from Zenodo (follow their license).
2. Export strong-phase predictions on a shared synthetic or COD subset.
3. Compare with GraphPhaseNet via `metrics/strong_seed.full_and_strong_metrics`
   (frac≤20°, strong MPE OI, seedOK).
4. Record results under `data/processed/` without committing third-party weights.

## Physics fallback

When external models are unavailable or fail, use:

- `gps-solve --method ensemble` (easy)
- `gps-solve --method partial_phaseed` + seeds (hard)
- Classical CF / RAAR / SHELXS runners (external binaries)
