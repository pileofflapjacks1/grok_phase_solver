# GraPhAI (external — not redistributed)

Melgalvis & Rekis, *JACS* 2026 (GraPhAI): graph neural network on diffraction
graphs for small-molecule phasing, strong on Z≥19 centrosymmetric cases.

## Policy

**This repository does not ship GraPhAI code or weights.**

1. Obtain the official Zenodo package from the paper’s data availability statement.
2. Unpack to e.g. `~/models/GraPhAI` or `third_party/graphai/` (gitignored contents).
3. Set:
   ```bash
   export GRAPHAI_HOME=/path/to/GraPhAI
   ```
4. Use `grok_phase_solver.models.graphai_external` for discovery + H2H skeleton.
   Wire local inference in a user plugin when the API matches.

## In-repo alternative

GraphPhaseNet v5–v11 (`scripts/run_strong_prior_v*.py`) provides an open multipath
diffraction-graph prior with honest seed-quality scoreboards.

## Fair comparison

Use the same reflection subset, origin-invariant metrics, and success definition
as GraphPhaseNet scoreboards (`frac ≤ 20°` on strong |E|, strict mapCC/R1/peaks).
