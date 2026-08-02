# Release notes — v0.11.0

**Theme:** Melgalvis large-cell synthetics + GraphPhaseNet v9 (d_in=26) + AI-PhaSeed hardening.

## Highlights

1. **Synthetic generator (Melgalvis & Rekis 2026 style)**  
   Ring scaffolds, void/short-contact packing, `large_cell` preset (Vol ~1000–3500 Å³), stronger HA Z≥19 bias. See `docs/math/synthetic_melgalvis.md`.

2. **GraphPhaseNet v9**  
   d_in=26 features + multipath edges; scoreboard `data/processed/strong_prior_v9.*`; train via  
   `python scripts/run_strong_prior_v9.py --quick --melgalvis-preset large`.

3. **AI-PhaSeed**  
   Multi-bin entropy filter, \|E\| floor, Vol-band seed fraction, Class 0/1 diagnostics markdown.

4. **Honesty preserved**  
   Strict success unchanged; 30% seed bar **not claimed** unless scoreboard says YES; no PhAI/SHELX/GraPhAI redistribution.

## Install

```bash
python -m pip install -e ".[dev,gui]"
pytest -q tests/test_v011_melgalvis_graph_seed.py tests/test_synthetic_melgalvis.py
python scripts/run_strong_prior_v9.py --quick --melgalvis-preset large
```

## Citations

- Melgalvis & Rekis (2026). Acta Cryst. A 82, 32–40.
- Carrozzini et al. (2025). J. Appl. Cryst. 58, 1859–1869.
- GraPhAI / related 2026 graph–diffraction work: external weights only (`docs/math/graphai_external.md`).
