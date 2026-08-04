# Release notes — v0.12.0

**Theme:** GraphPhaseNet v10 multipath depth + research generative structure proposal.

## Highlights

1. **GraphPhaseNet v10** (`d_in=30`) — hop3 multipath, multipath span, Wilson B proxy, E-outlier ratio; stronger κ edges; bin CE 0.24.  
   Train: `python scripts/run_strong_prior_v10.py --quick --melgalvis-preset large`

2. **Melgalvis realism** — optional B-factor inflation (radiation-damage-ish) and amplitude noise hooks on the synthetic curriculum.

3. **Generative structure (research)** — CF density peaks → Fcalc phase seed; CLI `--method generative_structure` (**not** used by `auto`). Physics fallback retained.

4. **Honesty** — Strict success unchanged; 30% seed bar **not claimed** unless scoreboard says YES; no PhAI/SHELX/GraPhAI redistribution.

## Install

```bash
python -m pip install -e ".[dev,gui]"
pytest -q
python scripts/run_strong_prior_v10.py --quick --melgalvis-preset large
```

## Citations

- Melgalvis & Rekis (2026). Acta Cryst. A / GraPhAI (JACS 2026 conceptual).
- Carrozzini et al. (2025). J. Appl. Cryst. 58, 1859–1869 (AI-PhaSeed).
- Generative diffraction models (XDXD / PXRDGen / XRDSol): conceptual only; no weights shipped.
