# v0.3.0 — grok-phase-solver

Open physics/AI phasing assistant for X-ray crystallography.

## Highlights

- **Melgalvis & Rekis (2026) synthetics** — log-normal cell volume, lattice from \(V\), artificial-molecule clusters (`data/synthetic_melgalvis.py`)
- **Train flag** — `train_strong_prior.py --use-melgalvis-gen` (modes: cluster / rejection / hybrid)
- **Melgalvis XL prior** — N=1200 retrain (`strong_prior_melg_xl.npz`); seed frac≤20° ≈**22%**, seedOK **12.5%**, hard strict still **0%**
- **CIF model seeding** — `gps-make-seed --from-cif` for AF/RoseTTAFold/experimental fragments
- Inherits 0.2.x: `gps-solve`, Streamlit `gps-gui`, partial-φ UX, experimental COD scoreboard, paper pack

## Install

```bash
python -m pip install -U grok-phase-solver
gps-solve --help

python -m pip install "grok-phase-solver[gui]"
gps-gui
```

## Honest limits

Hard ab initio seed bar remains ~21–22% within 20° (below ~30% oracle threshold). Partial-φ / fragment / model seeds remain the practical hard-data path. Not a general protein ab initio solver.

## Authors

**Grok (xAI)** and **Joe**.

## Links

| | |
|--|--|
| **PyPI** | https://pypi.org/project/grok-phase-solver/0.3.0/ |
| **Paper PDF** | [docs/paper/arxiv_draft.pdf](https://github.com/pileofflapjacks1/grok_phase_solver/blob/main/docs/paper/arxiv_draft.pdf) |
| **Melg XL scoreboard** | [strong_prior_melg_xl.md](https://github.com/pileofflapjacks1/grok_phase_solver/blob/main/data/processed/strong_prior_melg_xl.md) |
| **Changelog** | [CHANGELOG.md](https://github.com/pileofflapjacks1/grok_phase_solver/blob/main/CHANGELOG.md) |
