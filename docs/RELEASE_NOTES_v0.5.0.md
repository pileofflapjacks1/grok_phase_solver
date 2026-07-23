# v0.5.0 — grok-phase-solver

**Diffusion hybrid (experimental), fuller SG support, predicted-model seeding,
device backend, phase UQ.** MIT.

## Highlights

- **Diffusion hybrid** — physics Langevin reverse process (`diffusion_hybrid` /
  `diffusion_phaseed`); CLI `--diffusion`
- **Space group** — `physics/symmetry.py` (gemmi expand, centro, absences)
- **Predicted models** — `--predicted-model` / improved `gps-make-seed --from-cif`
  (AF / OpenFold3 / Boltz-style CIF + SG expansion)
- **Device** — `--device cpu|cuda|mps|auto`, `--gpu`; optional torch FFT
- **UQ** — multistart circular phase confidence + free-FOM bootstrap in report.md

## Honest limits

- Hard ab initio seed bar still ~21–22% ≤20°
- Diffusion is experimental (no trained equivariant weights)
- Partial-φ remains the pragmatic hard-data path
- Not a general protein ab initio solver

## Install

```bash
python -m pip install -U grok-phase-solver
python -m pip install "grok-phase-solver[gpu]"   # optional torch device
```

## Links

| | |
|--|--|
| **Repo** | https://github.com/pileofflapjacks1/grok_phase_solver |
| **Math** | docs/math/diffusion_phase.md · docs/math/symmetry.md |
