# CrystalX-inspired peak → atom typing (v0.13)

## Purpose

After density peak picking, assign **element labels** and optional **hydrogens**
so `trial.res` is closer to a refinable SHELXL model.

Inspired by CrystalX (arXiv 2410.13713 / JACS 2026 conceptual equivariant
Transformer). This package ships a **pure-NumPy geometric/height heuristic**
with a documented path for future equivariant weights (not redistributed).

## Code

- `pipeline/crystalx_typing.py` — `type_peaks_crystalx`, `typed_atoms_to_shelxl_res`
- Wired into `pipeline/export.py` → `trial.res` + `typed_atoms.csv`

## Fallback

If typing fails, export falls back to untyped C/Q peaks (`write_shelxl_res`).

## Non-claims

- Not trained CrystalX weights.
- Labels are operational heuristics — always refine in Olex2/SHELXL.
