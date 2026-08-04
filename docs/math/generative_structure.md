# Generative structure proposal (research, v0.12)

## Purpose

Lightweight interface for **candidate model → phase seed** proposals conditioned
on |F|, cell, and optional composition — inspired by end-to-end generative
crystallography (XDXD-style) and diffraction diffusion/flow models (PXRDGen /
XRDSol conceptual lineage).

## What ships

Code: `src/grok_phase_solver/models/generative_structure.py`

1. Composition guess from cell volume (or user n_atoms / HA element).
2. Short charge-flipping density → peak pick → trial atoms.
3. Soft Fcalc phase seed from trial atoms (modulus projection retained).
4. Optional polish: more CF or pure-physics Langevin (`diffusion_phase`).

CLI (research-only, **not** used by `auto`):

```bash
gps-solve --hkl data.hkl --ins data.ins --method generative_structure --out ./out
```

## Physics fallback

Always available without learned weights. If CF peaks fail, random trial atoms
are used as a degraded path. Prefer:

- `auto` / `ensemble` on easy cells
- `partial_phaseed` / fragment / HA / predicted-model on hard cells
- `diffusion_hybrid` for experimental Langevin phase completion

## Non-claims

- No trained generative weights are redistributed.
- Not a general ab initio solution of the phase problem.
- Full SE(3) equivariant atomic diffusion remains a stub (`diffusion_se3_stub`).
