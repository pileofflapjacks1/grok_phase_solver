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

## XDXD-inspired coordinate proposal (v0.13)

`xdxd_propose_coordinates`: multi-start CF density → peak atoms → Fcalc seed,
ranked by modulus R. Primary product is **trial fractional coordinates**.

```bash
gps-solve --hkl data.hkl --ins data.ins --method xdxd_structure --out ./out
```

## Non-claims

- No trained generative / XDXD weights are redistributed.
- Not a general ab initio solution of the phase problem.
- Full SE(3) equivariant atomic diffusion remains a stub (`diffusion_se3_stub`).

## Coordinate denoiser (`gps-generate`, research, not default)

Code: `src/grok_phase_solver/generate/`. This is a different track from the
weight-free proposal above and from `gps-solve --method diffusion_hybrid`.

The input is a known composition, cell, space group, and Z. The coordinate set
is the non-hydrogen asymmetric unit. Z is recorded and is not expanded into
extra copies. Training uses known structures only (a frozen COD subset already
in the repo, or the single CIF passed to `gps-generate`).

### Objective

Let \(x_0\) be the fractional coordinates. Draw \(\sigma\) log-uniform on
\([0.02, 0.45]\) and \(\varepsilon \sim \mathcal{N}(0, I)\). The noisy coordinates are

\[
x_t = (x_0 + \sigma \varepsilon) \bmod 1.
\]

`CoordDenoiser` is a small MLP. Each atom is an element embedding plus \(\sigma\),
a scaled cell, and a Gaussian-weighted average of minimum-image offsets to the
other atoms. The MLP predicts a fractional shift. The last linear layer is
initialized at zero, so an untrained step predicts no change. The training loss
is the mean squared minimum-image error

\[
L = \mathrm{mean}\,\lVert \mathrm{wrap}(\hat{x}_0 - x_0) \rVert^2,
\quad \mathrm{wrap}(\delta) = \delta - \mathrm{round}(\delta).
\]

There is no \(|F|\) term. Sampling starts from uniform fractional coordinates
and steps \(\sigma\) from 0.45 down to 0.02. The last step emits \(\hat{x}_0\).

If this model is less often clash-free than uniform draws that are rejected
only for a 1.0 Å overlap, the scoreboard says so. The cutoff is not moved.

### Checks

Validity is an expanded-cell minimum-image distance of at least 1.0 Å and an
exact element multiset. Uniqueness, among valid samples of one cell, is a
greedy same-element fractional RMSD of at least 0.15 after centroid alignment.
Reconstruction is held-out coordinate recovery (half the non-H sites within
0.75 Å on a 0.25 origin grid). It is not novelty. The structure-factor residual
compares \(|F_{\mathrm{calc}}|\) of the proposal with \(|F_{\mathrm{calc}}|\) of
the deposited sites, both at \(B = 5\,\text{Å}^2\) and \(d_{\min} = 1.5\,\text{Å}\).

### Still not claimed

No DFT, VASP, or LAMMPS. No flow-matching result. No foundation model.
No synthesizability or stability (there is no energy model). Novelty is not
measured. This does not design AR/VR materials and does not replace phasing.
