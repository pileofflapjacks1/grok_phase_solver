# Coordinate generation scoreboard

Written by `scripts/run_generate_scoreboard.py`. Do not edit the rates by hand.

This track proposes fractional coordinates for a known composition and cell. It does not discover materials and it does not phase a structure.

This track does not run DFT, VASP, or LAMMPS. It does not implement a flow-matching paper, train a foundation model, or claim synthesizability. It does not design materials for AR or VR. It does not replace `gps-solve` or solve the phase problem. `gps-solve --method diffusion_hybrid` is a different experimental flag (Langevin phase completion), not this coordinate model, and it is not the production default.

## Definitions

- **Validity.** A sample is valid when the symmetry-expanded cell has no minimum-image pair closer than 1.0 Å and the element multiset equals the requested composition.
- **Uniqueness.** A valid sample is unique when, after fractional-centroid alignment, its greedy same-element minimum-image RMSD to every earlier unique sample of the same cell is at least 0.15.
- **Reconstruction.** A held-out structure is recovered when at least one sample places at least half of its non-hydrogen sites within 0.75 Å after the best shift on a 0.25 fractional origin grid. This is not novelty.
- **Novelty.** not measured.
- **Stability / synthesizability.** not measured — no energy model.

## Protocol

- Seed 0. Denoiser training steps 40. Random and denoiser samples per cell: 8. Diffusion samples per cell: 1.
- Clash cutoff 1.0 Å on the symmetry-expanded cell. Uniqueness cutoff 0.15 fractional RMSD after centroid alignment. Reconstruction uses a 0.25 origin grid and 0.75 Å, and only the holdout.
- Structure-factor residual: both sides use B = 5 Å² and d_min = 1.5 Å. Reference amplitudes are Fcalc of the deposited coordinates, not measured Fobs.
- Holdout: 2200000. Training structures are not counted as reconstruction.
- Wall time 9.0 s.

## Cohort

Included: 2016452 (non-H 8, train), 2100301 (non-H 12, train), 2200000 (non-H 18, holdout)

Excluded: 2012000 (non-H count 27 outside 4–20), 2013000 (non-H count 30 outside 4–20), 2017775 (non-H count 59 outside 4–20)

## Results

| method | n | validity | uniqueness | reconstruction | novelty | stability / synthesizability |
|--------|---|----------|------------|----------------|---------|------------------------------|
| random + clash rejection | 24 | 24/24 (1.000) | 24/24 (1.000) | 0/1 (0.000) | not measured | not measured — no energy model |
| coordinate denoiser | 24 | 4/24 (0.167) | 4/4 (1.000) | 0/1 (0.000) | not measured | not measured — no energy model |
| diffusion_hybrid (experimental, not default) | 3 | 0/3 (0.000) | 0/0 | 0/1 (0.000) | not measured | not measured — no energy model |

## Per structure

| structure | holdout | method | n_valid | n_unique | mean min distance (Å) | best match fraction | mean R (Fcalc) |
|-----------|---------|--------|---------|----------|-----------------------|---------------------|----------------|
| 2016452 | no | random + clash rejection | 8/8 (1.000) | 8/8 (1.000) | 1.191 | 0.250 | 0.776 |
| 2100301 | no | random + clash rejection | 8/8 (1.000) | 8/8 (1.000) | 1.098 | 0.167 | 0.738 |
| 2200000 | yes | random + clash rejection | 8/8 (1.000) | 8/8 (1.000) | 1.155 | 0.222 | 0.625 |
| 2016452 | no | coordinate denoiser | 3/8 (0.375) | 3/3 (1.000) | 0.865 | 0.250 | 0.760 |
| 2100301 | no | coordinate denoiser | 1/8 (0.125) | 1/1 (1.000) | 0.650 | 0.167 | 0.742 |
| 2200000 | yes | coordinate denoiser | 0/8 (0.000) | 0/0 | 0.699 | 0.222 | 0.606 |
| 2016452 | no | diffusion_hybrid (experimental, not default) | 0/1 (0.000) | 0/0 | 0.881 | 0.250 | 0.693 |
| 2100301 | no | diffusion_hybrid (experimental, not default) | 0/1 (0.000) | 0/0 | 0.653 | 0.083 | 0.622 |
| 2200000 | yes | diffusion_hybrid (experimental, not default) | 0/1 (0.000) | 0/0 | 0.933 | 0.111 | 0.513 |

## Notes

- Uniform fractional coordinates. A draw is kept when the expanded cell has no pair closer than 1.0 Å, up to 80 tries. Composition is the requested multiset.
- Coordinate denoiser. Loss is mean squared minimum-image error of the predicted clean fractional coordinates. No |F| term. Last linear layer starts at zero. Trained 40 steps on 2016452, 2100301. Final loss 0.047963.
- Experimental phase-path flag `diffusion_hybrid`, not the gps-generate default and not the coordinate denoiser. Amplitudes are Fcalc of the deposited structure at B=5 Å² and d_min=1.5 Å, so this row is oracle-|F| conditioned. Peaks are unlabeled; the requested element list is copied in CIF order only when at least as many peaks as atoms are found. Fewer peaks are not padded, and composition_match is then false. Settings: n_steps=2, n_starts=1, n_polish=2, use_learned_score=false. One sample per cell.

## Baseline comparison

The coordinate denoiser validity is 0.167, below the random-plus-clash baseline at 1.000. The denoiser is trained with a minimum-image coordinate loss and never sees |F|. A few dozen steps on the training asymmetric units do not enforce the 1.0 Å clash cutoff, while the random baseline rejects draws that contain such a pair. The cutoff was not changed to hide that. Mean minimum distance was 0.738 Å (denoiser) and 1.148 Å (random).

