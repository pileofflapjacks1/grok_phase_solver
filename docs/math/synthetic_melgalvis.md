# Melgalvis & Rekis (2026) synthetic generation

## Citation

Melgalvis, D.M. & Rekis, T. (2026). *On artificial crystal structure generation
for solving the phase problem with deep learning.* **Acta Cryst. A** **82**, 32–40
(open access).

## Motivation

Uniform random cells and uncorrelated atom placement create a **domain gap**
versus experimental COD/CSD distributions. Melgalvis & Rekis show that
**volume-first lattice sampling** and **bonded artificial molecules** improve
deep-learning phase generalization to larger cells and experimental structures
(when combined with phase recycling / PhAI-style training).

## What we implement

Code: `src/grok_phase_solver/data/synthetic_melgalvis.py`

### 1. Log-normal unit-cell volume

Sample
\[
\log V \sim \mathcal{N}(\mu, \sigma^2)
\]
with defaults \(\mu=\log 450\), \(\sigma=0.55\) (Å³), truncated to
\([V_{\min}, V_{\max}]\) (~120–2500 Å³). Parameters are COD-inspired, not a
full CSD refit.

### 2. Lattice from volume

Given \(V\) and crystal system:

- **Orthorhombic:** sample axis ratios \(a/b\), \(c/b\), set \(b\) so
  \(abc=V\), random axis permutation.
- **Monoclinic:** sample \(\beta\in[92^\circ,125^\circ]\), use \(V=abc\sin\beta\).
- **Triclinic:** sample angles, use full volume formula
  \(V=abc\sqrt{1-\cos^2\alpha-\cos^2\beta-\cos^2\gamma+2\cos\alpha\cos\beta\cos\gamma}\).

### 3. Artificial-molecule clusters

- Seed atom with element drawn from empirical organic frequencies
  (general vs special-position tables).
- Grow non-H atoms by attaching at covalent-radius bond lengths.
- Clash rejection using min-image Cartesian distances.
- Optional H addition on C/N/O.
- Isotropic \(U_{\mathrm{iso}}\) sampled in \([0.01,0.10]\) Å².
- Density prior: volume per non-H atom \(\sim U(7,22)\) Å³ blended with \(\log V\).
- Optional near-inversion special-position seed + partial inversion images.

### 4. Modes

| Mode | Description |
|------|-------------|
| `cluster` | Melgalvis-style molecules + volume lattice |
| `rejection` | Legacy random atoms with volume-informed \(V/\mathrm{atom}\) |
| `hybrid` | Mix (~70% cluster / 30% rejection) |

## Training integration

```bash
python scripts/train_strong_prior.py --scale --wilson-match --use-melgalvis-gen
python scripts/train_strong_prior.py --quick --use-melgalvis-gen --melgalvis-mode cluster
```

`iter_hard_multsg_samples(..., use_melgalvis_gen=True)` in
`models/strong_prior.py`. Training shard mode: `write_training_shard(..., mode="melgalvis")`.

## Relation to Wilson matching

Wilson amplitude matching (`wilson_match.py`) remains complementary: Melgalvis
improves **geometry / chemistry statistics**; Wilson match improves **|F|
shells** toward experimental templates.

## Honest scope

This is a **faithful engineering subset** of Melgalvis & Rekis for open
training loops—not a full reproduction of their PhAI retrain numbers. Our
hard-region seed bar (~30% within 20°) may still require partial φ; improved
synthetics aim to **raise** ab initio seed quality and generalization, not
claim a solved phase problem.
