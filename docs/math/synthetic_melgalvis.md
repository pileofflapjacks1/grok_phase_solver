# Melgalvis & Rekis (2026) synthetic generation

## Citation

Melgalvis, D.M. & Rekis, T. (2026). *On artificial crystal structure generation
for solving the phase problem with deep learning.* **Acta Cryst. A** **82**, 32–40
(open access).

## v0.7–v0.11 curriculum extensions

| Preset / flag | Role |
|---------------|------|
| `cod_like_config()` | Log-normal volumes closer to COD organics; HA + partial occ |
| `hard_curriculum_config()` | Larger Z / volumes for hard / low-res panels |
| `ha_heavy_config()` | Z≥19-friendly Br/I bias (GraPhAI HA regime) |
| `actas2026_config()` | Acta 2026-style COD volumes + multi-fragment + rings |
| `large_cell_config()` | **v0.11** Vol ~1000–3500 Å³, multi-fragment, HA + rings |
| `p_heavy_atom` | Inject Br/Cl/S/P/I for HA-like partial-seed training |
| `p_partial_occupancy` | Random non-H occ ∈ [0.4, 0.9] |
| `p_ring_fragment` | **v0.11** phenyl / pyridine / carboxyl / imidazole scaffolds |
| `void_check` / `min_contact_frac` | **v0.11** packing quality (short contacts + voids) |
| `include_low_res` / train `--low-res-frac` | Force d_min ∈ [1.8, 2.5] Å fraction |
| CLI | `run_strong_prior_v9.py --melgalvis-preset large\|ha\|acta2026\|cod\|hard` |

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
\([V_{\min}, V_{\max}]\). Presets push the tail toward **~3500–4500 Å³**
(`large_cell`, `acta2026`, `cod`) for Carrozzini / Melgalvis larger-cell
regimes. Parameters are COD-inspired, not a full CSD refit.

### 2. Lattice from volume

Given \(V\) and crystal system:

- **Orthorhombic:** sample axis ratios \(a/b\), \(c/b\), set \(b\) so
  \(abc=V\), random axis permutation.
- **Monoclinic:** sample \(\beta\in[92^\circ,125^\circ]\), use \(V=abc\sin\beta\).
- **Triclinic:** sample angles, use full volume formula
  \(V=abc\sqrt{1-\cos^2\alpha-\cos^2\beta-\cos^2\gamma+2\cos\alpha\cos\beta\cos\gamma}\).

### 3. Artificial-molecule clusters + ring scaffolds (v0.11)

- Optional **database-guided scaffolds** (phenyl, pyridine, carboxyl, imidazole,
  short chain) then grow remaining non-H by covalent attachment.
- Element frequencies: general vs special-position tables.
- Clash rejection using min-image Cartesian distances (`min_contact_frac`).
- Optional H addition on C/N/O; multi-fragment packing.
- Isotropic \(U_{\mathrm{iso}}\) sampled in \([0.01,0.10]\) Å².
- Density prior: volume per non-H atom blended with \(\log V\).
- Optional near-inversion special-position seed + partial inversion images.
- **Void check:** random probes reject packs with large empty fractions.

### 3b. Lattice basis from volume

Given \(V\), sample axis ratios and crystal system, then solve for \(a,b,c\)
(and angles) so the cell volume matches the draw — orthorhombic
\(abc=V\), monoclinic \(abc\sin\beta\), triclinic full root formula.

### 4. Modes

| Mode | Description |
|------|-------------|
| `cluster` | Melgalvis-style molecules + volume lattice |
| `rejection` | Legacy random atoms with volume-informed \(V/\mathrm{atom}\) |
| `hybrid` | Mix (~70% cluster / 30% rejection) |

## Training integration

```bash
python scripts/train_strong_prior.py --scale --wilson-match --use-melgalvis-gen
python scripts/run_strong_prior_v9.py --quick --melgalvis-preset large
python scripts/run_strong_prior_v9.py --scale-xl --melgalvis-preset ha
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
