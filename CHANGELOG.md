# Changelog

## 0.9.0 — 2026-07

### What’s new in v0.9.0

#### A. GraphPhaseNet v7 (GraPhAI multipath + Carrozzini bins)
- Node features **d_in=22**: v6 + hop2 local E, edge geometric E, Wilson residual,
  centro-HA cue
- κ×E multipath edge reweight; residual self-loops
- Training loss: optional **discretized phase CE** (4-bin / centro auto)
- `scripts/run_strong_prior_v7.py` (quick/pilot/scale/scale-xl; presets cod/hard/acta2026)
- Scoreboard: `data/processed/strong_prior_v7.*`
- Math: `docs/math/graph_phase_net_v7.md`, external GraPhAI note (no weight redistribution)
- **Honest:** laptop pilots remain near ~21–25% frac≤20°; **30% bar not cleared**;
  strict hard ab initio still 0% without partial-φ. Scale-xl left for cluster.

#### B. AI-PhaSeed / seed quality
- `discretize_phases(..., mode="quadrant")` alias; `phase_bin_agreement` helper
- Multi-seed combine: bin-agreement boost with continuous agreement
- Seed-quality features: bin entropy, mean |cos φ|, top-10% E; heuristic uses them

#### C. Synthetic data (Acta 2026-style packing)
- Multi-fragment packing; `actas2026_config` / preset
- Realistic monoclinic angle nudge option

#### D. Density modification
- Solvent fraction estimate uses volume / N_atom density prior

#### E. Packaging
- Version **0.9.0**, RELEASE notes, tests

#### Honest limits
- Partial-φ / fragment / HA remains the reliable hard path.
- Physics fallbacks retained for all ML components.
- No PhAI / SHELX / GraPhAI weight redistribution.

## 0.8.0 — 2026-07

### What’s new in v0.8.0

#### A. GraphPhaseNet v6 (GraPhAI HA-aware)
- Node features **d_in=18**: v5 + `ha_E_tail`, `low_res_w`, `E·low_res`, `κ_centrality`
- Stronger κ-gated message passing + residual self-loops
- `scripts/run_strong_prior_v6.py` with `--quick` / `--pilot` / `--scale` / `--scale-xl`
  and `--continue-from` resume
- Scoreboard: `data/processed/strong_prior_v6.{npz,json,md}`
- Math: `docs/math/graph_phase_net_v6.md`
- **Honest pilot (N=200, cod preset):** mean frac≤20° ≈**24.1%**, seedOK ≈**20%**,
  strong MPE ≈58° — **above** legacy ~22% plateau, **does not** clear 30% oracle bar;
  strict hard solves still 0% without partial-φ

#### B. Carrozzini-style seed quality classifier
- Feature list aligned with max_W / N_asym / Vol / seed_fraction / free-FOM proxies
- Trainable pure-NumPy logistic classifier (sklearn RF when importable)
- `scripts/train_seed_quality_rf.py` → `data/processed/seed_quality_rf.npz`
- Pilot test accuracy ~78% on synthetic oracle labels (not published 1505-COD RF)
- Heuristic Class 0/1 predictor remains default fallback

#### C. Partial-φ / hard path
- Full Fcalc soft prior for fragment / predicted-model seeds (absorbed from 0.7.1)
- Richer `recommend_seed_fraction` (N_asym, free-FOM, fragment cues)
- COD hard-path validation retained (`fragment_half` ≈/≥ `partial_30` mapCC)

#### D. Diffusion & trial research modes
- Extended reciprocal score features; SE(3) research helpers
  (`diffusion_se3_stub.reciprocal_invariant_features`)
- Optional research-only trial completion after peak pick
  (`pipeline/trial_complete.py`) — not auto atom typing
- Diffusion still **off** default `auto` path; physics Langevin fallback retained

#### E. Space-group / packaging
- Common SG aliases (P21/c, Pbca, P212121, …) via `normalize_space_group_name`
- Version **0.8.0**, RELEASE notes, docs, tests

#### Honest limits
- Hard ab initio seed bar still below 30% on current pilots; partial-φ remains
  the practical hard path. No general macromolecular ab initio claim.
- Physics fallbacks preserved for all ML components.
- No PhAI / SHELX redistribution.

## 0.7.0 — 2026-07

### What’s new in v0.7.0

#### A. Melgalvis curriculum (Acta Cryst. A 2026 alignment)
- COD-like volume preset (`cod_like_config`) and hard/large-Z preset
  (`hard_curriculum_config`)
- Heavy-atom injection, partial occupancy, large-molecule bias
- Low-resolution sample fraction for GraPhAI-like panels
- `iter_melgalvis_samples(..., preset=, include_low_res=)`

#### B. GraphPhaseNet / GraPhAI edges
- v5.1 κ-gated edges: power-law emphasis + residual self-loops
- Training: `--melgalvis-preset cod|hard`, `--low-res-frac`
- Pilot retrain scoreboard → `strong_prior_v5` / v0.7 notes

#### C. AI-PhaSeed / partial-φ
- `recommend_seed_fraction` (Carrozzini-inspired heuristic, soft-blended)
- Multi-seed `combine_phase_seeds` agreement boost
- Clearer ≥~30% strong-φ ≤20° bar hints in seed quality

#### D. Density modification / SG
- Protein-mode / auto solvent fraction estimation
- Existing MERGE / symmetry helpers retained

#### E. Diffusion
- Research-only `diffusion_se3_stub.py` (no default routing)

#### F. Packaging
- Version **0.7.0**, docs, tests, RELEASE notes

#### Honest limits
- Hard ab initio seed bar still below 30% on pilots; partial-φ remains the
  practical hard path. Physics fallbacks preserved for all ML components.

## 0.6.0 — 2026-07

### What’s new in v0.6.0

#### A. GraphPhaseNet v5 strong prior
- d_in=14 Melgalvis-inspired diffraction-graph features (shell rank, E·deg,
  local neighbor E, shell-normalized |F|) + κ-gated triplet edges
- `scripts/run_strong_prior_v5.py` with `--quick` / `--pilot` / `--scale` / `--scale-xl`
  (Melgalvis gen + Wilson match + hard oversample defaults)
- Scoreboard: `data/processed/strong_prior_v5.{npz,md,json}`
- Math: `docs/math/graph_phase_net_v5.md`
- **Honest:** hold-out frac≤20° still near the ~21–22% plateau on quick/medium
  runs; 30% oracle bar not cleared. Scale-xl (5k+) left for cluster runs.

#### B. Diffusion score path (trainable)
- `models/diffusion_score.py` — PhaseScoreNet (denoising score matching)
- Wired into Langevin reverse process when checkpoint present
- `scripts/train_diffusion_score.py`; weights `data/processed/diffusion_score.npz`
- Methods: `diffusion_hybrid_v2` / `diffusion_phaseed_v2` (aliases)
- Physics Langevin fallback always available

#### C. Scientist path / performance
- GUI multi-plane density slices + optional plotly volume HTML
- MERGE-class `merge_symmetry_equivalents`
- Ensemble `n_jobs` threaded multistart
- Dockerfile for containerized `gps-solve`
- Predicted-model seeding retained/strengthened from v0.5

#### Honest limits (unchanged in spirit)
- Hard ab initio seed bar ~21–22% ≤20° on current v5 pilots
- Partial-φ remains the practical hard-data path
- Diffusion / score nets experimental; no PXRDnet parity claim

## Earlier

See git history and `docs/RELEASE_NOTES_v0.*.md` for 0.2–0.5.
