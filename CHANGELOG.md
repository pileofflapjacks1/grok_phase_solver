# Changelog

## 0.13.4 — 2026-08

Patch so **`--retry-with-peaks`** rides PyPI (was on `main` only after 0.13.3).  
Published: https://pypi.org/project/grok-phase-solver/0.13.4/

### One-command hard retry
- `gps-solve --retry-with-peaks`: if the first pass looks weak, re-run
  `partial_phaseed` using this run's `peaks.csv` (writes `--out/retry_peaks/`)
- Same gate as the GUI "Retry with peaks as seed" button
- Peaks-as-carbon is a cheap fallback, not a substitute for a real fragment/HA
- Science claims unchanged (still 0.13.1 freeze)

## 0.13.3 — 2026-08

Patch so the **CCP4 map + PyMOL/Coot handoff** rides PyPI (was on `main` only after 0.13.2).  
Published: https://pypi.org/project/grok-phase-solver/0.13.3/

### PyMOL / Coot map handoff
- `gps-solve` writes **`density.map`** (CCP4 MODE-2, unit cell) plus
  `open_in_pymol.pml`, `open_in_coot.sh`, and `peaks.pdb`
- Does not replace Olex2/SHELXL; visualization only
- Science claims unchanged (still 0.13.1 freeze)

## 0.13.2 — 2026-08

Patch so the **Vol-band next-action chooser** rides PyPI (was on `main` only after 0.13.1).  
Published: https://pypi.org/project/grok-phase-solver/0.13.2/

### Partial-φ next-action product
- `report.md` / `solve_summary.json` / GUI banner: Vol-band chooser
  (`pipeline/next_action.py`) names one seed source after `auto`
- Mid-band → fragment / predicted model; large cell → bigger fragment or HA;
  undersized seed → enlarge; healthy free-FOM → SHELXL inspect
- Evidence cited is the local COD Vol-band panel (C25), not a 1505-COD set

### Also on this tag
- Paper / claim freeze to v0.13.1 scoreboards (Figs. 1–6)
- Science claims unchanged; no GraphPhaseNet bump

## 0.13.1 — 2026-08

Patch release so the **COD Vol-band experimental panel** rides the published tag (was on `main` only after v0.13.0).

### COD Vol-band experimental panel
- Expanded `run_cod_stratified_bench.py`: Fobs + Fcalc; auto / partial_15 / partial_30 / fragment_half
- Added mid-band COD **2012000**, **2013000** (+ small **2200000**); 6 datasets, 48 runs
- Scoreboard: `data/processed/cod_stratified_bench.{md,json}`; math note `docs/math/cod_vol_band_panel.md`
- Mid-band (Vol 1000–3500): fragment_half mean mapCC **~0.71** vs auto **~0.27**
- Packaging: version **0.13.1**, PyPI + GitHub Release

## 0.13.0 — 2026-08

### What’s new in v0.13.0

#### A. GraphPhaseNet v11 (d_in=34) + Melgalvis packing
- Features: ⟨E⁴⟩ moment, shell skew, deg×E, centro intensity cue
- Stronger multipath edges; bin CE 0.26; presets include `xdxd` low-res curriculum
- Intermolecular contact packing + solvent-void option in Melgalvis
- `scripts/run_strong_prior_v11.py` + scoreboard `strong_prior_v11.*`
- **Honest:** 30% seed bar claimed only if scoreboard YES

#### B. CrystalX-inspired peak → atom typing
- `pipeline/crystalx_typing.py`: height/geometry element assignment + H placement
- `trial.res` + `typed_atoms.csv` from export; pure-physics untyped fallback

#### C. XDXD-inspired generative coordinates (research)
- `xdxd_propose_coordinates` multi-start CF→atoms→Fcalc
- CLI `--method xdxd_structure` (**not** auto)

#### D. GraPhAI external H2H skeleton
- `models/graphai_external.py` + `third_party/graphai/README.md` (user Zenodo only)

#### E. Auto routing
- Very low-res / sparse data path with explicit partial-φ recommendation

#### F. Packaging
- Version **0.13.0**, tests, docs/math notes, FOR_REVIEWERS C22–C24

## 0.12.0 — 2026-08

### What’s new in v0.12.0

#### A. GraphPhaseNet v10 (d_in=30)
- Features: hop3 multipath, multipath span, Wilson B proxy, E-outlier ratio
- Stronger κ-gated multipath edges (feature_version=9); bin CE weight 0.24
- `scripts/run_strong_prior_v10.py` with large/HA curriculum + P−1 hold-out mix
- Scoreboard: `data/processed/strong_prior_v10.*`
- Math: `docs/math/graph_phase_net_v10.md`
- **Honest:** pilots still below the 30% strong-seed bar unless scoreboard says YES

#### B. Melgalvis experimental realism
- Optional B-factor inflation (radiation-damage-ish) + amplitude noise hooks
- Large-cell / HA / ring packing retained from v0.11

#### C. Generative structure proposal (research)
- CF-peak → Fcalc phase seed API (`models/generative_structure.py`)
- CLI/pipeline method `generative_structure` (**not** used by `auto`)
- Physics Langevin / CF polish fallback; math note `docs/math/generative_structure.md`

#### D. Packaging
- Version **0.12.0**, tests, RELEASE notes
- Strict success, physics fallbacks, no PhAI/SHELX/GraPhAI redistribution

## 0.11.0 — 2026-07

### What’s new in v0.11.0

#### A. Melgalvis synthetic generator (Acta Cryst. A 2026 style)
- Ring / functional-group scaffolds (`build_ring_scaffold`), multi-fragment packing
- Void + short-contact packing quality checks
- New **`large_cell_config`** (Vol ~1000–3500 Å³) + stronger `ha` / `acta2026` / `hard`
- Presets wired: `cod | hard | acta2026 | ha | large` in training + `iter_melgalvis_samples`
- Docs: `docs/math/synthetic_melgalvis.md`

#### B. GraphPhaseNet v9 (d_in=26)
- Features: log Vol, shell E std, κ×E, low-res·rank·HA + stronger multipath edges
- `scripts/run_strong_prior_v9.py` with HA/Z **and Vol-band** stratified hold-out
- Scoreboard: `data/processed/strong_prior_v9.*`
- Math: `docs/math/graph_phase_net_v9.md`
- **Honest:** quick pilots expected near ~20–25% frac≤20°; **30% bar not claimed**

#### C. AI-PhaSeed / seeding hardening
- `recommend_seed_fraction` v11: Vol 1000–3500 → ~28–30% seed + practical-bar note
- `filter_seed_by_bin_quality`: multi-bin entropy, optional `|E|` floor
- Multi-seed `combine_phase_seeds`: continuous + bin agreement boost
- Class 0/1 `format_seed_class_diagnostics` for report.md / GUI

#### D. Packaging
- Version **0.11.0**, tests (`test_v011_melgalvis_graph_seed.py`), RELEASE notes
- Physics fallbacks, strict success definition, no PhAI/SHELX/GraPhAI redistribution

## 0.10.0 — 2026-07

### What’s new in v0.10.0

#### A. GraphPhaseNet v8 curriculum (same d_in=22 as v7)
- HA-heavy Melgalvis preset (`--melgalvis-preset ha`) for GraPhAI Z≥19 regime
- Stratified hold-out reporting (HA / max Z / organic): `metrics/stratified_prior.py`
- `scripts/run_strong_prior_v8.py` with scale-xl path; Wilson match default
- Docs: `docs/math/graph_phase_net_v8.md`
- **Honest:** laptop pilots remain near ~21–25% frac≤20°; 30% bar not cleared

#### B. Hybrid Difference Map (research)
- `hybrid_difference_map_solve`: DiffMap in protein + HIO in solvent
- CLI/pipeline method `hdm` (not used by `auto`)
- Math note: `docs/math/hybrid_difference_map.md`

#### C. AI-PhaSeed hardening
- `filter_seed_by_bin_quality` (entropy-based seed thinning)
- `select_seed_indices(..., e_min=)` strong-|E| floor
- `recommend_seed_fraction` Vol-band preference for ≥25% seed

#### D. Synthetic + COD stratified bench
- `ha_heavy_config` curriculum
- `scripts/run_cod_stratified_bench.py` (local COD Vol / Z skeleton)

#### E. Packaging
- Version **0.10.0**, RELEASE notes, tests

#### Honest limits
- Partial-φ / fragment / HA remains the reliable hard path.
- HDM and scale-xl cluster runs are optional; physics fallbacks retained.
- No PhAI / SHELX / GraPhAI weight redistribution.

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
