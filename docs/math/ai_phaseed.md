# AI-PhaSeed

## Idea

Neural nets (e.g. **PhAI**) can produce useful phase estimates but often need
classical **phase extension** and density modification to fill weak reflections
and refine density. Unconditional charge flipping after PhAI can *destroy* a
good seed (observed on COD 2016452 at low resolution).

**AI-PhaSeed** (Carrozzini *et al.*, *J. Appl. Cryst.* 2025; building on
Larsen *et al.*, *Science* 2024) is:

1. AI phase vector \(\varphi^{\mathrm{AI}}(h)\)
2. Select a **seed set** of strong reflections (high \(|E|\))
3. Initialize other phases randomly (optional **discretization**: centro / bins)
4. Iterate density modification + modulus projection, **re-imposing** seed phases
5. Optional **DM+AI modified tangent** (AI phases as *a priori* info)
6. Optional free-FOM–gated polish
7. Optional **seed-quality Class 0/1** diagnostics

This targets **basin (B)** failure: put the iterate near the correct density
before free search.

## Carrozzini et al. (2025) — what we implement

Primary reference: Carrozzini *et al.* (2025). *The AI-based phase-seeding
(AI-PhaSeed) method: early applications and statistical analysis.*
*J. Appl. Cryst.* **58**(Pt 6), 1859–1869.
[DOI: 10.1107/S1600576725008271](https://doi.org/10.1107/S1600576725008271).

Key paper elements and this repo’s alignment:

| Paper | This repository |
|-------|-----------------|
| (a) Modified tangent weighting AI phases as a priori info with reliability factors (eqs. ~3–5) | `dm_ai_hybrid_refine` / `tangent_formula_iteration(..., ai_weight=…)` in `direct_methods.py`; exposed as `dm_ai_weight` in `ai_phaseed_solve` / `phase_extend` / CLI `--ai-dm-hybrid` |
| (b) Statistical Class 0/1 (k-means + RF on MPE_seed, CORR_seed + max W, N_asym, Vol, seed fraction); Class 1 >90% efficiency on their panel | `metrics/seed_quality.py`: truth-free **heuristic** Class + optional sklearn RF if a model is trained; **not** a bit-for-bit clone of the published RF on 1505 COD structures |
| (c) EDM + DM hybrid protocols useful at lower res / larger volumes (~1000–3500 Å³, P2₁/c focus) | `low_res_path` schedule: solvent flatten, longer seed anneal, more frequent DM+AI steps |
| (d) Validation on 1505 COD structures | Reproducible **subset** harness: `scripts/run_ai_phaseed_extended_benchmark.py` + existing COD scoreboards; honest about scale |

### Modified tangent (conceptual)

Classical κ-weighted tangent accumulates complex contributions from triplets.
The hybrid injects an AI term (schematic)::

\[
A_h \leftarrow \sum_{t\in\mathrm{triplets}} w_t\, e^{i(\ldots)}
\;+\; \lambda\, r_h\, |E_h|\, e^{i\varphi_h^{\mathrm{AI}}}
\]

with \(\lambda\) = `dm_ai_weight` (typical 0.3–0.7) and \(r_h\) higher on the
strong seed set. Seeds are **re-imposed** after tangent so the AI prior is not
washed out. Default \(\lambda=0\) preserves pre-0.4.0 behaviour.

### Seed-quality features

`predict_seed_quality(hkl, amplitudes, cell, seed_phases, …)` returns:

- `predicted_class` ∈ {0, 1}
- `success_probability` ∈ [0, 1]
- `predicted_mpe_deg`, `predicted_corr` (estimates; free-FOM proxies when truth-free)
- `features`: max_W, N_asym, Vol, seed_fraction, free_fom_composite, …

**Honest limit:** without ground-truth phases these are operational heuristics
for UX warnings, not the paper’s oracle MPE_seed / CORR_seed labels.

### Discretization / non-centro binning

`discretize_phases(mode="none"|"centro"|"bins", n_bins=4)`:

- **centro**: {0, π} for centrosymmetric seeds
- **bins**: equal bins on \((-\pi,\pi]\) — hook for non-centrosymmetric
  phase quantization (paper-style coarse seeds)

Space-group expansion beyond centro heuristics remains a roadmap item
(gemmi SG expansion).

## Algorithm (this repository)

```text
φ_AI ← PhAI_fair(|F|) or external or partial seed
S ← top fraction of reflections by |E|
seed_quality ← predict_seed_quality(…)   # Class 0/1 diagnostics
φ[S] ← φ_AI[S]   (optional discretize: centro / bins)
φ[∖S] ← random, softly blended toward φ_AI
optional: DM+AI modified tangent once

for c = 1 … n_extend:
    ρ ← IFFT(|F| e^{iφ})
    ρ ← positivity (and optional solvent flatten; low_res_path ↑ solvent)
    φ ← arg(FFT(ρ));  reimpose |F|
    φ ← blend(φ_AI, φ; prior_weight)     # soft full prior
    φ[S] ← reimpose(φ_AI[S]; w_seed)     # hard/soft seed (annealed)
    every k cycles: DM+AI hybrid (λ = dm_ai_weight)
    reimpose seeds

optional: conditional CF/RAAR polish if free FOM v2.1 accepts
multistart over random ∖S; pick max free-FOM composite
```

Implementation: `solvers/ai_phaseed.py`, `solvers/direct_methods.py`,
`metrics/seed_quality.py`.

| Function | Role |
|----------|------|
| `select_seed_indices` | strong \(|E|\) or \(|F|\) subset |
| `phase_extend` | extension loop + optional DM+AI / low-res path |
| `ai_phaseed_solve` | full pipeline from external AI phases |
| `phai_phaseed_solve` | PhAI fair → AI-PhaSeed |
| `predict_seed_quality` | Class 0/1-style diagnostics |
| `dm_ai_hybrid_refine` | modified tangent with AI prior |

## CLI / GUI

```bash
gps-solve --hkl data.hkl --ins data.ins --method phai_phaseed \
  --ai-dm-hybrid --low-res-path --seed-quality-filter
# or: --dm-ai-weight 0.5 --prior-weight 0.30 --seed-fraction 0.25
```

GUI Advanced options: “DM+AI hybrid tangent”, “Low-res EDM path”,
“Warn on Class 0 seed quality”. Results panel shows predicted class / P(success).

## Relation to free FOM and taxonomy

- **Free FOM v2.1** ranks multistart extension trials and gates final polish.
- **Rewrite trust-region** blocks CF that rewrites phases without large \(R_+\) gain.
- **Seed Class 0** often correlates with taxonomy **B** (never enter basin) even
  when information metrics look OK — see `failure_taxonomy.py` note.
- Expected taxonomy shift: **B+C → near/solved** when AI seed is in-domain
  (e.g. COD \(P2_1/c\)); less help on out-of-domain synthetic P1 (see
  `phai_taxonomy.md`).

## Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `seed_fraction` | 0.25 (API) / 0.30 (CLI) | strong seed set size |
| `seed_weight` → `seed_weight_final` | 1.0 → 0.75 | anneal |
| `prior_weight` | 0.30 | soft full AI prior |
| `dm_ai_weight` | **0.0** | off for backward compatibility; 0.5 via `--ai-dm-hybrid` |
| `low_res_path` | False (auto if d_min≥1.35 and Vol≥1000) | EDM schedule |
| `n_extend` | 15 | positivity ER-like cycles |
| `polish` | charge_flipping | free-FOM gated |
| `n_starts` | 2 | free-FOM pick |

## P2₁/c strengths and generalization

PhAI and the Carrozzini panel emphasize small-molecule / \(P2_1/c\)-like
organics. This repo’s auto-routing prefers `phai_phaseed` when SG looks
P2₁/c-like and weights exist. **Future generalization:** full gemmi SG
expansion, phase binning for non-centro groups, larger-volume Melgalvis
curriculum — not claimed solved here.

## Empirical notes (this repo)

- **Oracle seed** (true \(\varphi\) as AI): mapCC \(\approx 1\) after extension — algorithm is sound.
- **Partial seed** (\(\sim\)55% true + noise): **solves** hard synthetic cells where CF fails (mapCC \(\sim 0.8\)–0.9). Shows AI-PhaSeed works when the prior is moderately good.
- **PhAI on random P1 synthetic**: still weak (domain gap; PhAI trained for COD-like \(P2_1/c\)).
- **Hard-P1 domain prior** (`models/hard_p1_prior.py`): origin-invariant PhaseMLP on hard P1; hold-out prior mapCC ~0.5 (vs ~0.3 random/PhAI-on-P1). Still rarely strict-solves — small MLP is a weak prior; algorithm is validated by oracle/partial seeds.
- **COD 2016452**: `phai_phaseed` improves over CF at low res; at 0.9 Å free-FOM–gated CF after PhAI (`phai_cf_cond`) remains the strict solver — extension alone does not replace a *helpful* polish when free FOM accepts it.
- **Seed bar:** GraphPhaseNet / Melgalvis XL still ~21–22% frac≤20° on strong |E| (below ~30% oracle bar). Hybrid tooling improves **use of** seeds; it does not by itself clear the ab initio seed bar.

## References

1. Larsen, A. S., Rekis, T. & Madsen, A. Ø. (2024). *Science* **385**, 522–528 (PhAI).
2. Carrozzini *et al.* (2025). *J. Appl. Cryst.* **58**, 1859–1869.
   DOI: [10.1107/S1600576725008271](https://doi.org/10.1107/S1600576725008271).
3. Melgalvis & Rekis (2026). Synthetic structure generator (integrated as `synthetic_melgalvis`).
4. Cowtan, K. Density modification / solvent flattening.
5. Project: `docs/math/free_fom.md`, `docs/math/failure_taxonomy.md`,
   `docs/math/partial_seed.md`, `docs/math/phase_problem_overview.md`.
