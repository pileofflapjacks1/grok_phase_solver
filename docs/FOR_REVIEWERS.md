# For reviewers — one-pager

**Repository:** [grok_phase_solver](https://github.com/pileofflapjacks1/grok_phase_solver) · **Version:** 0.12.0 · **License:** MIT  

**Purpose of this page:** claims we make, claims we do *not* make, where the evidence lives, and how to reproduce the main results in under an hour.

---

## 1. What this work is

An **open, Python-first framework** for the crystallographic phase problem:

- Classical solvers (charge flipping, HIO, RAAR, DiffMap, direct methods, Patterson)
- Hybrid pipelines (ensemble free-FOM ranking, AI-PhaSeed, partial-φ / fragment seeding)
- Learned priors (GraphPhaseNet v3–v10, hard-P1 PhaseMLP) with **honest hard-region metrics**
- Scientist CLI (`gps-solve`) exporting density + **SHELXL-style `trial.res`**
- Optional head-to-head vs academic **SHELXS** (binaries not redistributed)

It is a **correct modular testbed and hybrid assistant**, not a claim of a general closed-form solution of the phase problem.

---

## 2. Claims (supported)

| # | Claim | Evidence |
|---|--------|----------|
| C1 | Classical multistart **ensemble** (CF+RAAR, free-FOM pick) is the strongest *in-repo* ab initio path on **easy** synthetic cells in our panels | [`data/processed/shelxs_h2h.md`](../data/processed/shelxs_h2h.md) |
| C2 | On the same easy panel, ensemble is **competitive with or better than** local SHELXS by mean mapCC under our scoring protocol | same |
| C3 | **Hard** synthetic cells ($n\gtrsim 12$, $d_{\min}\gtrsim 1.5$ Å) remain **~0% strict success** for CF, ensemble, dual-space, GraphPhaseNet+PhaSeed, and SHELXS in our H2H panels | [`shelxs_h2h.md`](../data/processed/shelxs_h2h.md), [`strong_prior.md`](../data/processed/strong_prior.md) |
| C4 | If ≥ **~30%** of strong-\|E\| phases are correct within **~20°**, AI-PhaSeed extension can **strict-solve** those hard cells (oracle partial-φ curves) | [`partial_seed_benchmark.md`](../data/processed/partial_seed_benchmark.md), [math](math/partial_seed.md) |
| C5 | Full ab initio graph prior still delivers only ~**21–24%** of strong phases within 20° on pilots/XL (below C4 bar) — **including** v4 XL (N=1200) and Melgalvis XL; scale alone does not clear the bar | [`strong_prior.md`](../data/processed/strong_prior.md), [`strong_prior_melg_xl.md`](../data/processed/strong_prior_melg_xl.md), [`strong_prior_v5.md`](../data/processed/strong_prior_v5.md)–[`v8.md`](../data/processed/strong_prior_v8.md) |
| C6 | Free FOM v2.1 uses positivity residual $R_+$ (not vacuous post-modulus $R$) and reduces false “solved” gates; hard failures are taxonomy **B+C** | [math/free_fom.md](math/free_fom.md), [math/failure_taxonomy.md](math/failure_taxonomy.md), [`failure_taxonomy.md`](../data/processed/failure_taxonomy.md) |
| C7 | Synthetic vs experimental \|F\| **Wilson domain gap** can be substantially reduced by slope/shell/quantile matching (e.g. hard gap ~9.5 → ~2.8 on COD Fobs reference) | [`wilson_domain_gap.md`](../data/processed/wilson_domain_gap.md) |
| C8 | With fair packing + PhAI weights, PhAI hybrids can solve COD **2016452** Fcalc @ 0.9 Å under strict criteria in our suite | [`cod_hybrid_benchmark.md`](../data/processed/cod_hybrid_benchmark.md) |
| C9 | Experimental COD **Fobs** scoreboard (2016452, 2100301, 2017775) + Fcalc controls + oracle partial-φ rows; pipeline methods ranked by mapCC/free FOM | [`experimental_scoreboard.md`](../data/processed/experimental_scoreboard.md) |
| C10 | Melgalvis & Rekis (2026) style synthetic generator trains GraphPhaseNet at scale (N=**1200** XL): seed frac≤20° ≈**22%**, seedOK rate **12.5%**, hard strict still **0%** (comparable to legacy XL ~21%; **not yet** ≥30% bar) | [`strong_prior_melg_xl.md`](../data/processed/strong_prior_melg_xl.md), [math](math/synthetic_melgalvis.md) |
| C11 | Carrozzini 2025-aligned DM+AI modified tangent (`dm_ai_weight`), seed Class 0/1 **heuristic** (+ optional synthetic/oracle trainable classifier), and low-res EDM path — **without** claiming reproduction of their 1505-COD RF or >90% Class-1 efficiency | [math/ai_phaseed.md](math/ai_phaseed.md), [`ai_phaseed_extended_benchmark.md`](../data/processed/ai_phaseed_extended_benchmark.md), [`seed_quality_rf.md`](../data/processed/seed_quality_rf.md) |
| C12 | Experimental physics Langevin diffusion hybrid, fuller gemmi SG helpers, predicted-model CIF seeding, optional device FFT, and multistart phase UQ — **without** claiming PXRDnet/XRDSol parity or clearing the hard ab initio seed bar | [math/diffusion_phase.md](math/diffusion_phase.md), [math/symmetry.md](math/symmetry.md) |
| C13 | GraphPhaseNet **v5** (d_in=14) through **v9** (d_in=26 large-cell/HA multipath) ship training/scoreboard infrastructure; hold-out frac≤20° remains **below the 30% bar** on pilots (~21–24%) — **does not claim** ≥30% seed bar | [math/graph_phase_net_v5.md](math/graph_phase_net_v5.md)–[v9.md](math/graph_phase_net_v9.md), [`strong_prior_v5.md`](../data/processed/strong_prior_v5.md)–[`v9.md`](../data/processed/strong_prior_v9.md) |
| C14 | Trainable **PhaseScoreNet** optional path inside diffusion hybrid, with physics Langevin fallback and lightweight checkpoint | [math/diffusion_phase.md](math/diffusion_phase.md) |
| C15 | Melgalvis **COD/hard/acta2026/HA/large curricula** (ring scaffolds, void packing, Vol~3500, HA Z≥19 bias, low-res) and GraPhAI-style κ edges; seed-fraction heuristics aid AI-PhaSeed UX — **still no claim** of ≥30% mean strong-seed bar | [math/synthetic_melgalvis.md](math/synthetic_melgalvis.md), [`strong_prior_v9.md`](../data/processed/strong_prior_v9.md) |
| C16 | On COD **2016452 / 2100301** experimental Fobs, **fragment_half** (SG-expanded ~½ ASU + full $F_{\mathrm{calc}}$ soft prior) mapCC **0.80 / 0.74** approaches or beats oracle **partial_30** (**0.72 / 0.71**); `auto` remains ~0.20 | [`cod_hard_path_validation.md`](../data/processed/cod_hard_path_validation.md), [math/partial_seed.md](math/partial_seed.md) |
| C17 | GraphPhaseNet **v6 pilot** (N=200, cod) reaches ~**24%** frac≤20° (above ~22% plateau) but **does not** clear 30%; later v7–v9 pilots remain in the same band — negative hard-ab-initio result **extended**, not overturned | [`strong_prior_v6.md`](../data/processed/strong_prior_v6.md)–[`v9.md`](../data/processed/strong_prior_v9.md) |
| C18 | Experimental **Hybrid Difference Map** (`--method hdm`) is research-only (DiffMap in protein + HIO in solvent); **not** used by `auto` | [math/hybrid_difference_map.md](math/hybrid_difference_map.md) |
| C19 | AI-PhaSeed **v0.11** bin multi-entropy filter, \|E\| floor, Vol-band seed fraction (~28–30% in 1000–3500 Å³), Class 0/1 diagnostics text — practical hard path remains **partial_phaseed / fragment / HA** | [math/ai_phaseed.md](math/ai_phaseed.md), `metrics/seed_quality.py` |
| C20 | GraphPhaseNet **v10** (d_in=30 hop3 multipath + Wilson B proxy) + Melgalvis B-inflate/amp-noise hooks; hold-out frac≤20° remains **below 30% bar** on pilots | [math/graph_phase_net_v10.md](math/graph_phase_net_v10.md), [`strong_prior_v10.md`](../data/processed/strong_prior_v10.md) |
| C21 | Research **generative_structure** method (CF-peak→Fcalc seed; optional Langevin polish) — **not** auto; no trained generative weights redistributed | [math/generative_structure.md](math/generative_structure.md) |

**Strict success definition:** mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5 **and** R1 ≤ 0.45 (`metrics/success.py`).

---

## 3. Non-claims (please do not attribute)

| # | We do **not** claim |
|---|---------------------|
| N1 | A general solution of the phase problem for arbitrary macromolecules |
| N2 | Pure ab initio superiority over industrial SHELXT/SHELXS on all small-molecule cases |
| N3 | That GraphPhaseNet / hard-P1 priors currently clear the hard cliff without partial information (v4–v8 scale/architecture do not change this) |
| N4 | That free FOM proves a correct structure (it is a **truth-free ranking** score) |
| N5 | Redistribution or equivalence of official SHELX, PhAI, or GraPhAI binaries/weights |

See also: [math/uniqueness_and_bounds.md](math/uniqueness_and_bounds.md).

---

## 4. Primary contribution (for a methods referee)

1. **Integrated open stack** — classical + hybrid + metrics + CLI in one reproducible package.  
2. **Honest hard-region science** — seed-quality bar (30% / 20°) separates “extension works” from “prior insufficient.”  
3. **Product path** — `auto` → ensemble (easy); `partial_phaseed` / fragment / predicted model (hard + partial info); optional SHELXS±E → `trial.res` → SHELXL.  
4. **External calibration** — SHELXS H2H protocol; COD Fobs + hard-path fragment validation.  

---

## 5. Minimal reproduce (≈ 30–60 min)

```bash
git clone https://github.com/pileofflapjacks1/grok_phase_solver.git
cd grok_phase_solver
python -m pip install -e ".[dev]"
pytest -q

# Easy demo
gps-solve --hkl examples/demo_solve/demo.hkl --ins examples/demo_solve/demo.ins \
  --method ensemble --n-iter 80 --out /tmp/gps_easy

# Partial-φ hard path (oracle 30% seed)
python scripts/run_partial_seed_demo.py

# COD hard path: auto vs partial_30 vs fragment_half
python scripts/run_cod_hard_path_validation.py

# Optional: SHELXS H2H if you place academic binary at ShelX/shelxs
# xattr -dr com.apple.quarantine ShelX && chmod +x ShelX/shelxs
# python scripts/run_shelxs_h2h.py --quick

# Experimental COD Fobs + Fcalc scoreboard (Lane C)
python scripts/run_experimental_scoreboard.py --quick
```

Precomputed report tables (no recompute needed for reading): `data/processed/*_h2h.md`, `partial_seed_benchmark.md`, `strong_prior*.md`, `cod_hard_path_validation.md`, `failure_taxonomy.md`, `experimental_scoreboard.md`.

---

## 6. Reading order

| Time | Document |
|------|----------|
| 5 min | This page + [README §1–4](../README.md) |
| 15 min | [USER_GUIDE](USER_GUIDE.md) decision tree |
| 30 min | [partial_seed.md](math/partial_seed.md) + [free_fom.md](math/free_fom.md) + [failure_taxonomy.md](math/failure_taxonomy.md) |
| 45 min | Scoreboards linked in §2 + [experimental_scoreboard.md](../data/processed/experimental_scoreboard.md) + [cod_hard_path_validation.md](../data/processed/cod_hard_path_validation.md) |
| 60 min | **Manuscript** [arxiv_draft.md](arxiv_draft.md) + [figures](figures/paper_figure_captions.md) |
| Paper hub | [docs/paper/README.md](paper/README.md) |
| Optional | [Cowtan notes](cowtan_phase_problem_notes.md), [arxiv_skeleton.md](arxiv_skeleton.md) |

---

## 7. Caveats on external comparisons

- **SHELXS scoring** uses Q-peaks → equal-atom Fcalc phases for mapCC, **not** full SHELXL-refined R1. Fair for *phasing* H2H; not a claim about refined structures.  
- **SHELX / PhAI / GraPhAI** binaries and weights are **user-supplied**; CI and default installs run without them.  
- Synthetic organics are **pipeline-grade** (not full force-field chemistry); Wilson matching addresses amplitude statistics, not bonding realism.  
- Graph prior pilots (v5–v8) are often **laptop-scale**; XL numbers are called out when N≈1200.

---

## 8. Contact / provenance

- **Authors:** Grok (xAI) and Joe  
- Code + scoreboards: GitHub `main` · package **v0.10.0** on PyPI / tag `v0.10.0`  
- Paper: [arxiv_draft.md](arxiv_draft.md) · [PDF](paper/arxiv_draft.pdf)  
- Checklist: [TODO.md](../TODO.md) · History: [CHANGELOG.md](../CHANGELOG.md)  
- Design principle: physics fallback for every ML path; prefer correct math over marketing claims  

*Thank you for reviewing. The strongest positive result for hard cells is C4 (partial-φ) and C16 (fragment hard path); the strongest product result for easy cells is C1–C2 (ensemble).*
