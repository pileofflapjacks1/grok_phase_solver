# For reviewers — one-pager

**Repository:** [grok_phase_solver](https://github.com/pileofflapjacks1/grok_phase_solver) · **Version:** 0.2.0 · **License:** MIT  

**Purpose of this page:** claims we make, claims we do *not* make, where the evidence lives, and how to reproduce the main results in under an hour.

---

## 1. What this work is

An **open, Python-first framework** for the crystallographic phase problem:

- Classical solvers (charge flipping, HIO, RAAR, DiffMap, direct methods, Patterson)
- Hybrid pipelines (ensemble free-FOM ranking, AI-PhaSeed, partial-φ seeding)
- Learned priors (GraphPhaseNet, hard-P1 PhaseMLP) with **honest hard-region metrics**
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
| C5 | Full ab initio graph prior (v3, Wilson-matched train) still delivers only ~**21%** of strong phases within 20° (below C4 bar) | [`strong_prior.md`](../data/processed/strong_prior.md) |
| C6 | Free FOM v2.1 uses positivity residual $R_+$ (not vacuous post-modulus $R$) and reduces false “solved” gates; hard failures are taxonomy **B+C** | [math/free_fom.md](math/free_fom.md), [math/failure_taxonomy.md](math/failure_taxonomy.md), [`failure_taxonomy.md`](../data/processed/failure_taxonomy.md) |
| C7 | Synthetic vs experimental \|F\| **Wilson domain gap** can be substantially reduced by slope/shell/quantile matching (e.g. hard gap ~9.5 → ~2.8 on COD Fobs reference) | [`wilson_domain_gap.md`](../data/processed/wilson_domain_gap.md) |
| C8 | With fair packing + PhAI weights, PhAI hybrids can solve COD **2016452** Fcalc @ 0.9 Å under strict criteria in our suite | [`cod_hybrid_benchmark.md`](../data/processed/cod_hybrid_benchmark.md) |

**Strict success definition:** mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5 **and** R1 ≤ 0.45 (`metrics/success.py`).

---

## 3. Non-claims (please do not attribute)

| # | We do **not** claim |
|---|---------------------|
| N1 | A general solution of the phase problem for arbitrary macromolecules |
| N2 | Pure ab initio superiority over industrial SHELXT/SHELXS on all small-molecule cases |
| N3 | That GraphPhaseNet / hard-P1 priors currently clear the hard cliff without partial information |
| N4 | That free FOM proves a correct structure (it is a **truth-free ranking** score) |
| N5 | Redistribution or equivalence of official SHELX or PhAI binaries/weights |

See also: [math/uniqueness_and_bounds.md](math/uniqueness_and_bounds.md).

---

## 4. Primary contribution (for a methods referee)

1. **Integrated open stack** — classical + hybrid + metrics + CLI in one reproducible package.  
2. **Honest hard-region science** — seed-quality bar (30% / 20°) separates “extension works” from “prior insufficient.”  
3. **Product path** — `auto` → ensemble (easy); `partial_phaseed` (hard + known φ); optional SHELXS±E → `trial.res` → SHELXL.  
4. **External calibration** — SHELXS H2H protocol (fixed-format HKL, Q-peaks → Fcalc phases for mapCC).  

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

# Optional: SHELXS H2H if you place academic binary at ShelX/shelxs
# xattr -dr com.apple.quarantine ShelX && chmod +x ShelX/shelxs
# python scripts/run_shelxs_h2h.py --quick
```

Precomputed report tables (no recompute needed for reading): `data/processed/*_h2h.md`, `partial_seed_benchmark.md`, `strong_prior.md`, `failure_taxonomy.md`.

---

## 6. Reading order

| Time | Document |
|------|----------|
| 5 min | This page + [README §1–4](../README.md) |
| 15 min | [USER_GUIDE](USER_GUIDE.md) decision tree |
| 30 min | [partial_seed.md](math/partial_seed.md) + [free_fom.md](math/free_fom.md) + [failure_taxonomy.md](math/failure_taxonomy.md) |
| 45 min | Scoreboards linked in §2 |
| Optional | [Cowtan notes](cowtan_phase_problem_notes.md), [arxiv_skeleton.md](arxiv_skeleton.md) |

---

## 7. Caveats on external comparisons

- **SHELXS scoring** uses Q-peaks → equal-atom Fcalc phases for mapCC, **not** full SHELXL-refined R1. Fair for *phasing* H2H; not a claim about refined structures.  
- **SHELX / PhAI** binaries and weights are **user-supplied**; CI and default installs run without them.  
- Synthetic organics are **pipeline-grade** (not full force-field chemistry); Wilson matching addresses amplitude statistics, not bonding realism.

---

## 8. Contact / provenance

- Code + scoreboards: GitHub `main`  
- Checklist: [TODO.md](../TODO.md) · History: [CHANGELOG.md](../CHANGELOG.md)  
- Design principle: physics fallback for every ML path; prefer correct math over marketing claims  

*Thank you for reviewing. The strongest positive result for hard cells is C4 (partial-φ); the strongest product result for easy cells is C1–C2 (ensemble).*
