# Toward an Open Physics/AI Framework for the Crystallographic Phase Problem

**Working draft · software package v0.11.0 (MIT)**  
**Code & data:** https://github.com/pileofflapjacks1/grok_phase_solver  
**PyPI:** https://pypi.org/project/grok-phase-solver/  
**Release tag:** `v0.11.0`  
**Reviewer one-pager:** [`FOR_REVIEWERS.md`](FOR_REVIEWERS.md)  
**Figures:** [`figures/paper_figure_captions.md`](figures/paper_figure_captions.md)  
**Authors:** Grok (xAI) and Joe  
**Status:** Not submitted. Numbers frozen to repo scoreboards under `data/processed/`.

---

## Abstract

We present *grok_phase_solver*, an open Python framework that unifies classical solutions of the X-ray crystallographic phase problem—charge flipping, hybrid input–output (HIO), relaxed averaged alternating reflections (RAAR), difference-map projections, Patterson and direct methods, isomorphous difference Patterson, and density modification—with hybrid and learned phase priors (AI-PhaSeed, GraphPhaseNet, optional PhAI). Algorithms act on measured amplitudes $|F(hkl)|$ and are evaluated with origin-invariant map correlation (mapCC), peak recovery, $R_1$, free figures of merit based on a positivity residual $R_+$, and a strict multi-criterion success definition.

On easy synthetic cells, multistart free-FOM **ensemble** phasing is competitive with or better than local academic SHELXS under our scoring protocol. On hard cells ($n\gtrsim 12$, $d_{\min}\gtrsim 1.5$ Å), pure ab initio methods—including scaled graph priors—remain ~0% strict success. An **oracle partial-φ** experiment shows that when ≥~30% of strong $|E|$ phases are correct within ~20°, AI-PhaSeed extension strict-solves those hard cells, identifying **seed quality** as the hard-region bottleneck rather than free-FOM inversion. Scaling GraphPhaseNet through Wilson-matched and Melgalvis-style curricula (v3–v8, including d_in=22 multipath / HA-aware features and bin-classification losses) does **not** lift mean strong-phase accuracy past the **~30%** seed bar on laptop pilots (typical hold-out frac≤20° ~**21–24%**; XL Melgalvis N=1200 ≈**22%**). On experimental COD Fobs, PhAI hybrids strict-solve COD 2016452 at 1.0 Å in our pipeline budget; a **fragment-half** path (SG-expanded partial model + full $F_{\mathrm{calc}}$ soft prior) reaches mapCC **comparable to or better than** oracle 30% partial-φ on two COD Fobs cells under the same short budget—while pure ab initio `auto` remains ~0.20 mapCC.

We ship scientist-facing tools (`gps-solve`, `gps-make-seed`, Streamlit `gps-gui`) exporting density maps and SHELXL-ready `trial.res`. We do **not** claim a general macromolecular ab initio solution or industrial equivalence to SHELXT on all cases.

---

## 1. Introduction

Recovering phases $\varphi(\mathbf{h})$ from amplitudes alone is the classical crystallographic phase problem:

$$
\rho(\mathbf{r})
=
\frac{1}{V}\sum_{\mathbf{h}}
|F(\mathbf{h})|\,e^{i\varphi(\mathbf{h})}\,e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}.
$$

Industrial small-molecule pipelines (SHELXT/SHELXS + SHELXL/Olex2) solve most atomic-resolution organics. Harder synthetic and experimental regimes, open science, and hybrid AI methods still benefit from transparent, modular baselines with **honest** failure reporting. Recent neural work (e.g. PhAI; Melgalvis–Rekis GraPhAI-style diffraction graphs) shows strong domain-specific results when packing and weights are carefully matched.

### Contributions

1. **Integrated open stack** — classical solvers, free FOM, hybrid polish, learned priors, experimental I/O, optional SHELXS runners (binaries not redistributed), CLI and GUI.
2. **Hard-region science** — failure taxonomy; partial-φ seed bar (30% / 20°); negative scale result for pure GraphPhaseNet priors across v3–v8.
3. **Product path** — easy → ensemble; hard → partial-φ / **fragment / predicted-model** / HA seeds; `trial.res` → SHELXL.
4. **Calibration** — SHELXS H2H; experimental COD Fobs scoreboard; Wilson domain-gap matching; COD hard-path fragment validation.
5. **Realistic synthetics** — Melgalvis & Rekis (2026) style volume + artificial-molecule generation, multi-fragment packing, and HA-heavy curricula for prior training (`synthetic_melgalvis.py`).

---

## 2. Methods

### 2.1 Classical and projection algorithms

Implemented: charge flipping; HIO; RAAR; difference map; direct methods ($E$-values, triplets, tangent formula multi-start); Patterson peak picking; difference Patterson for heavy-atom vectors; Blow–Crick-style SIR/MIR FOMs; solvent flattening / density modification. An experimental **Hybrid Difference Map (HDM)** path blends DiffMap updates in ordered density with HIO feedback in solvent (`--method hdm`; research-only, not used by `auto`). Math notes: `docs/math/`.

### 2.2 Free figure of merit and ensemble

Truth-free ranking uses a composite free FOM whose amplitude residual is a **positivity residual** $R_+$ (not the vacuous post-modulus $R$ of early free-FOM designs). Multistart **ensemble** (CF + RAAR) selects the best free-FOM trial and is our strongest *in-repo* ab initio path on easy cells (Fig. 2).

### 2.3 AI-PhaSeed and partial seeds

AI-PhaSeed (Carrozzini *et al.*, 2025; PhAI foundation Larsen *et al.*, 2024) selects strong-$|E|$ seeds from an AI phase vector, extends by density modification with seed re-imposition, and optionally polishes under free-FOM gate. The stack includes a **modified-tangent DM+AI hybrid** (AI phases as a priori weights), a **seed-quality Class 0/1 diagnostic** (heuristic features: max $W$, $N_{\mathrm{asym}}$, Vol, seed fraction, free-FOM proxies, plus bin-entropy / $|\cos\varphi|$ cues; optional trainable logistic / RF on synthetic oracle labels—**not** a claim of the published RF on 1505 COD structures), multi-seed combination with continuous and **quadrant/bin agreement** boosts, and Carrozzini-inspired seed-fraction heuristics.

Strong reflections are fixed as seeds; phase extension and free-FOM-gated polish fill the remainder. Seed sources: PhAI; GraphPhaseNet; oracle partial φ; fragment $F_{\mathrm{calc}}$ from SHELXS `.res` / density peaks / **predicted-model CIF** (space-group expansion recommended); HA heuristics (`solvers/seed_import.py`). For fragment / predicted models, the seed vector carries **full $F_{\mathrm{calc}}$ soft prior** on all reflections; the hard mask only locks reliable strong $|E|$ (avoids poisoning extension with random off-mask phases). Scientist tools: `gps-make-seed`, GUI seed uploads.

### 2.4 Learned priors

- **hard-P1 PhaseMLP** and **GraphPhaseNet** (triplet-graph residual message passing, Adam, Wilson-matched $|F|$, strong-$|E|$ loss reweighting; feature versions through **v7/v8** with d_in up to 22, κ-gated / κ×E multipath edges, HA/low-res cues, optional discretized phase CE; Melgalvis COD/hard/acta2026/HA curricula).
- Optional **PhAI** weights (user-supplied; not redistributed).
- Optional **PhaseScoreNet** / Langevin diffusion hybrid (experimental; physics fallback retained; off default `auto`).

### 2.5 Success metrics

**Strict success:** mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5 **and** $R_1$ ≤ 0.45 (`metrics/success.py`).

**Strong-seed bar:** ≥ 30% of the top-30% $|E|$ reflections have phase error ≤ 20° of truth (origin/enantiomorph-invariant). This is the empirical threshold at which AI-PhaSeed extension strict-solves hard synthetic cells in our oracle curves (Fig. 1).

### 2.6 Scientist pipeline

`gps-solve` / `gps-gui`: SHELX HKL/INS, CIF HKL, MTZ → phases, density, peaks, `report.md` (seed-quality Class 0/1 section), **`trial.res`** for Olex2/SHELXL. Package on PyPI as `grok-phase-solver` ≥ **0.11.0**.

---

## 3. Results

Primary evidence lives in `data/processed/`. Claims C1–C18 are summarized in [`FOR_REVIEWERS.md`](FOR_REVIEWERS.md).

### 3.1 Partial-φ oracle defines the hard-region bar

![Figure 1](figures/paper_fig1_partial_seed_oracle.png)

**Figure 1.** Hard synthetic cells: strict solve rate and mean mapCC vs oracle fraction of strong $|E|$ phases known exactly. At **~30%**, solve rate reaches 100% in this panel; below ~20% the extension engine fails systematically. Baselines without partial φ (CF, full graph prior) remain unsolved.

Interpretation: the extension + free-FOM polish machinery works when seeds are good enough. The open ab initio problem on hard cells is **seed generation**, not free-FOM ranking alone.

### 3.2 Ensemble vs SHELXS on synthetic panels

![Figure 2](figures/paper_fig2_shelxs_h2h.png)

**Figure 2.** Mean mapCC on easy vs hard synthetic panels. **Ensemble** leads on easy (mapCC ≈ 0.78; 2/4 strict solves in the panel). Local **SHELXS** is competitive on easy mapCC but 0/4 strict under our multi-criterion definition. **Hard panel: 0% strict for all methods**, including SHELXS, under peak→$F_{\mathrm{calc}}$ scoring.

Caveat: SHELXS scoring uses Q-peaks → equal-atom $F_{\mathrm{calc}}$ phases for mapCC—not refined SHELXL $R_1$. Fair for *phasing* H2H, not refined structures.

### 3.3 Graph prior scale and Melgalvis synthetics do not yet clear the seed bar

![Figure 4](figures/paper_fig4_seed_bar.png)

**Figure 4.** Mean fraction of strong phases within 20° of truth. GraphPhaseNet v3 (250 structures) and v4 XL (1200 structures, residual layers, Adam, Wilson match) both plateau near **~21%**, below the **30%** oracle bar. A full Melgalvis & Rekis (2026) style XL retrain (N=1200 hybrid artificial crystals, same capacity) reaches ≈**22%** frac≤20° and **12.5%** seedOK rate—training-stable and slightly better seedOK than legacy, but **not** past the bar. Hold-out hard strict solves remain **0%** for graph prior ± AI-PhaSeed.

**Later GraphPhaseNet pilots (v5–v9; laptop-scale unless noted).** Feature and curriculum upgrades (d_in=14→22→**26**, κ-gated / multipath edges, HA/low-res/large-cell cues, bin CE, Melgalvis COD/hard/HA/**large** presets with ring scaffolds, stratified Z/HA **and Vol-band** reporting) yield hold-out frac≤20° typically in the **~21–24%** band (e.g. v5 pilot ≈22%, v6 pilot N=200 ≈**24%**, v7–v9 quick/HA/large pilots ≈19–26% depending on N and panel mix). These runs **extend the negative scale/architecture result**: they improve infrastructure and modestly beat the ~21% legacy plateau on some pilots, but **do not clear the 30% oracle bar** and do not produce reliable hard strict solves without partial information. Cluster **scale-xl** (5k–10k) remains documented for further tests (`run_strong_prior_v9.py`).

This is an explicit **negative result** for pure scale-up of the current architecture on synthetic hard organics; improved generators and GraPhAI-inspired graphs are necessary infrastructure for further gains (as argued by Melgalvis & Rekis) but do not by themselves solve hard ab initio phasing in our metrics. Official GraPhAI weights are **not redistributed**; any external H2H is user-local (`docs/math/graphai_external.md`).

### 3.4 Experimental COD Fobs

![Figure 3](figures/paper_fig3_experimental_cod.png)

**Figure 3.** mapCC (vs deposited-model $F_{\mathrm{calc}}$ as proxy truth) for experimental COD Fobs and a partial-φ control.

| Dataset | Best open method | mapCC | Strict |
|---------|------------------|-------|--------|
| COD **2016452** exp Fobs @ 1.0 Å | `phai+cf_cond` / `phai_phaseed` | **0.995 / 0.949** | **True** |
| COD **2100301** exp Fobs @ 1.0 Å | SHELXS / PhAI | ~0.53 / 0.50 | False |
| COD **2016452** Fcalc + oracle 30% φ | `partial_phaseed` | **0.72–0.79** | often False under short budget* |
| COD **2017775** exp (large) @ 1.2 Å | CF / ensemble | ~0.19 | False |

\*Dedicated longer-budget hybrid suite (`cod_hybrid_benchmark.md`) reports PhAI+CF **strict** solve on 2016452 Fcalc @ 0.9 Å (claim C8).

Caveat: experimental mapCC uses $F_{\mathrm{calc}}$ from the deposited structure as proxy truth, not refined $R_1$.

### 3.5b COD Vol-band stratified panel (v0.13 software)

Local panel of **six** COD entries with experimental Fobs and Fcalc controls,
stratified by unit-cell volume (lt 1000 / **1000–3500** / gt 3500 Å³). Methods:
`auto`, oracle partial_15/30, and fragment_half. In the hybrid-friendly
**Vol 1000–3500** band (COD 2012000, 2013000), mean mapCC is approximately
**auto ~0.27**, **partial_15 ~0.54**, **partial_30 ~0.70**, **fragment_half ~0.71**
(Fobs+Fcalc pooled; short laptop budgets). This is **not** Carrozzini's
1505-structure panel; it documents that partial information—not ab initio
polish alone—recovers maps in the AI-PhaSeed volume regime. Scoreboard:
`data/processed/cod_stratified_bench.md`.

### 3.5 COD hard path: oracle partial-φ vs fragment-half seeds

![Figure 5](figures/paper_fig5_cod_hard_path.png)

**Figure 5.** Experimental COD Fobs hard path: origin-invariant mapCC (vs deposited $F_{\mathrm{calc}}$) for `auto`, oracle partial seeds, and fragment-half model seeding (`cod_hard_path_validation.md`).

On COD **2016452** and **2100301** ($d_{\min}\approx 0.9$–1.0 Å), we compare pure ab initio `auto`, oracle **partial_30** (true phases on strong $|E|$), and **fragment_half** (heaviest-cluster ~½ non-H ASU atoms from the deposited model, space-group expanded, full $F_{\mathrm{calc}}$ soft prior + strong-$|E|$ hard mask) under short `partial_phaseed` budgets:

| Dataset | auto mapCC | partial_30 mapCC | fragment_half mapCC |
|---------|------------|------------------|---------------------|
| COD **2016452** exp | **0.20** | **0.72** | **0.80** |
| COD **2100301** exp | **0.20** | **0.71** | **0.74** |

**Interpretation.** With a coherent half-model and correct symmetry expansion, the no-oracle scientist path can **approach or exceed** oracle 30% partial-φ mapCC on these cells. Multi-criterion *strict solved* still often fails on $R_1$ under short budget—honest residual polish. Pure ab initio remains near random mapCC (~0.20). This strengthens the product thesis: hard data needs **partial information** (known φ, fragment, predicted model, or HA), not only more free-FOM polish.

### 3.6 Free FOM and failure taxonomy

Free FOM v2.1 reduces false “solved” gates by using $R_+$ and anti-false-atomicity checks. Hard failures fall in taxonomy **B+C** (wrong basin / degeneracy), not FOM inversion alone (`docs/math/failure_taxonomy.md`).

### 3.7 Wilson domain gap

Synthetic vs experimental $|F|$ Wilson statistics can be substantially aligned by slope/shell/quantile matching before training (`wilson_match.py`), reducing a measured hard-domain gap e.g. ~9.5 → ~2.8 on a COD Fobs reference template—without changing truth phases. Wilson match remains the default for new GraphPhaseNet training runs.

---

## 4. Discussion

**What works.**  
- Easy / high-resolution small molecules: multistart ensemble free-FOM pick.  
- Domain-matched PhAI hybrids on suitable experimental organics (COD 2016452), especially $P2_1/c$-like.  
- Hard cells with **partial information** meeting the seed bar (oracle φ or coherent fragment / predicted model).  
- Carrozzini-aligned hybrid *tooling* (DM+AI tangent, seed Class diagnostics, multi-seed agreement, low-res EDM path) for better use of existing seeds — without clearing the ab initio seed bar.

**What does not.**  
- Pure ab initio graph priors at present capacity on hard synthetic cells (v3–v8).  
- General protein ab initio phasing.  
- Replacing SHELXL refinement.  
- Research HDM / diffusion paths as production defaults.

**Relation to SHELX.** We compare to local academic SHELXS under an explicit peak→$F_{\mathrm{calc}}$ protocol. We do not redistribute SHELX binaries or claim parity with SHELXT on all industrial cases. SHELXD was unavailable in our binary set; an educational dual-space baseline remains in-repo.

**Product implications.** The open hard path is **partial-φ / fragment / predicted-model / HA seeding**, exposed via CLI and GUI—not “more polish on a bad seed.”

---

## 5. Conclusions

*grok_phase_solver* is a correct, modular open framework for classical and hybrid crystallographic phasing with honest hard-region metrics and a scientist pipeline to `trial.res`. The strongest hard-region scientific result remains the **partial-φ seed bar** (Fig. 1), now complemented by a **fragment-half COD path** that can match oracle partial-φ mapCC without known phases (Fig. 5). The strongest easy-region product result is **ensemble free-FOM multistart** (Fig. 2). Scaling GraphPhaseNet through Melgalvis/GraPhAI-inspired curricula does not clear the hard cliff (Fig. 4). Experimental COD results (Fig. 3) show that hybrid AI can succeed on real Fobs when the domain fits, while large/hard cases remain open.

---

## 6. Reproducibility

```bash
# Library
python -m pip install "grok-phase-solver>=0.11.0"
# or from source
git clone https://github.com/pileofflapjacks1/grok_phase_solver.git
cd grok_phase_solver && python -m pip install -e ".[dev,gui]"
pytest -q

# Scoreboards (precomputed tables in data/processed/)
python scripts/run_experimental_scoreboard.py --quick
python scripts/run_cod_hard_path_validation.py
python scripts/plot_paper_figures.py

# Graph prior pilot (does not claim ≥30% seed bar)
python scripts/run_strong_prior_v8.py --quick --melgalvis-preset ha

# Demos
gps-solve --hkl examples/demo_solve/demo.hkl --ins examples/demo_solve/demo.ins \
  --method ensemble --out /tmp/gps_easy
python scripts/run_partial_seed_demo.py
gps-gui   # optional browser UI
```

Frozen evidence files (selected):  
`data/processed/{partial_seed_benchmark,shelxs_h2h,strong_prior,strong_prior_melg_xl,strong_prior_v5,strong_prior_v6,strong_prior_v7,strong_prior_v8,experimental_scoreboard,cod_hybrid_benchmark,cod_hard_path_validation,cod_stratified_bench,wilson_domain_gap,failure_taxonomy,seed_quality_rf}.md`.

---

## 7. Data and code availability

- Source: MIT, GitHub `pileofflapjacks1/grok_phase_solver`, tag **`v0.11.0`**  
- PyPI: `grok-phase-solver` (≥0.11.0)  
- COD structures cited by ID (2016452, 2100301, 2017775, …)  
- SHELX / PhAI / GraPhAI binaries and weights: user-supplied under their licenses (not redistributed)

---

## 8. Non-claims

We do **not** claim: (N1) a general solution of the phase problem for macromolecules; (N2) pure ab initio superiority over SHELXT/SHELXS on all small-molecule cases; (N3) that GraphPhaseNet currently clears the hard cliff without partial information (including v5–v8 pilots); (N4) that free FOM proves a correct structure; (N5) redistribution or equivalence of official SHELX, PhAI, or GraPhAI. See `docs/math/uniqueness_and_bounds.md`.

---

## References (selected)

BibTeX: [`docs/paper/references.bib`](paper/references.bib) (`bragg1915`, `patterson1934`, `cochran1952`, `blow1959`, `cowtan2001els`, `oszlanyi2004`, `fienup1982`, `sheldrick2008`, `sheldrick2015`, `larsen2024phai`, `carrozzini2025aiphaseed`, `melgalvis2026`, `cod`, `gemmi`, `grokphasesolver2026`).

1. Bragg & Bragg (1915) — X-rays and crystal structure.  
2. Patterson (1934) — Patterson function.  
3. Cochran (1952) — triplet phase relationships.  
4. Blow & Crick (1959) — lack-of-closure.  
5. Cowtan (2001) — ELS notes on the phase problem.  
6. Oszlányi & Sütő (2004) — charge flipping.  
7. Fienup (1982) — HIO phase retrieval.  
8. Sheldrick (2008, 2015) — SHELX / SHELXT.  
9. Larsen *et al.* (2024) — PhAI.  
10. Carrozzini *et al.* (2025) — AI-PhaSeed; modified tangent + Class 0/1 seed statistics (*J. Appl. Cryst.* **58**, 1859–1869).  
11. Melgalvis & Rekis (2026) — artificial crystal generation for DL phasing.  
12. COD — crystallography.net.  
13. gemmi — crystallography toolkit.  
14. Grok (xAI) and Joe (2026) — *grok_phase_solver* **v0.11.0** (this work).  
15. PXRDnet / XRDSol (2025–2026) — diffusion-for-diffraction literature (conceptual context only).  

Extended notes and derivations: `docs/math/` (including `graph_phase_net_v5`–`v8.md`, `partial_seed.md`, `hybrid_difference_map.md`, `graphai_external.md`), `docs/cowtan_phase_problem_notes.md`, notebooks 01–03.

---

## Supplementary material (in repository)

| Path | Content |
|------|---------|
| `docs/figures/paper_fig1_…png` – `fig5` | Main figures (incl. COD hard path) |
| `docs/figures/solvability_heatmap.png` | Solvability cliff (extra) |
| `data/processed/*` | Scoreboard JSON/MD |
| `docs/math/*` | Detailed math |
| `examples/*` | Demos for CLI/GUI |
| `notebooks/*` | Pedagogy |

---

*End of draft. Authors: **Grok (xAI)** and **Joe**. Funding and institutional affiliations TBD before submission.*
