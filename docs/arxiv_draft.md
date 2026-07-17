# Toward an Open Physics/AI Framework for the Crystallographic Phase Problem

**Working draft (Lane C review pack)** · Version 0.2.1 · MIT  
**Code:** https://github.com/pileofflapjacks1/grok_phase_solver  
**One-pager:** [`FOR_REVIEWERS.md`](FOR_REVIEWERS.md)

> Status: methods preprint skeleton fleshed with **supported claims** and
> **honest non-claims**. Numbers match repo scoreboards under `data/processed/`.
> Not submitted; authors TBD.

---

## Abstract

We present *grok_phase_solver*, an open Python framework that unifies classical
solutions of the X-ray crystallographic phase problem—charge flipping, HIO,
RAAR, difference-map projections, Patterson and direct methods, isomorphous
difference Patterson, and density modification—with hybrid and learned priors
(AI-PhaSeed, GraphPhaseNet, optional PhAI). All algorithms act on measured
amplitudes $|F(hkl)|$ and are evaluated with origin-invariant map correlation,
R1, peak recovery, free figures of merit, and a strict multi-criterion success
definition. On easy synthetic cells, multistart ensemble phasing is competitive
with or better than local SHELXS under our protocol. On hard cells
($n\gtrsim 12$, $d_{\min}\gtrsim 1.5$ Å), pure ab initio methods including
scaled graph priors remain ~0% strict success; an oracle partial-φ path shows
that ≥~30% of strong $|E|$ phases within ~20° of truth enables strict solves via
phase extension. Scaling GraphPhaseNet (1200 structures, residual GNN, Adam)
does not lift mean strong-phase accuracy above ~21% within 20°. We ship a
scientist CLI (`gps-solve`) with fragment/HA seed importers and SHELXL-ready
`trial.res` export. We do **not** claim a general macromolecular solution.

---

## 1. Introduction

Recovering phases $\varphi(\mathbf{h})$ from amplitudes alone is the classical
phase problem:

$$
\rho(\mathbf{r})
=
\frac{1}{V}\sum_{\mathbf{h}}
|F(\mathbf{h})|\,e^{i\varphi(\mathbf{h})}\,e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}.
$$

Industrial small-molecule pipelines (SHELXT/SHELXS + SHELXL/Olex2) solve most
atomic-resolution organics; hard cases and open science still benefit from
transparent, modular baselines. Recent neural work (e.g. PhAI) shows strong
results on restricted domains when weights and packing are carefully matched.

**Contributions of this work**

1. An **integrated open stack**: classical solvers, free FOM, hybrid polish,
   learned priors, experimental I/O, and external SHELXS runners (binaries not
   redistributed).
2. **Honest hard-region science**: failure taxonomy; seed-quality bar
   (30% / 20°); Lane A scale negative result for pure graph priors.
3. **Product path**: easy → ensemble; hard → partial-φ / fragment / HA seeds
   (`gps-make-seed`, `report.md` diagnostics); `trial.res` → SHELXL.
4. **External calibration**: SHELXS H2H; experimental COD Fobs scoreboard;
   Wilson domain-gap closing.

---

## 2. Mathematical background (pointers)

Full notes live in `docs/math/`:

| Topic | Doc |
|-------|-----|
| Phase problem overview | `math/phase_problem_overview.md` |
| Free FOM v2.1 ($R_+$) | `math/free_fom.md` |
| Failure taxonomy A/B/C | `math/failure_taxonomy.md` |
| Partial-φ / seed bar | `math/partial_seed.md` |
| GraphPhaseNet prior | `math/strong_prior.md` |
| Wilson domain gap | `math/wilson_domain_gap.md` |
| Uniqueness / non-claims | `math/uniqueness_and_bounds.md` |
| Cowtan ELS notes | `cowtan_phase_problem_notes.md` |

---

## 3. Methods

### 3.1 Classical and projection methods

Charge flipping, HIO, RAAR, DiffMap, direct methods (E-values, triplets, tangent),
Patterson peak pick, difference Patterson for HA vectors, Blow–Crick SIR/MIR
FOMs, solvent flattening / density modification.

### 3.2 Free FOM and ensemble

Truth-free composite ranking uses positivity residual $R_+$ (not vacuous
post-modulus $R$), atomicity scores, and calibrated gates. Multistart
**ensemble** (CF+RAAR) picks by free FOM — strongest *in-repo* easy ab initio
path in our H2H.

### 3.3 AI-PhaSeed and partial seeds

Strong reflections are held as seeds; phase extension + free-FOM-gated polish
extends to the full set. Seeds may come from PhAI, GraphPhaseNet, oracle
partial φ, fragment Fcalc (SHELXS `.res` / peaks), or HA heuristics
(`solvers/seed_import.py`).

### 3.4 Learned priors

- **hard-P1 PhaseMLP** and **GraphPhaseNet** (triplet-graph residual MP, Adam,
  Wilson-matched train, strong-|E| loss).
- Optional **PhAI** weights (user-supplied; not redistributed).

### 3.5 Scientist pipeline

`gps-solve` / `gps-make-seed`: SHELX HKL/INS, CIF HKL, MTZ → phases, density,
peaks, **trial.res**, `report.md` with seed-quality section.

### 3.6 Success metrics

Strict success: mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5 **and** R1 ≤ 0.45
(`metrics/success.py`). Strong-seed bar: ≥30% of top-30% $|E|$ phases within 20°.

---

## 4. Experiments and results

Primary tables are generated artifacts under `data/processed/`. Summary of
**supported claims** (see also FOR_REVIEWERS C1–C8):

| # | Finding | Evidence |
|---|---------|----------|
| C1–C2 | Ensemble best/competitive open ab initio on **easy** synthetic vs CF/SHELXS | `shelxs_h2h.md` |
| C3 | Hard ab initio ~0% strict (CF, ensemble, dual-space, graph+PhaSeed, SHELXS) | same + `strong_prior.md` |
| C4 | ≥~30% strong φ within 20° → hard strict solves (oracle) | `partial_seed_benchmark.md` |
| C5 | Graph prior ~**21%** ≤20° even after **v4 XL** (1200 structs) | `strong_prior.md` |
| C6 | Free FOM v2.1 reduces false “solved” gates; hard failures B+C | `failure_taxonomy.md` |
| C7 | Wilson synth→exp gap closable by amplitude matching | `wilson_domain_gap.md` |
| C8 | PhAI hybrids can strict-solve COD 2016452 Fcalc @ 0.9 Å (fair suite) | `cod_hybrid_benchmark.md` |

### 4.1 Experimental / COD scoreboard (Lane C)

`scripts/run_experimental_scoreboard.py` → `experimental_scoreboard.md`:

- Demo easy cell (free FOM ranking)
- COD **2016452** and **2100301**: Fcalc controls + **experimental Fobs**
- Oracle **partial_phaseed** 30% on Fcalc (product hard path)
- COD **2017775** large experimental Fobs (struggle expected)
- Optional local SHELXS and GraphPhaseNet prior

**Headline experimental results (this repo run):**

| Dataset | Best open method | mapCC | Strict |
|---------|------------------|-------|--------|
| COD 2016452 **exp** Fobs @ 1.0 Å | `phai+cf_cond` / `phai_phaseed` | **0.99 / 0.95** | **True** |
| COD 2100301 **exp** Fobs @ 1.0 Å | `shelxs` / `phai_phaseed` | 0.53 / 0.50 | False |
| COD 2016452 Fcalc 0.9 Å | `shelxs` (0.72); partial30 (0.79) | — | partial not full strict* |
| COD 2017775 exp @ 1.2 Å | CF/ensemble ~0.19 | — | False |

\*Strict requires mapCC + peak recovery + R1; high mapCC alone is not enough under our thresholds. Dedicated `cod_hybrid_benchmark` with longer settings reports PhAI+CF strict solve on 2016452 Fcalc @ 0.9 Å (C8).

Caveat: experimental mapCC uses Fcalc from deposited model as proxy truth, not
refined R1.

### 4.2 Lane A scale (negative for pure prior)

Residual GraphPhaseNet + Adam + 1200 Wilson-matched structures does not clear
the 30%/20° seed bar on mean metrics. Occasional individual cells hit seedOK;
strict hard solves remain 0%.

### 4.3 Lane B product path

Seed importers (`.res`, peaks, atoms, HA pair) and `gps-make-seed` make the
partial-φ path usable without hand-built phase tables.

---

## 5. Discussion

**What works.** Easy/high-resolution small molecules: ensemble and PhAI hybrids
(when weights match). Hard cells: partial information (HA/MAD/MR/fragment), not
more free-FOM polish alone.

**What does not.** Pure ab initio graph priors at current capacity; general
protein phasing; replacement of SHELXL refinement.

**Relation to SHELX.** We calibrate against local academic SHELXS; we do not
redistribute SHELX or claim industrial parity on all cases. SHELXD binary was
not available in our environment; dual-space educational baseline remains
in-repo.

**Open problems.** Hit seed bar with qualitatively different models (equivariant
nets, DM-hybrid features, 10⁴-scale data); richer experimental panels; GUI.

---

## 6. Conclusions

*grok_phase_solver* is a correct, modular open framework for classical and hybrid
phasing with honest hard-region metrics. The strongest scientific result for hard
cells is the **partial-φ seed bar**; the strongest product result for easy cells
is the **ensemble free-FOM path**. Scale alone does not solve the hard cliff.

---

## Reproducibility

```bash
git clone https://github.com/pileofflapjacks1/grok_phase_solver.git
cd grok_phase_solver && python -m pip install -e ".[dev]"
pytest -q
python scripts/run_experimental_scoreboard.py
python scripts/train_strong_prior.py --scale-xl --wilson-match  # optional, ~30 min
python scripts/run_partial_seed_demo.py
# Optional academic SHELXS:
# python scripts/run_shelxs_h2h.py --quick
```

---

## References (selected)

- Bragg (1915); Patterson (1934); Cochran (1952); Blow & Crick (1959)
- Cowtan, ELS notes (2001); Oszlányi & Sütő (charge flipping); Fienup (HIO)
- Sheldrick SHELX suite; Larsen et al., *Science* (2024) PhAI
- COD (crystallography.net); gemmi

Full path bibliography: `docs/cowtan_phase_problem_notes.md`, package README cites.

---

## Supplementary

- Notebooks 01–03 (`notebooks/`)
- Scoreboard JSON under `data/processed/`
- Math notes under `docs/math/`
- Reviewer one-pager: `docs/FOR_REVIEWERS.md`
