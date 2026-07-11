# Solvability frontier and fair PhAI comparison

**Date:** 2026-07-10  
**Scripts:** `scripts/run_solvability_diagram.py`, `scripts/run_fair_phai_benchmark.py`  
**Results:** `data/processed/solvability_diagram.*`, `data/processed/fair_phai_benchmark.*`

---

## 1. Strict success criterion

A trial is **solved** only if all hold:

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| mapCC_OI | ≥ 0.70 | Origin/enantiomorph-invariant density correlation |
| peak recovery | ≥ 0.50 | Non-H atoms near a density peak (origin search) |
| R1 | ≤ 0.45 | Carbons at top peaks vs \|F\| (unrefined; lenient) |

This is stricter than mapCC alone and closer to “can I start a structure?”

---

## 2. Solvability phase diagram (synthetic P1)

**192 trials:** \(n\in\{4,8,12,20\}\), \(d_{\min}\in\{0.9,1.2,1.5,2.0\}\) Å, completeness \(\in\{1.0,0.7\}\), methods CF / recycle / DM, 2 seeds.

### Overall success rates

| Method | Solved / Total | Rate |
|--------|----------------|------|
| charge_flipping | 18 / 64 | **28%** |
| phase_recycle | 2 / 64 | 3% |
| direct_methods (educational) | 1 / 64 | 2% |

### Charge flipping @ completeness = 1.0 (mean mapCC in parentheses)

| n_atoms \ d_min | 0.9 Å | 1.2 Å | 1.5 Å | 2.0 Å |
|-----------------|-------|-------|-------|-------|
| 4 | 50% (0.85) | 50% (0.85) | 50% (0.83) | 50% (0.67) |
| 8 | 50% (0.78) | 0% (0.56) | 0% (0.49) | 0% (0.45) |
| 12 | **100%** (0.89) | 50% (0.68) | 0% (0.51) | 0% (0.47) |
| 20 | 0% (0.49) | 0% (0.44) | 0% (0.39) | 0% (0.44) |

### Physics takeaway

1. **Atomic resolution + modest \(N\)** is the classical success region (atomicity / peak separation).  
2. **Increasing \(N\) or \(d_{\min}\)** collapses success — this *is* the open phase problem in numbers.  
3. Seed dependence is large (many 50% = 1/2 seeds) → multistart is essential.  
4. Educational DM ≠ SHELXD; do not over-read DM failure.  
5. Any new method must beat **this frontier under the same criterion**.

Heatmap: `docs/figures/solvability_heatmap.png`

---

## 3. Fair PhAI protocol

Aligned with public PhAI notebook + `crystallography_module.merge_reflections`:

1. **reindex_monoclinic** → \(k\ge 0\), \(l\ge 0\) (with locus fix)  
2. Average duplicate \(hkl\)  
3. Scale \(|F| \leftarrow |F| / \max|F|\)  
4. Pack into \((1,21,11,11)\) grid, `max_index=10`  
5. 5 neural recycle cycles, random 0/π phase init  

Code: `models/phai_fair.py` (`run_phai_fair`)

---

## 4. Fair PhAI results (Fcalc truth)

Primary structure: **COD 2016452** (P2₁/c, PhAI demo chemistry/cell).  
Secondary: **COD 2100301** (P2₁/c).

### COD 2016452 (mapCC_OI)

| d_min | CF | recycle | **phai_fair** | phai_fair+CF | solved? |
|-------|-----|---------|---------------|--------------|---------|
| 0.9 | 0.35 | 0.38 | **0.56** | **0.87** | **yes** (hybrid only) |
| 1.2 | 0.44 | 0.42 | **0.61** | 0.49 | no |
| 1.5 | 0.53 | 0.53 | **0.62** | 0.44 | no |
| 2.0 | 0.48 | 0.59 | **0.63** | 0.44 | no |

### COD 2100301 (mapCC_OI)

| d_min | CF | recycle | phai_fair | phai_fair+CF |
|-------|-----|---------|-----------|--------------|
| 0.9 | 0.45 | 0.35 | **0.48** | **0.64** |
| 1.2 | 0.40 | 0.46 | **0.58** | 0.39 |
| 1.5 | 0.51 | 0.46 | 0.41 | 0.36 |

### Interpretation (truth-seeking)

1. **Fair packing changes the story:** with official merge+max scaling, **PhAI alone beats CF on mapCC** for 2016452 at all listed resolutions (and often 2100301).  
2. **Hybrid win at atomic res:** `phai_fair+CF` on 2016452 @ 0.9 Å → mapCC **0.87**, peak rec 1.0, R1 0.34 → **strict solved**. Pure CF failed the same trial.  
3. **At lower res, CF polish can *hurt* PhAI** (overwriting a better neural prior with a bad classical trajectory). Hybrid policy should be conditional, not always CF.  
4. **Strict “solved” remains rare** on 60-atom cells: peak recovery can be high while R1 stays >0.45 without refinement/element assignment.  
5. We still do **not** claim reproduction of all Science paper experimental statistics — only a **fairer Fcalc protocol** on COD structures.

---

## 5. What this means for “solving the phase problem”

| Finding | Implication |
|---------|-------------|
| Solvability cliff with \(N\) and \(d_{\min}\) | Classical methods are insufficient alone for hard cases |
| PhAI > CF on mapCC with fair prep | Neural prior is real; data prep was the previous bottleneck |
| PhAI+CF solves 2016452 @ 0.9 Å under strict criteria | Hybrid architecture is the right research direction |
| R1 / peak criteria fail more often than mapCC | Need better atomic interpretation + refinement loop |

### Next research steps (status)

1. [x] Conditional hybrid (free-FOM gate): `conditional_hybrid.py`  
2. [x] Multistart CF+RAAR + free-FOM pick: `ensemble.py`, `scripts/run_ensemble_benchmark.py`  
3. [x] Physics-recycle net on hard region: `recycle_net.py`, `scripts/train_recycle_net.py`  
4. [x] COD 2016452 PhAI+RAAR conditional hybrid: `scripts/run_cod_hybrid_benchmark.py`  
5. [x] DiffMap retune (β, charge-flip \(P_S\)): `retune_difference_map`, `scripts/run_diffmap_retune.py`  
6. [x] Free-FOM v2 calibration + rewrite trust-region (`docs/math/free_fom.md`)  
7. [ ] Experimental HKL (not only Fcalc) with same success metrics at scale  
8. [ ] Larger equivariant nets / fine-tune PhAI on hard cells  
9. [x] Solvability failure taxonomy (selection vs optimization vs information) — `docs/math/failure_taxonomy.md`
