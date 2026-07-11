# AI-PhaSeed

## Idea

Neural nets (e.g. **PhAI**) can produce useful phase estimates but often need
classical **phase extension** and density modification to fill weak reflections
and refine density. Unconditional charge flipping after PhAI can *destroy* a
good seed (observed on COD 2016452 at low resolution).

**AI-PhaSeed** (Carrozzini *et al.* 2025; building on Larsen *et al.* 2024) is:

1. AI phase vector \(\varphi^{\mathrm{AI}}(h)\)
2. Select a **seed set** of strong reflections (high \(|E|\))
3. Initialize other phases randomly
4. Iterate density modification + modulus projection, **re-imposing** seed phases
5. Optional free-FOM–gated polish

This targets **basin (B)** failure: put the iterate near the correct density
before free search.

## Algorithm (this repository)

```text
φ_AI ← PhAI_fair(|F|) or external
S ← top fraction of reflections by |E|
φ[S] ← φ_AI[S]   (optional discretize: centro / bins)
φ[∖S] ← random, softly blended toward φ_AI

for c = 1 … n_extend:
    ρ ← IFFT(|F| e^{iφ})
    ρ ← positivity (and optional solvent flatten)
    φ ← arg(FFT(ρ));  reimpose |F|
    φ ← blend(φ_AI, φ; prior_weight)     # soft full prior
    φ[S] ← reimpose(φ_AI[S]; w_seed)     # hard/soft seed
    anneal w_seed → w_seed_final

optional: conditional CF/RAAR polish if free FOM v2.1 accepts
multistart over random ∖S; pick max free-FOM composite
```

Implementation: `solvers/ai_phaseed.py`

| Function | Role |
|----------|------|
| `select_seed_indices` | strong \(|E|\) or \(|F|\) subset |
| `phase_extend` | extension loop with seed reimpose |
| `ai_phaseed_solve` | full pipeline from external AI phases |
| `phai_phaseed_solve` | PhAI fair → AI-PhaSeed |

## Relation to free FOM and taxonomy

- **Free FOM v2.1** ranks multistart extension trials and gates final polish.
- **Rewrite trust-region** blocks CF that rewrites phases without large \(R_+\) gain.
- Expected taxonomy shift: **B+C → near/solved** when AI seed is in-domain
  (e.g. COD \(P2_1/c\)); less help on out-of-domain synthetic P1 (see
  `phai_taxonomy.md`).

## Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `seed_fraction` | 0.25 | strong seed set size |
| `seed_weight` → `seed_weight_final` | 1.0 → 0.75 | anneal |
| `prior_weight` | 0.30 | soft full AI prior |
| `n_extend` | 15 | positivity ER-like cycles |
| `polish` | charge_flipping | free-FOM gated |
| `n_starts` | 2 | free-FOM pick |

## Empirical notes (this repo)

- **Oracle seed** (true \(\varphi\) as AI): mapCC \(\approx 1\) after extension — algorithm is sound.
- **Partial seed** (\(\sim\)55% true + noise): **solves** hard synthetic cells where CF fails (mapCC \(\sim 0.8\)–0.9). Shows AI-PhaSeed works when the prior is moderately good.
- **PhAI on random P1 synthetic**: still weak (domain gap; PhAI trained for COD-like \(P2_1/c\)).
- **Hard-P1 domain prior** (`models/hard_p1_prior.py`): origin-invariant PhaseMLP on hard P1; hold-out prior mapCC ~0.5 (vs ~0.3 random/PhAI-on-P1). Still rarely strict-solves — small MLP is a weak prior; algorithm is validated by oracle/partial seeds.
- **COD 2016452**: `phai_phaseed` improves over CF at low res; at 0.9 Å free-FOM–gated CF after PhAI (`phai_cf_cond`) remains the strict solver — extension alone does not replace a *helpful* polish when free FOM accepts it.

## References

1. Larsen, A. S., Rekis, T. & Madsen, A. Ø. (2024). *Science* **385**, 522–528 (PhAI).
2. Carrozzini *et al.* (2025). Phase-seeding / AI-PhaSeed hybrid protocols (IUCr journals).
3. Cowtan, K. Density modification / solvent flattening.
4. Project: `docs/math/free_fom.md`, `docs/math/failure_taxonomy.md`.
