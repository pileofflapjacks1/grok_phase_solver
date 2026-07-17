# Partial-φ / fragment seed (science track B)

## Motivation

Hard synthetic cells ($n \gtrsim 12$, $d_{\min} \gtrsim 1.5\,\text{Å}$) remain
**strict-unsolved** under CF, dual-space, and full AI priors (mapCC ≈ 0.5).
Oracle AI-PhaSeed tests show the *extension engine* works when enough strong
phases are correct. Track B makes that path explicit:

1. Measure **how much** known φ or fragment is enough (benchmark curves).
2. Ship a **user/API path** for partial phases from MAD/SIR/heavy-atom/MR-lite.
3. Quantify **Wilson domain gap** so synthetic priors can be re-matched to experiment.

## API

Code: `solvers/partial_seed.py`

| Function | Role |
|----------|------|
| `oracle_partial_seed` / `oracle_partial_phaseed_solve` | Benchmark: true φ on strong \|E\| mask ± noise |
| `fragment_seed_phases` / `fragment_phaseed_solve` | Partial atoms → $F_\mathrm{calc}$ phases → PhaSeed |
| `load_phase_seed_csv` / `write_phase_seed_csv` | File I/O: `h,k,l,phase_deg` |
| `partial_phaseed_solve` | Generic full-length seed → AI-PhaSeed |

### CLI

```bash
# CSV of known phases → extension + free-FOM polish
gps-solve --hkl data.hkl --ins data.ins \
  --method partial_phaseed \
  --phase-seed-csv known_phases.csv \
  --out ./out
```

CSV header: `h,k,l,phase_deg` (radians also accepted via API).

### Python

```python
from grok_phase_solver.solvers.partial_seed import (
    oracle_partial_phaseed_solve,
    fragment_phaseed_solve,
    load_phase_seed_csv,
    partial_phaseed_solve,
)

# Oracle fraction sweep (research)
ph, rho, info = oracle_partial_phaseed_solve(
    hkl, amp, cell, phases_true, fraction=0.30, mode="strong_E"
)

# Fragment / heavy-atom model
ph, rho, info = fragment_phaseed_solve(hkl, amp, cell, fracs_ha, elements_ha)

# Experimental partial phases
seed, mask, meta = load_phase_seed_csv("known.csv", hkl)
ph, rho, info = partial_phaseed_solve(hkl, amp, cell, seed, mask=mask)
```

## Benchmarks

```bash
python scripts/run_partial_seed_benchmark.py
# → data/processed/partial_seed_benchmark.{json,md}

python scripts/run_wilson_domain_gap.py
# → data/processed/wilson_domain_gap.{json,md}
```

### Expected science reading

| Curve | Question |
|-------|----------|
| Oracle fraction | At what known-strong fraction do hard cells **strict-solve**? |
| Noise @ 30% | How bad can seed MPE be before extension fails? |
| Fragment fraction | How large a true-atom subset as $F_\mathrm{calc}$ seed is enough? |
| Wilson gap | How far is hard synthetic \|F\| from experimental Fobs? |

## Product path (Lane B)

Scientist-facing seed importers (`solvers/seed_import.py`, CLI `gps-make-seed`):

| Source | Flag / tool |
|--------|-------------|
| Known φ CSV | `--phase-seed-csv` |
| SHELXS / fragment `.res` | `--phase-seed-res` (Fcalc seed) |
| `peaks.csv` | `--seed-peaks-csv` |
| Atoms CSV | `--seed-atoms-csv` |
| Isomorphous pair | `--native-hkl` + `--derivative-hkl` |
| Offline build | `gps-make-seed --from-res … -o seed.csv` |

`report.md` reports truth-free **seed quality**: strong-|E| coverage vs the 30% bar
and free FOM of the raw seed. Size adequacy ≠ correctness.

If oracle ≥30–40% strong φ **solves** while GraphPhaseNet does not, the cliff is
**prior quality**, not free-FOM or extension — invest in better seeds (partial
experimental phasing, heavier models, larger nets), not more CF polish.

## Wilson domain gap

`data/wilson.py`:

- `wilson_plot` — shell $\ln\langle I\rangle$ vs $s^2$
- `domain_gap_wilson` / `domain_gap_report` — slope + intensity quantiles + moments
- `mean_domain_gap_vs_experiment` — batch synthetic vs one experimental set

Lower `domain_gap_score` ⇒ closer amplitude statistics.

## Scope

Partial-φ is the production path for **hybrid** experiments (SIR/MAD fragments,
partial MR, heavy-atom substructure). It is **not** pure ab initio. Full ab initio
on hard cells remains open research.
