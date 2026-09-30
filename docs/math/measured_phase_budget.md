# Measured-phase information budget

```
Ticket:              P2 (+ optional P1 audit / P3 trail / hygiene)
Information source:  measured-φ (simulated) / fragment / n/a (docs)
Method:              partial_phaseed + measured_phase_noise
Vol-band / d_min / N: hard P1 Vol=308 Å³ d_min=1.7 N=14; COD 2012000 Fcalc Vol=1027 Å³ d_min=1.0 N_cell=118 (ASU n_nonH=27)
Seed quality:        frac≤20° (all strong, mean) = 32%   (bar = 30%)   C4 crossings = 6/18   Class 0/1 = n/a (simulator)
Strict success:      mixed across grid  (6/18)
  mapCC_OI= 0.575 (mean)   peak recovery= 0.814 (mean)   R1= 0.415 (mean)
Free FOM:            rank only; composite=0.756  R₊=0.216
Claim status:        does not touch freeze C1–C25; new simulator only
PR / path:           feat/measured-phase-budget
Next action:         If a lab measured-φ seed has ≥~30% of strong |E| within 20°, run partial_phaseed then inspect trial.res in Olex2 / SHELXL.
What we will not do: GraphPhaseNet v12; claim detector exists
Blocker for Joe:     none (--quick used in-repo 2012000.cif; --full would add the other five stratified ids)
```

The detector article in the workspace is **conceptual**; this module is a
**simulator**. A phase-sensitive detector was not built. Do not quote σ_φ as
experimental or “beamline typical.”

Link: C4 bar and the partial-φ path — [partial_seed.md](partial_seed.md).
Oracle curves: [`data/processed/partial_seed_benchmark.md`](../../data/processed/partial_seed_benchmark.md).

## Physics

The experiment measures intensities. The structure is

$$\rho(\mathbf{r}) = \frac{1}{V}\sum_h |F(h)|\,e^{i\varphi(h)}\,e^{-2\pi i \mathbf{h}\cdot\mathbf{r}}$$

$|F|$ is data. $\varphi$ is not, unless a seed, a heavy atom, a fragment,
SIR/MAD, a predicted model, or a measured phase puts it there.

This note does **not** apply the 20° window to centrosymmetric Patterson
(PATT is a 0/π sign problem on $|F|^2$). The 20° window is the
non-centrosymmetric TREF / C4 bar.

## Wrap model

Code: `src/grok_phase_solver/physics/measured_phase_noise.py`

Angle wrap reuses `metrics.phase_error.wrap_phase` (radians → (−π, π]),
mapped to degrees (−180, 180].

$$\varphi_{\mathrm{meas}} = \mathrm{wrap}(\varphi_{\mathrm{true}} + \varepsilon)$$

$\varepsilon$ has circular standard deviation $\approx \sigma_\varphi$:

- **von Mises** (default): concentration $\kappa$ from
  $R = I_1(\kappa)/I_0(\kappa) = \exp(-\sigma^2/2)$ inverted with the
  Sra/Fisher piecewise approximation, $\sigma$ in radians.
- **wrapped normal**: add $\mathcal{N}(0,\sigma_\varphi)$ in degrees, then wrap.

$\sigma_\varphi \to 0$ recovers $\varphi_{\mathrm{true}}$. Large $\sigma_\varphi$
sends $\mathrm{frac}\le 20^\circ$ toward chance ($40/360 \approx 11\%$).

## Optional $|E|$-dependent width

Simulator schedule, **not** a calibration:

$$\sigma(|E|) = \max\bigl(\sigma_{\mathrm{floor}},\; \sigma_0 / \max(|E|/E_{\mathrm{ref}}, \varepsilon)\bigr)$$

Default $E_{\mathrm{ref}}=1$, $\sigma_{\mathrm{floor}}=1^\circ$. Stronger $|E|$
→ narrower $\sigma$. Do not call this “beamline typical.”

## Measured fraction

**Measured fraction is among the strong-$|E|$ set only.**

The strong set is the top 30% of reflections by $|E|$ (the C4 / AI-PhaSeed
seed set), unless `e_min` is given ($|E|\ge e_{\min}$).

`--measured-phase-frac` $f$ then keeps the highest-$|E|$ fraction $f$ of that
strong set. $f=1$ measures the whole strong set; $f=0.3$ measures 30% of the
strong set (about 9% of all reflections).

The seed is the same CSV / `(mask, φ)` object `partial_phaseed` already
consumes (`write_phase_seed_csv` / `load_phase_seed_csv`). There is no new
solver. CLI flags `--measured-phase-sigma-deg` and `--measured-phase-frac`
default **off** and corrupt an existing seed path.

## How frac≤20° is computed when only a subset is measured

Let $S$ be the strong-$|E|$ index set, $M$ the measured mask, $\Delta\varphi$
the wrapped error in degrees.

- **Among measured strong:** $\mathrm{frac}\le 20^\circ = \mathrm{mean}(|\Delta\varphi|\le 20^\circ)$ on $M\cap S$.
- **Among all strong (C4 metric):** unmeasured members of $S$ count as
  *not* within 20°. So

$$\mathrm{frac}\le 20^\circ_{\text{all strong}} = \frac{\#\{h\in M\cap S: |\Delta\varphi(h)|\le 20^\circ\}}{|S|}$$

Perfect measurement of a fraction $f$ of $S$ therefore scores $\approx f$
on the C4 metric. Measuring 10% of strong $|E|$ perfectly **cannot** cross
the 30% bar. Measuring 100% of strong $|E|$ with large $\sigma_\varphi$ may
still miss it.

## What crosses the C4 line

C4: $\gtrsim\sim 30\%$ of strong-$|E|$ phases correct within $\sim 20^\circ$
lets AI-PhaSeed / `partial_phaseed` strict-solve hard cells
($n\gtrsim 12$, $d_{\min}\gtrsim 1.5\,\text{Å}$) on the in-repo oracle
curves. GraphPhaseNet v3–v11 stays ~21–24% ≤20° (documented negative;
architecture frozen at v11).

Gaussian approximation $P(|\Delta\varphi|\le 20^\circ)\approx
\mathrm{erf}\bigl(20/(\sigma_\varphi\sqrt{2})\bigr)$ times the measured
fraction predicts, on the `--quick` grid:

| σ_φ | frac measured | predicted ≤20° (all strong) | vs C4 30% |
|-----|---------------|------------------------------|-----------|
| 10° | 0.10 | ~10% | below |
| 10° | 0.30 | ~29% | on the line |
| 10° | 1.00 | ~95% | above |
| 20° | 0.30 | ~20% | below |
| 20° | 1.00 | ~68% | above |
| 45° | 1.00 | ~33% | on/above |
| 45° | 0.10 | ~3% | below |

**This `--quick` run** (hard P1 n=14, $d_{\min}=1.7$ Å + COD 2012000 Fcalc):
C4 crossings were the six $f=1.00$ points (all strong $|E|$ measured) at
$\sigma_\varphi\in\{10^\circ,20^\circ,45^\circ\}$ on both cells. Measuring
30% of strong $|E|$ at $\sigma_\varphi=10^\circ$ landed at 28–29% ≤20°
(on the line; COD 2012000 strict-solved, hard P1 did not). Measuring 10%
never crossed C4 and never strict-solved. Full strong-set at
$\sigma_\varphi=45^\circ$ was C4-yes (~38–39%) with mixed strict success
(COD yes / hard P1 mapCC 0.630). Scoreboard:
[`data/processed/measured_phase_budget.md`](../../data/processed/measured_phase_budget.md).

Strict success is the triple mapCC_OI ≥ 0.7 **and** peak recovery ≥ 0.5
**and** R1 ≤ 0.45. Free FOM ranks trials; it does not prove a structure.

Figure: [`docs/figures/measured_phase_budget.png`](../figures/measured_phase_budget.png)
(two views: strict-success vs $\sigma_\varphi$, and vs measured fraction;
dotted C4 line at 30% ≤20°). Caption: hardware does not exist.

## How to regenerate

```bash
python scripts/run_measured_phase_budget.py --quick
python scripts/run_measured_phase_budget.py --full
```

`--quick` (default unless `--full`): σ_φ ∈ {10°, 20°, 45°}, fraction
∈ {0.10, 0.30, 1.00}, one hard synthetic P1-like cell from the existing
partial-seed generator (`n=14`, $d_{\min}=1.7$ Å) plus COD **2012000**
(true mid-band Vol ~1027 Å³ on the stratified catalog). `--full` loops
the existing partial-seed case list and the six COD stratified ids.
Missing panel files are skipped and named, not replaced with a new
synthetic curriculum.

This does not change paper Figs 1–6 or claims C1–C25.
