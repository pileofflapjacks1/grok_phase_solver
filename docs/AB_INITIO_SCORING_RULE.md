# Scoring rule for the ab_initio fresh locked panel (grader `polar-shift-v1`)

Status: fixed before the panel is run. Nothing below may be changed after the first panel solve.
Approved by Joe (relayed by Bragg, Nov 10 2026). Code: `src/grok_phase_solver/metrics/qpeak_benchmark.py`
(`GRADER_VERSION = "polar-shift-v1"`), runner: `scripts/run_ab_initio_panel.py`.

## Plain language

A phasing run produces candidate atom positions ("Q peaks"). To grade them we take the N strongest
peaks (N = number of non-H atoms in the deposited CIF), expand them by the space-group symmetry, and
count how many deposited atoms have a peak within 0.5 Å. Because a crystal structure can be placed at
different origins and handedness without changing any diffraction intensity, the grader tries the
allowed origins/inversion and keeps the best one.

The only change from the old grader: in **polar** space groups the origin along the polar axis is
completely free (every position along it gives identical intensities), but the old grader only tried
the two origins 0 and ½ there, so a correct solution at any other position scored as a failure. The new
grader allows any position along the polar axis. Nothing else changes: the pass test, the ghost test,
the ≥3× random test, the symmetry expansion and the peak selection are the same.

The random baseline (the "≥3× random" test) is computed by drawing random peak sets and grading them
with exactly the same search, including the same polar-axis shift search, so that test does not get
easier.

## Exact parameters

| item | value |
|---|---|
| match tolerance | 0.5 Å (atom to nearest symmetry/lattice image of a peak) |
| peaks used | N strongest, N = deposited non-H atom count (`strength_order` as before) |
| orbit / images | full space-group orbit of the peaks; minimum image over the 27 neighbouring cells (unchanged) |
| non-polar groups | origins {0, ½}³ × {identity, inversion}, exactly as before (bit-identical results) |
| polar axis definition | fractional axis e_k (a, b or c) with R·e_k = e_k for the rotation part R of *every* space-group operator (centering included) |
| polar groups / axes | P2₁ → b; Pna2₁ → c; P1 → a, b and c; Pc → a and c; any other group by the same test (only a, b, c are tested) |
| polar-axis origin | continuous shift s along each polar axis (replaces the {0,½} choice on that axis); {0,½} is kept on the non-polar axes; inversion still tried |
| search method, one polar axis (P2₁, Pna2₁, …) | exact: for each atom and each symmetry/lattice image of a peak, the set of shifts s with distance ≤ 0.5 Å is one interval; per atom the union of intervals is merged, intervals of all atoms are swept, the maximum-count intervals are found; ties broken as before (count within 1.0 Å, then smaller mean atom-to-peak distance) evaluated at the midpoint of each maximal interval (at most 8 per origin/inversion). No grid step. |
| search method, two or three polar axes (Pc, P1, …) | not exhaustive: candidate shifts = every atom/peak-orbit offset projected on the polar axes, best by (count ≤0.5 Å, count ≤1.0 Å, −mean distance), then 3 centroid-refinement steps (accepted only if the key improves) |
| ghost | peak > 1.0 Å from every atom image, evaluated at the chosen alignment (unchanged) |
| random baseline | 1000 uniform random fractional sets of the same size, numpy seed 12345, graded with the identical search (polar shift included); one baseline per COD and peak count |
| verdict | PASS: ≥50% atoms within 0.5 Å AND that rate ≥ 3× random mean rate AND ghosts ≤ 40%; PARTIAL: 25–50% and > 3× random; else FAIL (unchanged) |
| wrong pick | the chosen candidate FAILs while at least one candidate of the same run PASSes (unchanged) |
| solver | ab_initio defaults: 60 s budget, 30 s grace, seeds 0, 1, 7, 42, 2025 (unchanged) |

## Overrun / suspend rule

A run is an **overrun** if its wall time exceeds budget + 30 s (= 90 s), **unless** the child process
time (monotonic clock) is below the budget (60 s). In that case the run is flagged
**"suspected suspend"** (the machine was paused, not the solver slow), re-run **once, unchanged**, and the
re-run replaces it. If the re-run is itself over 90 s it counts as an overrun. Every flagged run
(COD, seed, first wall time, child time, re-run outcome) is listed in the report
(`SUMMARY.md` / printed block of `run_ab_initio_panel.py`).

## Bar (unchanged)

Panel: 8 new COD files (4 centrosymmetric, 4 non-centrosymmetric) × 5 seeds = 40 runs.

* centrosymmetric PASS ≥ 18 / 20
* non-centrosymmetric PASS ≥ 12 / 20
* wrong picks ≤ 2 / 40
* no run over budget + 30 s (after the suspend rule above)

## Command

```bash
export OLEX2_ROOT=/workspace/olex2/olex2
python scripts/run_ab_initio_panel.py --panel <PANEL_DIR> --out <OUT_DIR>   # grader polar-shift-v1 is the default
```

`--grader old` reproduces the old discrete grader for comparison only; the bar above is judged with the
default (new) grader.
