"""Score ``trial.res`` Q peaks against a deposited structure.

This is the pre-registered Q-peak bar. It was fixed before any numbers were
seen and must not be tuned to make a run pass.

* N = number of deposited non-H atoms in the CIF asymmetric unit.
* Use the N strongest Q peaks. Strength is the ``peaks.csv`` height, mapped
  onto the ``trial.res`` Q list through the space-group operators.
* Alignment search: origin shifts {0, 1/2}^3 x {identity, inversion}. The Q
  set is expanded to its full space-group orbit, and distances use the
  minimum lattice image, so symmetry mates and lattice translations count.
  The best alignment maximises atoms within 0.5 A (ties: within 1.0 A, then
  smaller mean atom-to-Q distance).
* Ghost = one of the N Q peaks that is more than 1.0 A from every image of
  every deposited non-H atom.
* Random baseline: ``n_random`` sets of N uniform fractional points in the
  same cell, scored with the same orbit expansion and alignment search.
  The seed is fixed so the baseline is reproducible.
* PASS: >= 50% of atoms within 0.5 A, that rate >= 3x the random mean, and
  ghosts <= 40%. PARTIAL: 25-50% within 0.5 A and > 3x the random mean.
  FAIL: anything else.

The scorer ignores atom types: Q peaks carry none. A uniform random point
set already has ~70-75% ghosts at a 1.0 A radius, so the ghost bound is
strict on purpose.
"""

from __future__ import annotations

import csv
import itertools
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gemmi
import numpy as np

DEFAULT_SEED = 12345
DEFAULT_N_RANDOM = 1000

_SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
_ORIGINS = np.array(list(itertools.product((0, 0.5), repeat=3)), float)

Ops = List[Tuple[np.ndarray, np.ndarray]]


def _num(s: str) -> float:
    return float(re.sub(r"\(.*\)", "", s))


def read_cif_nonh(path: str | Path):
    """Return ``(cell, hm, atoms)``; ``atoms`` is ``[(label, element, xyz), ...]`` without H/D."""
    block = gemmi.cif.read(str(path)).sole_block()
    cell = [_num(block.find_value("_cell_length_" + k)) for k in "abc"]
    cell += [_num(block.find_value("_cell_angle_" + k)) for k in ("alpha", "beta", "gamma")]
    hm = block.find_value("_symmetry_space_group_name_H-M") or block.find_value(
        "_space_group_name_H-M_alt"
    )
    hm = hm.strip("'\" ")
    table = block.find("_atom_site_", ["label", "fract_x", "fract_y", "fract_z", "type_symbol"])
    atoms = []
    for row in table:
        el = row[4] if row[4] not in ("?", ".") else re.match(r"[A-Za-z]+", row[0]).group(0)
        if el.upper() in ("H", "D"):
            continue
        atoms.append((row[0], el, [_num(row[i]) for i in (1, 2, 3)]))
    return cell, hm, atoms


def read_res_q(path: str | Path) -> np.ndarray:
    """Fractional coordinates of every ``Qn`` line in a SHELX ``.res``."""
    qs = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if p and re.match(r"^Q\d+$", p[0]) and len(p) >= 5:
                qs.append([float(p[2]), float(p[3]), float(p[4])])
    return np.array(qs, dtype=float).reshape(-1, 3)


def space_group_ops(hm: str) -> Ops:
    """All operators (with centering) as ``(rot, tran)`` in fractional units."""
    sg = gemmi.find_spacegroup_by_name(hm)
    if sg is None:
        raise ValueError(f"unknown space group {hm!r}")
    return [
        (np.array(op.rot, float) / 24.0, np.array(op.tran, float) / 24.0)
        for op in sg.operations()
    ]


def orth_matrix(cell: Sequence[float]) -> np.ndarray:
    return np.array(gemmi.UnitCell(*cell).orth.mat.tolist())


def orbit(points: np.ndarray, ops: Ops) -> np.ndarray:
    return np.concatenate([points @ m.T + t for m, t in ops])


def min_dist(a: np.ndarray, b: np.ndarray, orth: np.ndarray) -> np.ndarray:
    """(n, m) minimum-image Cartesian distances between fractional point sets."""
    d = a[:, None, :] - b[None, :, :]
    d -= np.round(d)
    best = None
    # Rounding alone is not enough in oblique cells; check the 27 neighbours.
    for s in _SHIFTS:
        r = np.sqrt((((d + s) @ orth.T) ** 2).sum(-1))
        best = r if best is None else np.minimum(best, r)
    return best


def evaluate(atoms: np.ndarray, atoms_orbit: np.ndarray, q: np.ndarray, ops: Ops,
             orth: np.ndarray) -> Dict:
    """Best alignment of ``q`` onto the atoms over origin shifts x inversion."""
    best = None
    for inv in (1, -1):
        for t in _ORIGINS:
            qa = inv * q + t
            d_atom = min_dist(atoms, orbit(qa, ops), orth).min(1)
            h05 = int((d_atom <= 0.5).sum())
            h10 = int((d_atom <= 1.0).sum())
            key = (h05, h10, -float(d_atom.mean()))
            if best is None or key > best[0]:
                d_q = min_dist(qa, atoms_orbit, orth).min(1)
                best = (key, dict(
                    hit05=h05, hit10=h10, ghost=int((d_q > 1.0).sum()), inv=inv,
                    shift=t.tolist(), meanNN=float(d_atom.mean()),
                    median_atom_to_Q=float(np.median(d_atom)),
                ))
    return best[1]


def strength_order(peaks_csv: str | Path | None, ops: Ops, q: np.ndarray):
    """Order Q peaks strongest first, mapping each to a ``peaks.csv`` row via symmetry."""
    try:
        with open(peaks_csv) as fh:
            rows = list(csv.DictReader(fh))
        p = np.array([[float(r["x_frac"]), float(r["y_frac"]), float(r["z_frac"])] for r in rows])
        h = np.array([float(r["height"]) for r in rows])
    except (OSError, TypeError, KeyError, ValueError):
        return np.arange(len(q)), "res-order (no peaks.csv)"
    st = np.full(len(q), np.nan)
    for i, qi in enumerate(q):
        for j, pj in enumerate(p):
            for m, t in ops:
                for inv in (1, -1):
                    d = (m @ pj + t) * inv - qi
                    d -= np.round(d)
                    if np.abs(d).max() < 2e-3:
                        st[i] = h[j] if np.isnan(st[i]) else max(st[i], h[j])
    n_miss = int(np.isnan(st).sum())
    st[np.isnan(st)] = -1
    return np.argsort(-st, kind="stable"), f"peaks.csv height via symmetry map ({n_miss} unmapped)"


def verdict(rate05: float, ratio05: float, ghost_frac: float) -> str:
    if rate05 >= 0.5 and ratio05 >= 3 and ghost_frac <= 0.4:
        return "PASS"
    if 0.25 <= rate05 < 0.5 and ratio05 > 3:
        return "PARTIAL"
    return "FAIL"


def score_q_peaks(cid: str, cif: str | Path, res: str | Path, peaks_csv: str | Path | None,
                  n_random: int = DEFAULT_N_RANDOM, seed: int = DEFAULT_SEED) -> Dict:
    """Score one ``trial.res`` against the deposited CIF. Returns a JSON-able dict."""
    cell, hm, atoms = read_cif_nonh(cif)
    orth = orth_matrix(cell)
    ops = space_group_ops(hm)
    af = np.array([a[2] for a in atoms])
    n = len(af)
    a_orb = orbit(af, ops)
    q_all = read_res_q(res)
    order, how = strength_order(peaks_csv, ops, q_all)
    n_use = min(n, len(q_all))
    q = q_all[order][:n_use]
    real = evaluate(af, a_orb, q, ops, orth)

    rng = np.random.default_rng(seed)
    r05, r10, rg = [], [], []
    for _ in range(n_random):
        r = evaluate(af, a_orb, rng.random((n_use, 3)), ops, orth)
        r05.append(r["hit05"])
        r10.append(r["hit10"])
        rg.append(r["ghost"])
    r05, r10, rg = np.array(r05), np.array(r10), np.array(rg)

    out = dict(
        id=cid, sg=hm, cell=cell, N_nonH=n, n_Q_in_res=len(q_all), n_Q_used=n_use,
        strength=how, formula_atoms=sorted({a[1] for a in atoms}), real=real,
        rate05=real["hit05"] / n, rate10=real["hit10"] / n, ghost_frac=real["ghost"] / n_use,
        rand=dict(
            n=n_random, seed=seed, hit05_mean=float(r05.mean()), hit05_sd=float(r05.std()),
            hit05_max=int(r05.max()), hit05_p95=float(np.percentile(r05, 95)),
            hit10_mean=float(r10.mean()), hit10_sd=float(r10.std()),
            ghost_mean=float(rg.mean()), ghost_frac_mean=float(rg.mean() / n_use),
        ),
    )
    base05 = r05.mean() / n
    out["ratio05"] = out["rate05"] / base05 if base05 > 0 else float("inf")
    out["p_random_ge_real_hit05"] = float((r05 >= real["hit05"]).mean())
    out["verdict"] = verdict(out["rate05"], out["ratio05"], out["ghost_frac"])
    return out


def planted_selftest(cif: str | Path, seed: int = 1, jitter_sigma: float = 0.15) -> Dict:
    """Plant the deposited atoms as Q peaks and check the scorer finds them.

    Each atom is moved to a random symmetry image, then the whole set is
    inverted and shifted by (1/2, 0, 1/2). An exact plant must score N/N
    within 0.5 A; ``jitter`` adds ``jitter_sigma`` A Gaussian noise per axis.
    """
    cell, hm, atoms = read_cif_nonh(cif)
    ops = space_group_ops(hm)
    orth = orth_matrix(cell)
    af = np.array([a[2] for a in atoms])
    a_orb = orbit(af, ops)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(len(ops)) for _ in af]
    q = np.array([ops[k][0] @ a + ops[k][1] for k, a in zip(idx, af)])
    q = -q + np.array([0.5, 0.0, 0.5])
    exact = evaluate(af, a_orb, q, ops, orth)
    qj = q + rng.normal(0, jitter_sigma, q.shape) @ np.linalg.inv(orth).T
    jitter = evaluate(af, a_orb, qj, ops, orth)
    return dict(N_nonH=len(af), exact=exact, jitter=jitter, jitter_sigma=jitter_sigma)
