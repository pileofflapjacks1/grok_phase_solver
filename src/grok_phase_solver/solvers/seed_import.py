"""
Scientist-facing seed importers for the hard-data partial-φ path.

Convert common lab artifacts into phase seeds for ``partial_phaseed_solve``:

1. **Phase CSV** — h,k,l,phase_deg (MAD/MR/manual)
2. **SHELX .res / .ins atoms** — Fcalc from partial model (SHELXS Q-peaks, fragment)
3. **peaks.csv** — density peaks as light atoms → Fcalc
4. **atoms CSV** — x,y,z,element fractional coordinates
5. **Isomorphous pair** — difference Patterson → heavy-atom sites (HA seed)

Also: seed-quality diagnostics for report.md (truth-free).

v0.5: predicted-model / MR-lite helpers (OpenFold3 / Boltz / AF-style CIF),
space-group expansion of fragments, multi-seed phase combination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from grok_phase_solver.io.shelx_write import parse_shelx_res_atoms
from grok_phase_solver.solvers.direct_methods import normalize_E
from grok_phase_solver.solvers.partial_seed import (
    fragment_seed_phases,
    load_phase_seed_csv,
    write_phase_seed_csv,
)

PathLike = Union[str, Path]

# Common SFAC order when .res lacks readable SFAC mapping
_DEFAULT_SFAC = {1: "C", 2: "H", 3: "N", 4: "O", 5: "Cl", 6: "S", 7: "Br", 8: "I", 9: "P", 10: "F"}


def _label_to_element(label: str, sfac: int = 1) -> str:
    lab = (label or "C").strip()
    # Strip trailing digits: Br1 → Br, Q12 → C (Q peaks → carbon placeholder)
    i = 0
    while i < len(lab) and lab[i].isalpha():
        i += 1
    el = lab[:i] if i else ""
    if el.upper().startswith("Q") or el == "":
        return _DEFAULT_SFAC.get(int(sfac), "C")
    # Capitalize properly (BR → Br)
    if len(el) == 1:
        return el.upper()
    return el[0].upper() + el[1:].lower()


def load_atoms_from_res(
    path: PathLike,
    *,
    max_atoms: Optional[int] = None,
    skip_q: bool = False,
    q_as_element: str = "C",
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Parse SHELX .res/.ins atom list → fractional coords + element symbols.

    Q-peaks become ``q_as_element`` (default C) unless ``skip_q``.
    """
    path = Path(path)
    raw = parse_shelx_res_atoms(path)
    fracs: List[np.ndarray] = []
    els: List[str] = []
    for a in raw:
        lab = str(a["label"])
        is_q = lab.upper().startswith("Q")
        if is_q and skip_q:
            continue
        el = q_as_element if is_q else _label_to_element(lab, int(a.get("sfac", 1)))
        fracs.append(np.asarray(a["fract"], dtype=np.float64))
        els.append(el)
        if max_atoms is not None and len(fracs) >= max_atoms:
            break
    if not fracs:
        raise ValueError(f"No atoms parsed from {path}")
    meta = {
        "kind": "res_atoms",
        "path": str(path),
        "n_atoms": len(fracs),
        "n_raw": len(raw),
        "elements": els[:30],
    }
    return np.vstack(fracs), els, meta


def load_atoms_from_peaks_csv(
    path: PathLike,
    *,
    element: str = "C",
    max_atoms: Optional[int] = 20,
    min_sigma: float = 2.5,
) -> Tuple[np.ndarray, List[str], Dict]:
    """Load gps-solve / density ``peaks.csv`` as a fragment model."""
    path = Path(path)
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if raw.dtype.names is None:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # rank,x,y,z,height,height_sigma
        xyz = data[:, 1:4]
        sig = data[:, 5] if data.shape[1] > 5 else np.ones(len(data))
    else:
        names = [n.lower() for n in raw.dtype.names]

        def col(*cands):
            for c in cands:
                if c in names:
                    return np.asarray(raw[raw.dtype.names[names.index(c)]])
            raise KeyError(cands)

        xyz = np.column_stack(
            [col("x_frac", "x", "fract_x"), col("y_frac", "y", "fract_y"), col("z_frac", "z", "fract_z")]
        ).astype(np.float64)
        try:
            sig = col("height_sigma", "sigma", "height").astype(np.float64)
        except KeyError:
            sig = np.ones(len(xyz))

    keep = sig >= float(min_sigma)
    xyz = xyz[keep]
    sig = sig[keep]
    order = np.argsort(-sig)
    xyz = xyz[order]
    if max_atoms is not None:
        xyz = xyz[: int(max_atoms)]
    if len(xyz) == 0:
        raise ValueError(f"No peaks above {min_sigma}σ in {path}")
    els = [element] * len(xyz)
    meta = {
        "kind": "peaks_csv",
        "path": str(path),
        "n_atoms": len(xyz),
        "element": element,
        "min_sigma": float(min_sigma),
    }
    return np.asarray(xyz, dtype=np.float64), els, meta


def load_atoms_from_atoms_csv(path: PathLike) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Load explicit fragment: x,y,z,element  (fractional) or x_frac,y_frac,z_frac,element.
    """
    path = Path(path)
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if raw.dtype.names is None:
        data = np.loadtxt(path, delimiter=",", dtype=str)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        xyz = data[:, :3].astype(float)
        els = [str(e).strip() for e in data[:, 3]]
    else:
        names = [n.lower() for n in raw.dtype.names]

        def col(*cands):
            for c in cands:
                if c in names:
                    return np.asarray(raw[raw.dtype.names[names.index(c)]])
            raise KeyError(cands)

        xyz = np.column_stack(
            [
                col("x_frac", "x", "fract_x"),
                col("y_frac", "y", "fract_y"),
                col("z_frac", "z", "fract_z"),
            ]
        ).astype(np.float64)
        el_col = col("element", "el", "atom", "type")
        els = [str(e).strip() for e in el_col]
    if len(xyz) == 0:
        raise ValueError(f"No atoms in {path}")
    meta = {"kind": "atoms_csv", "path": str(path), "n_atoms": len(xyz), "elements": els[:30]}
    return xyz, els, meta


def seed_from_fragment_atoms(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    fracs: np.ndarray,
    elements: Sequence[str],
    *,
    b_iso: float = 8.0,
    fcalc_min_rel: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Partial atoms → Fcalc phases; mask = reflections with strong |Fcalc|.

    Returns (seed_phases_full, mask, meta).
    """
    ph, fcalc = fragment_seed_phases(hkl, fracs, elements, cell, b_iso=b_iso)
    thr = float(fcalc_min_rel) * float(fcalc.max() + 1e-16)
    mask = fcalc >= thr
    # Ensure at least a few seeds
    if mask.sum() < 8:
        k = min(max(8, int(0.15 * len(fcalc))), len(fcalc))
        top = np.argsort(-fcalc)[:k]
        mask = np.zeros(len(fcalc), dtype=bool)
        mask[top] = True
    rng = np.random.default_rng(seed)
    seed_ph = rng.uniform(-np.pi, np.pi, size=len(amplitudes))
    seed_ph[mask] = ph[mask]
    meta = {
        "kind": "fragment_fcalc",
        "n_atoms": len(np.asarray(fracs).reshape(-1, 3)),
        "n_seed_refl": int(mask.sum()),
        "fraction": float(mask.mean()),
        "mean_fcalc": float(np.mean(fcalc)),
        "fcalc_min_rel": float(fcalc_min_rel),
        "elements": list(elements)[:30],
    }
    return seed_ph, mask, meta


def ha_sites_from_difference_patterson(
    hkl: np.ndarray,
    F_native: np.ndarray,
    F_derivative: np.ndarray,
    cell: np.ndarray,
    *,
    n_ha: int = 1,
    n_peaks: int = 12,
) -> Tuple[np.ndarray, Dict]:
    """
    Crude HA placement from difference Patterson peaks.

    For a single heavy atom in P1, the strongest non-origin peak is ~2x
    (interatomic vector). We place sites at peak/2 (and origin-proximal
    multi-HA peaks as additional trial positions).

    Honest scope: bootstrap for hybrid tests / single-HA organics — not SHELXC/D.
    """
    from grok_phase_solver.solvers.difference_patterson import locate_heavy_atom_vectors

    peaks, P, info = locate_heavy_atom_vectors(
        hkl, F_native, F_derivative, cell, n_peaks=n_peaks
    )
    sites: List[np.ndarray] = []
    for p in peaks:
        # PattersonPeak has .fract or similar — check
        fr = getattr(p, "fract", None)
        if fr is None:
            fr = getattr(p, "position", None)
        if fr is None:
            continue
        fr = np.asarray(fr, dtype=np.float64) % 1.0
        # skip near-origin vectors
        dist = np.min([np.linalg.norm(fr), np.linalg.norm(fr - 1.0)])
        if dist < 0.05:
            continue
        sites.append((0.5 * fr) % 1.0)
        if len(sites) >= n_ha:
            break
    if not sites:
        # fallback: map maximum off origin
        sites.append(np.array([0.25, 0.25, 0.25]))
    meta = {
        "kind": "ha_diff_patterson",
        "n_ha": len(sites),
        "n_patterson_peaks": len(peaks),
        "map_max": info.get("map_max"),
        "note": "peak/2 heuristic; single-HA oriented; not full SHELXD substructure",
    }
    return np.vstack(sites), meta


def ha_seed_from_isomorphous_pair(
    hkl: np.ndarray,
    F_native: np.ndarray,
    F_derivative: np.ndarray,
    cell: np.ndarray,
    *,
    n_ha: int = 1,
    ha_element: str = "Br",
    b_iso: float = 5.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Difference Patterson → HA sites → Fcalc seed phases."""
    sites, pmeta = ha_sites_from_difference_patterson(
        hkl, F_native, F_derivative, cell, n_ha=n_ha
    )
    els = [ha_element] * len(sites)
    seed_ph, mask, meta = seed_from_fragment_atoms(
        hkl, F_derivative, cell, sites, els, b_iso=b_iso, seed=seed
    )
    meta.update(pmeta)
    meta["ha_element"] = ha_element
    return seed_ph, mask, meta


def patterson_heavy_seed(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    n_ha: int = 2,
    ha_element: str = "Br",
    b_iso: float = 5.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Single-dataset heuristic: strongest Patterson peaks → trial HA sites at peak/2.

    Useful when a heavy atom is present but no derivative pair is available.
    Much weaker than true SIR/SAD — labeled as such in meta.
    """
    from grok_phase_solver.physics.patterson import find_patterson_peaks, patterson_from_amplitudes
    from grok_phase_solver.physics.reciprocal import d_spacing

    d_min = float(np.min(d_spacing(hkl, cell)))
    P = patterson_from_amplitudes(
        hkl, amplitudes, cell, d_min=d_min, remove_origin=True, origin_sharpen=5.0
    )
    peaks = find_patterson_peaks(P, n_peaks=max(8, n_ha * 4))
    sites: List[np.ndarray] = []
    for p in peaks:
        fr = np.asarray(getattr(p, "fract", getattr(p, "position", [0, 0, 0])), dtype=np.float64) % 1.0
        dist = np.min([np.linalg.norm(fr), np.linalg.norm((fr - 0.5))])
        if np.linalg.norm(np.minimum(fr, 1 - fr)) < 0.08:
            continue
        sites.append((0.5 * fr) % 1.0)
        if len(sites) >= n_ha:
            break
    if not sites:
        sites = [np.array([0.1, 0.2, 0.3])]
    els = [ha_element] * len(sites)
    seed_ph, mask, meta = seed_from_fragment_atoms(
        hkl, amplitudes, cell, np.vstack(sites), els, b_iso=b_iso, seed=seed
    )
    meta.update(
        {
            "kind": "patterson_heavy_heuristic",
            "n_ha": len(sites),
            "note": "Single-dataset Patterson→HA/2; weak vs true SIR/SAD",
        }
    )
    return seed_ph, mask, meta


def resolve_phase_seed(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    phase_seed_csv: Optional[str] = None,
    phase_seed_res: Optional[str] = None,
    seed_atoms_csv: Optional[str] = None,
    seed_peaks_csv: Optional[str] = None,
    seed_element: str = "C",
    seed_n_atoms: Optional[int] = None,
    seed_min_peak_sigma: float = 2.5,
    seed_b_iso: float = 8.0,
    native_amp: Optional[np.ndarray] = None,
    derivative_amp: Optional[np.ndarray] = None,
    n_ha: int = 1,
    ha_element: str = "Br",
    use_patterson_ha: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Resolve any supported seed source → (seed_phases, mask, meta).

    Priority: phase CSV → res → atoms CSV → peaks CSV → isomorphous HA → Patterson HA.
    """
    if phase_seed_csv:
        seed_ph, mask, meta = load_phase_seed_csv(phase_seed_csv, hkl)
        meta["source"] = "phase_seed_csv"
        return seed_ph, mask, meta

    if phase_seed_res:
        fracs, els, ameta = load_atoms_from_res(
            phase_seed_res, max_atoms=seed_n_atoms
        )
        seed_ph, mask, meta = seed_from_fragment_atoms(
            hkl, amplitudes, cell, fracs, els, b_iso=seed_b_iso, seed=seed
        )
        meta.update(ameta)
        meta["source"] = "phase_seed_res"
        return seed_ph, mask, meta

    if seed_atoms_csv:
        fracs, els, ameta = load_atoms_from_atoms_csv(seed_atoms_csv)
        if seed_n_atoms is not None:
            fracs, els = fracs[:seed_n_atoms], els[:seed_n_atoms]
        seed_ph, mask, meta = seed_from_fragment_atoms(
            hkl, amplitudes, cell, fracs, els, b_iso=seed_b_iso, seed=seed
        )
        meta.update(ameta)
        meta["source"] = "seed_atoms_csv"
        return seed_ph, mask, meta

    if seed_peaks_csv:
        fracs, els, ameta = load_atoms_from_peaks_csv(
            seed_peaks_csv,
            element=seed_element,
            max_atoms=seed_n_atoms or 20,
            min_sigma=seed_min_peak_sigma,
        )
        seed_ph, mask, meta = seed_from_fragment_atoms(
            hkl, amplitudes, cell, fracs, els, b_iso=seed_b_iso, seed=seed
        )
        meta.update(ameta)
        meta["source"] = "seed_peaks_csv"
        return seed_ph, mask, meta

    if native_amp is not None and derivative_amp is not None:
        seed_ph, mask, meta = ha_seed_from_isomorphous_pair(
            hkl,
            native_amp,
            derivative_amp,
            cell,
            n_ha=n_ha,
            ha_element=ha_element,
            seed=seed,
        )
        meta["source"] = "isomorphous_ha"
        return seed_ph, mask, meta

    if use_patterson_ha:
        seed_ph, mask, meta = patterson_heavy_seed(
            hkl, amplitudes, cell, n_ha=n_ha, ha_element=ha_element, seed=seed
        )
        meta["source"] = "patterson_ha"
        return seed_ph, mask, meta

    raise ValueError(
        "No phase seed source provided. Use one of:\n"
        "  --phase-seed-csv  (h,k,l,phase_deg)\n"
        "  --phase-seed-res  (SHELXS/SHELXL .res atoms → Fcalc)\n"
        "  --seed-atoms-csv  (x,y,z,element)\n"
        "  --seed-peaks-csv  (peaks.csv from a prior gps-solve)\n"
        "  --native-hkl + --derivative-hkl  (isomorphous HA)\n"
        "  --patterson-ha    (single-dataset HA heuristic)"
    )


def load_predicted_model_atoms(
    path: PathLike,
    *,
    max_atoms: Optional[int] = None,
    min_occupancy: float = 0.3,
    expand_symmetry: bool = True,
    skip_hydrogen: bool = True,
    space_group: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Load a predicted / experimental model (CIF) as a fragment for Fcalc seeding.

    Designed for AF / OpenFold3 / Boltz-2 / RF / experimental partial models.
    Filters low occupancy, optional H skip, optional SG expansion of ASU.

    Returns fracs (N,3), elements, meta.
    """
    from grok_phase_solver.io.cif import load_cif

    path = Path(path)
    st = load_cif(str(path))
    fracs: List[np.ndarray] = []
    els: List[str] = []
    occs: List[float] = []
    for a in st.atoms:
        el = (a.element or "C").strip()
        if skip_hydrogen and el.upper() in ("H", "D"):
            continue
        occ = float(getattr(a, "occupancy", 1.0) or 1.0)
        if occ < min_occupancy:
            continue
        fracs.append(np.asarray(a.fract, dtype=np.float64))
        els.append(el)
        occs.append(occ)
        if max_atoms is not None and len(fracs) >= max_atoms:
            break
    if not fracs:
        raise ValueError(f"No usable atoms from predicted model {path}")

    fr = np.vstack(fracs)
    el_list = list(els)
    expand_meta: Dict = {"expanded": False}
    sg = space_group or getattr(st, "space_group_hm", None)
    if expand_symmetry and sg:
        try:
            from grok_phase_solver.physics.symmetry import expand_fractional_coords

            fr2, el2, expand_meta = expand_fractional_coords(
                fr, sg, elements=el_list
            )
            # Cap expansion explosion for huge Z
            if max_atoms is not None and len(fr2) > max_atoms * 4:
                # keep heaviest elements first
                order = np.argsort([-_element_weight(e) for e in el2])
                fr2 = fr2[order[: max_atoms * 4]]
                el2 = [el2[i] for i in order[: max_atoms * 4]]
            fr, el_list = fr2, el2
        except Exception as e:
            expand_meta = {"expanded": False, "error": str(e)}

    meta = {
        "kind": "predicted_model",
        "path": str(path),
        "n_atoms": len(fr),
        "n_asu_kept": len(fracs),
        "space_group": sg,
        "mean_occupancy": float(np.mean(occs)) if occs else 1.0,
        "elements": el_list[:40],
        "expand": expand_meta,
        "source_formats": "cif (AF/OpenFold3/Boltz/experimental style)",
        "note": (
            "MR-lite fragment seed via Fcalc phases — not full molecular replacement. "
            "Physics fallback: same partial_phaseed path as SHELXS fragments."
        ),
    }
    return fr, el_list, meta


def _element_weight(el: str) -> float:
    """Rough atomic number proxy for fragment prioritization."""
    table = {
        "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
        "CL": 17, "Cl": 17, "BR": 35, "Br": 35, "I": 53, "FE": 26, "Fe": 26,
        "ZN": 30, "Zn": 30, "MG": 12, "Mg": 12, "CA": 20, "Ca": 20,
    }
    return float(table.get(el, table.get(el.upper(), 6)))


def seed_from_predicted_model(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    model_path: PathLike,
    *,
    max_atoms: Optional[int] = None,
    b_iso: float = 12.0,
    expand_symmetry: bool = True,
    space_group: Optional[str] = None,
    seed: int = 0,
    fcalc_min_rel: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Predicted model CIF → Fcalc phase seed (+ mask)."""
    fracs, els, mmeta = load_predicted_model_atoms(
        model_path,
        max_atoms=max_atoms,
        expand_symmetry=expand_symmetry,
        space_group=space_group,
    )
    seed_ph, mask, meta = seed_from_fragment_atoms(
        hkl,
        amplitudes,
        cell,
        fracs,
        els,
        b_iso=b_iso,
        seed=seed,
        fcalc_min_rel=fcalc_min_rel,
    )
    meta.update(mmeta)
    meta["source"] = "predicted_model"
    return seed_ph, mask, meta


def combine_phase_seeds(
    phase_sets: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    weights: Optional[Sequence[float]] = None,
    amplitudes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Combine multiple partial phase seeds (circular weighted mean).

    Physics MR-lite style: when several fragments / HA / phase CSVs exist,
    average on the unit circle where masks overlap; union of masks elsewhere.

    Returns combined_phases, combined_mask, meta.
    """
    if not phase_sets:
        raise ValueError("no phase sets")
    arr = [np.asarray(p, dtype=np.float64) for p in phase_sets]
    msk = [np.asarray(m, dtype=bool) for m in masks]
    n = len(arr[0])
    for a, m in zip(arr, msk):
        if len(a) != n or len(m) != n:
            raise ValueError("phase/mask length mismatch")
    if weights is None:
        w = np.ones(len(arr), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / (np.sum(w) + 1e-16)

    out = np.zeros(n, dtype=np.float64)
    comb_mask = np.zeros(n, dtype=bool)
    n_sources = np.zeros(n, dtype=np.int32)
    for i in range(n):
        zs = []
        ws = []
        for s, (ph, m) in enumerate(zip(arr, msk)):
            if m[i]:
                zs.append(np.exp(1j * ph[i]))
                ws.append(w[s])
                n_sources[i] += 1
        if zs:
            ww = np.asarray(ws, dtype=np.float64)
            ww = ww / (np.sum(ww) + 1e-16)
            z = np.sum(ww * np.asarray(zs))
            out[i] = float(np.angle(z))
            comb_mask[i] = True
    meta = {
        "kind": "combined_seeds",
        "n_sets": len(arr),
        "n_seed_refl": int(comb_mask.sum()),
        "mean_sources_per_seeded": float(n_sources[comb_mask].mean())
        if comb_mask.any()
        else 0.0,
        "weights": w.tolist(),
    }
    if amplitudes is not None:
        meta["amp_weighted_coverage"] = float(
            np.sum(np.asarray(amplitudes)[comb_mask])
            / (np.sum(amplitudes) + 1e-16)
        )
    return out, comb_mask, meta


def assess_seed_quality(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    seed_phases: np.ndarray,
    mask: np.ndarray,
    *,
    within_deg: float = 20.0,
    bar_fraction: float = 0.30,
) -> Dict:
    """
    Truth-free seed diagnostics for report.md.

    Estimates whether the seed set is large enough vs the 30% strong-|E| bar
    (size only — correctness unknown without truth).
    """
    from grok_phase_solver.solvers.free_fom import free_fom
    from grok_phase_solver.physics.density import density_from_structure_factors

    mask = np.asarray(mask, dtype=bool)
    n = len(amplitudes)
    n_seed = int(mask.sum())
    E = normalize_E(hkl, amplitudes, cell)
    # strong set = top 30% |E|
    n_strong = max(1, int(round(0.30 * n)))
    strong_idx = np.argsort(-E)[:n_strong]
    strong_mask = np.zeros(n, dtype=bool)
    strong_mask[strong_idx] = True
    n_strong_seeded = int((mask & strong_mask).sum())
    frac_strong_seeded = n_strong_seeded / max(n_strong, 1)
    size_ok = frac_strong_seeded >= bar_fraction - 1e-9 or (n_seed / max(n, 1)) >= bar_fraction

    fom = None
    try:
        rho = density_from_structure_factors(
            hkl, amplitudes * np.exp(1j * seed_phases), cell
        )
        fom = free_fom(hkl, amplitudes, seed_phases, cell, density=rho)
    except Exception:
        pass

    hints: List[str] = []
    if n_seed < 5:
        hints.append("Very few reflections mapped — check seed file / indices.")
    if not size_ok:
        hints.append(
            f"Seed covers only {frac_strong_seeded:.0%} of the strong-|E| set "
            f"(oracle bar ≈ {bar_fraction:.0%} of strong phases correct within "
            f"{within_deg:.0f}°). Add more known φ / heavier fragment / HA sites."
        )
    else:
        hints.append(
            f"Seed size looks adequate vs the {bar_fraction:.0%} strong-|E| bar "
            f"({n_strong_seeded}/{n_strong} strong reflections seeded). "
            "Correctness still unknown (truth-free)."
        )
    if fom is not None and fom.get("composite", 0) < 0.35:
        hints.append(
            "Free FOM of the raw seed is low — extension may struggle; "
            "verify phases/fragment or try shelxs+shelxe."
        )

    return {
        "n_seed": n_seed,
        "fraction_all": n_seed / max(n, 1),
        "n_strong": n_strong,
        "n_strong_seeded": n_strong_seeded,
        "frac_strong_seeded": frac_strong_seeded,
        "size_meets_bar": bool(size_ok),
        "bar_fraction": bar_fraction,
        "within_deg": within_deg,
        "seed_free_fom_composite": None if fom is None else float(fom["composite"]),
        "seed_free_fom_R_pos": None if fom is None else float(fom.get("R_pos", float("nan"))),
        "hints": hints,
    }


def export_seed_csv(
    path: PathLike,
    hkl: np.ndarray,
    seed_phases: np.ndarray,
    mask: np.ndarray,
) -> Path:
    """Write mapped seed phases for reuse / inspection."""
    return write_phase_seed_csv(path, hkl, seed_phases, mask=mask, phase_unit="deg")
