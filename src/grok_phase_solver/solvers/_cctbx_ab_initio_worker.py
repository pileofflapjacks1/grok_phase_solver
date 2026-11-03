"""cctbx/smtbx worker for ab_initio (runs under Olex2 Python 3.8 or any cctbx).

Expects REQUEST_JSON / RESULT_JSON paths injected before exec, or argv:
  python _cctbx_ab_initio_worker.py request.json result.json

Does not import grok_phase_solver. Returns all candidates with peaks;
host ranks by method-independent E-map check (no model, not CF own FOM).
"""
from __future__ import division, print_function

import json
import math
import os
import sys
import time
import traceback


def _combined_fom(consistency, skew):
    sk = max(float(skew), 0.0)
    cons = float(consistency)
    return cons * (0.5 + 0.5 * math.tanh(sk / 2.0)) + 0.05 * sk


def _map_skewness(emap):
    """Skewness of a miller.fft_map (or compatible). Avoid double-access asserts."""
    try:
        # Prefer cctbx statistics when available
        st = emap.statistics()
        if hasattr(st, "skewness"):
            return float(st.skewness())
    except Exception:
        pass
    from cctbx.array_family import flex
    try:
        vals = emap.real_map_unpadded(in_place=False).as_1d()
    except TypeError:
        vals = emap.real_map_unpadded().as_1d()
    except Exception:
        # Last-resort: use already-accessed real_map if present
        try:
            vals = emap.real_map().as_1d()
        except Exception:
            return 0.0
    mean = flex.mean(vals)
    sig = float(emap.statistics().sigma()) if hasattr(emap, "statistics") else float(flex.mean_sq(vals - mean) ** 0.5)
    if sig <= 0:
        return 0.0
    m3 = flex.mean(flex.pow(vals - mean, 3))
    return float(m3 / (sig ** 3))


def _tangent_consistency(triplets, amplitudes, phases):
    from cctbx.array_family import flex
    ph2 = triplets.apply_tangent_formula(
        amplitudes=amplitudes, phases_rad=phases,
        selection_fixed=None, use_fixed_only=False, reuse_results=False)
    return float(flex.mean(flex.cos(ph2 - phases)))


def _build_miller(cell, sg_str, hkl, amplitudes):
    from cctbx import crystal, miller
    from cctbx.array_family import flex
    sym = crystal.symmetry(unit_cell=tuple(cell), space_group_symbol=str(sg_str))
    indices = flex.miller_index()
    data = flex.double()
    for (h, k, l), a in zip(hkl, amplitudes):
        if a is None or a <= 0:
            continue
        indices.append((int(h), int(k), int(l)))
        data.append(float(a))
    ms = miller.set(crystal_symmetry=sym, indices=indices, anomalous_flag=False)
    ma = miller.array(ms, data=data).set_observation_type_xray_amplitude()
    # merge equivalents for CF / DM stability
    ma = ma.merge_equivalents().array()
    return ma


def _peak_list(emap, n_peaks, min_dist):
    from cctbx import maptbx
    # Match prototype + Olex2 CF peak search style
    try:
        emap.apply_sigma_scaling()
    except Exception:
        pass
    params = maptbx.peak_search_parameters(
        interpolate=True,
        min_distance_sym_equiv=float(min_dist),
        max_clusters=int(max(n_peaks, 1)),
    )
    try:
        ps = emap.peak_search(parameters=params, verify_symmetry=False)
    except TypeError:
        ps = emap.peak_search(parameters=params)
    peaks = ps.all(max_clusters=int(max(n_peaks, 1)))
    sites = [list(s) for s in peaks.sites()]
    heights = [float(h) for h in peaks.heights()]
    return sites, heights


def _phases_on_input(ma_phased, input_hkl):
    """Map phases from a miller array onto the original input index list."""
    from cctbx.array_family import flex
    lookup = {}
    for i in range(ma_phased.size()):
        hkl = ma_phased.indices()[i]
        lookup[(int(hkl[0]), int(hkl[1]), int(hkl[2]))] = float(ma_phased.data()[i])
    # If complex, use arg; if already phases stored separately use that path
    out = []
    # Prefer phases from phase_transfer source if amplitudes-only: caller passes
    # a complex array via .data() as complex — handle both.
    return lookup  # unused fallback


def run(req):
    from cctbx import dmtbx, maptbx, miller
    from cctbx.array_family import flex
    from scitbx.random import set_random_seed
    from smtbx.ab_initio import charge_flipping
    from libtbx import group_args

    t0 = time.time()
    cell = req["cell"]
    sg = req.get("space_group") or "P 1"
    n_nonH = int(req.get("n_nonH") or 20)
    seed = int(req.get("seed") or 20251102)
    budget = float(req.get("time_budget_s") or 60.0)
    e_min = float(req.get("e_min") or 1.2)
    e_cap = int(req.get("e_cap") or max(50, 10 * n_nonH))
    n_tan = int(req.get("n_tangent_cycles") or 12)
    min_dist = float(req.get("min_peak_dist_A") or 1.2)
    sigma2 = bool(req.get("sigma2_only", True))
    cf_params = req.get("cf") or {}

    ma = _build_miller(cell, sg, req["hkl"], req["amplitudes"])
    is_centric = ma.space_group().is_centric()
    input_indices = [(int(h), int(k), int(l)) for h, k, l in req["hkl"]]

    candidates = []
    tangent_meta = {"error": None}
    cf_meta = {"error": None}

    # ------------------------------------------------------------------ tangent
    try:
        set_random_seed(seed)
        ma.setup_binner(auto_binning=True)
        all_e = ma.quasi_normalize_structure_factors().sort(by_value="data")
        large = all_e.select(all_e.data() > e_min)
        if large.size() > e_cap:
            large = large[:e_cap]
        triplets = dmtbx.triplet_generator(
            large, amplitudes=large.data(), sigma_2_only=sigma2)
        nrel = triplets.n_relations()
        amps = large.data()
        cf_reserve = float(req.get("cf_time_reserve_s") or 12.0)
        loop_budget = max(5.0, budget - cf_reserve)
        t_loop0 = time.time()
        n_trials = 0
        best = None
        while (time.time() - t_loop0) < loop_budget:
            ph = large.random_phases_compatible_with_phase_restrictions().data()
            for _ in range(n_tan):
                ph = triplets.apply_tangent_formula(
                    amplitudes=amps, phases_rad=ph,
                    selection_fixed=None, use_fixed_only=False,
                    reuse_results=True)
            cons = _tangent_consistency(triplets, amps, ph)
            emap = large.phase_transfer(phase_source=ph).fft_map(
                symmetry_flags=maptbx.use_space_group_symmetry)
            emap.apply_sigma_scaling()
            skew = _map_skewness(emap)
            fom = _combined_fom(cons, skew)
            if best is None or fom > best["fom"]:
                best = {
                    "fom": fom, "cons": cons, "skew": skew, "phases": ph,
                    "i": n_trials, "emap_max": float(emap.statistics().max()),
                }
            n_trials += 1
        if best is None:
            raise RuntimeError("no tangent trials completed")
        # peaks from best
        emap = large.phase_transfer(phase_source=best["phases"]).fft_map(
            symmetry_flags=maptbx.use_space_group_symmetry)
        emap.apply_sigma_scaling()
        sites, heights = _peak_list(emap, n_nonH, min_dist)
        # phases on input HKL via phase_transfer of Fo with interpolated phases
        phased = large.phase_transfer(phase_source=best["phases"])
        phase_lookup = {}
        pdata = phased.data()
        for i in range(phased.size()):
            idx = phased.indices()[i]
            z = pdata[i]
            phase_lookup[(int(idx[0]), int(idx[1]), int(idx[2]))] = math.atan2(
                z.imag, z.real)
        phases_in = [phase_lookup.get(tuple(ix), 0.0) for ix in input_indices]
        cand = {
            "name": "tangent",
            "fom": best["fom"],
            "cons": best["cons"],
            "skew": best["skew"],
            "peaks": [{"fract": s, "height": h} for s, h in zip(sites, heights)],
            "phases_rad": phases_in,
            "n_trials": n_trials,
            "n_e_used": int(large.size()),
            "n_triplets": int(flex.sum(nrel)),
            "best_trial": best["i"],
        }
        candidates.append(cand)
        tangent_meta = {
            "n_trials": n_trials,
            "n_e_used": int(large.size()),
            "n_triplets": int(flex.sum(nrel)),
            "fom": best["fom"],
            "loop_time_s": round(time.time() - t_loop0, 3),
        }
    except Exception as e:
        tangent_meta = {"error": "%s: %s" % (type(e).__name__, e),
                        "traceback": traceback.format_exc()}

    # ----------------------------------------------------------- charge flipping
    # Multi-seed CF: fixed documented seed list, within remaining wall time.
    cf_seed_list = req.get("cf_seed_list") or [0, 1, 7, 42, 2025]
    cf_runs = []
    cf_deadline = t0 + budget  # hard stop inside overall budget
    try:
        for cf_seed in cf_seed_list:
            if time.time() >= cf_deadline:
                cf_runs.append({"cf_seed": cf_seed, "skipped": True,
                                "reason": "budget_exhausted"})
                continue
            try:
                set_random_seed(int(seed) + 17 + int(cf_seed))
                flipping = charge_flipping.weak_reflection_improved_iterator(
                    delta=None)
                extra = group_args(
                    max_attempts_to_get_phase_transition=int(
                        cf_params.get("max_attempts_to_get_phase_transition", 5)),
                    max_attempts_to_get_sharp_correlation_map=int(
                        cf_params.get(
                            "max_attempts_to_get_sharp_correlation_map", 5)),
                    max_solving_iterations=int(
                        cf_params.get("max_solving_iterations", 500)),
                    yield_solving_interval=int(
                        cf_params.get("flipping_interval", 60)),
                    yield_during_delta_guessing=True,
                )
                amp_type = cf_params.get("amplitude_type", "quasi-E")
                if amp_type == "quasi-E":
                    extra.normalisations_for = (
                        charge_flipping.amplitude_quasi_normalisations)
                solving = charge_flipping.solving_iterator(
                    flipping, ma, **extra.__dict__)
                charge_flipping.loop(solving, verbose=False)
                had = bool(getattr(solving, "had_phase_transition", False))
                note = None
                if solving.f_calc_solutions:
                    f_calc, shift, cc_peak_height = solving.f_calc_solutions[0]
                    fft_map = f_calc.fft_map(
                        symmetry_flags=maptbx.use_space_group_symmetry)
                    try:
                        fft_map.apply_volume_scaling()
                    except Exception:
                        pass
                    try:
                        fft_map.apply_sigma_scaling()
                    except Exception:
                        pass
                    skew = _map_skewness(fft_map)
                    fom = _combined_fom(1.0, skew)
                    sites, heights = _peak_list(fft_map, n_nonH, min_dist)
                    phase_lookup = {}
                    pdata = f_calc.data()
                    for i in range(f_calc.size()):
                        idx = f_calc.indices()[i]
                        z = pdata[i]
                        phase_lookup[(int(idx[0]), int(idx[1]), int(idx[2]))] = (
                            math.atan2(z.imag, z.real))
                    phases_in = [
                        phase_lookup.get(tuple(ix), 0.0) for ix in input_indices
                    ]
                    n_sol = len(solving.f_calc_solutions)
                    cc_val = (
                        float(cc_peak_height)
                        if cc_peak_height is not None else None)
                else:
                    fl = solving.flipping_iterator
                    fft_map = fl.rho_map
                    skew = _map_skewness(fft_map)
                    fom = _combined_fom(1.0, skew)
                    sites, heights = _peak_list(fft_map, n_nonH, min_dist)
                    phases_in = [0.0] * len(input_indices)
                    n_sol = 0
                    cc_val = None
                    note = "no f_calc_solutions; used last flipping map"
                cand = {
                    "name": "charge_flip_seed_%s" % cf_seed,
                    "method": "charge_flip",
                    "cf_seed": int(cf_seed),
                    "fom": fom,
                    "cons": 1.0,
                    "skew": skew,
                    "cc_peak_height": cc_val,
                    "had_phase_transition": had,
                    "peaks": [
                        {"fract": s, "height": h}
                        for s, h in zip(sites, heights)
                    ],
                    "phases_rad": phases_in,
                    "n_solutions": n_sol,
                }
                if note:
                    cand["note"] = note
                candidates.append(cand)
                cf_runs.append({
                    "cf_seed": int(cf_seed),
                    "fom": fom,
                    "skew": skew,
                    "had_phase_transition": had,
                    "n_solutions": n_sol,
                    "n_peaks": len(sites),
                    "cc_peak_height": cc_val,
                })
            except Exception as e:
                cf_runs.append({
                    "cf_seed": int(cf_seed),
                    "error": "%s: %s" % (type(e).__name__, e),
                    "traceback": traceback.format_exc(),
                })
        cf_meta = {
            "n_seeds_requested": len(cf_seed_list),
            "n_seeds_ok": sum(1 for r in cf_runs if "fom" in r),
            "runs": cf_runs,
        }
    except Exception as e:
        cf_meta = {"error": "%s: %s" % (type(e).__name__, e),
                   "traceback": traceback.format_exc(),
                   "runs": cf_runs}

    # Mark tangent method field
    for c in candidates:
        if c.get("name") == "tangent":
            c["method"] = "tangent"

    if not candidates:
        return {
            "error": "both tangent and charge_flip failed",
            "tangent": tangent_meta,
            "charge_flip": cf_meta,
            "solver_time_s": round(time.time() - t0, 3),
        }

    # Diagnostic only: own-FOM "winner" (host re-selects by E-map check)
    fom_winner = max(candidates, key=lambda c: float(c.get("fom", float("-inf"))))
    return {
        "error": None,
        "winner": fom_winner["name"],  # diagnostic; host overrides via emap_check
        "fom": fom_winner["fom"],
        "peaks": fom_winner.get("peaks") or [],
        "phases_rad": fom_winner.get("phases_rad"),
        "candidates_full": candidates,
        "candidates": [
            {k: v for k, v in c.items() if k not in ("phases_rad", "peaks")}
            for c in candidates
        ],
        "n_nonH": n_nonH,
        "sg": str(ma.space_group_info()),
        "is_centrosymmetric": bool(is_centric),
        "n_trials": tangent_meta.get("n_trials"),
        "n_e_used": tangent_meta.get("n_e_used"),
        "n_triplets": tangent_meta.get("n_triplets"),
        "tangent": tangent_meta,
        "charge_flip": cf_meta,
        "cf_seed_list": list(cf_seed_list),
        "solver_time_s": round(time.time() - t0, 3),
    }


def main():
    req_path = globals().get("REQUEST_JSON")
    out_path = globals().get("RESULT_JSON")
    if req_path is None and len(sys.argv) >= 3:
        req_path, out_path = sys.argv[1], sys.argv[2]
    if not req_path or not out_path:
        raise SystemExit("need REQUEST_JSON and RESULT_JSON")
    with open(req_path, "r") as fh:
        req = json.load(fh)
    try:
        result = run(req)
    except Exception as e:
        result = {
            "error": "%s: %s" % (type(e).__name__, e),
            "traceback": traceback.format_exc(),
        }
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
else:
    if globals().get("REQUEST_JSON"):
        main()
