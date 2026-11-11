"""cctbx/smtbx worker for ab_initio (runs under Olex2 Python 3.8 or any cctbx).

Expects REQUEST_JSON / RESULT_JSON paths injected before exec, or argv:
  python _cctbx_ab_initio_worker.py request.json result.json

Does not import grok_phase_solver. Returns all candidates with peaks;
host ranks by method-independent E-map check (centrosymmetric) or by the
refined fit below (non-centrosymmetric); no model, not CF own FOM.

Checkpoints: after the tangent block and after every CF seed the full result
(``"partial": true``) is atomically written to ``req["checkpoint_path"]`` (the
same file as RESULT_JSON), so a host that kills this process at the hard
wall-clock limit still has every candidate finished so far.

Non-centrosymmetric only: after phasing, each candidate's top-N peaks
(N = N_nonH) are taken as isotropic carbon (U = 0.05 A^2) and refined for a few
least-squares cycles against Fo^2 (smtbx, unit weights, scale factor optimised
out); wR2 / R1 of the refined model are returned as ``refined_wR2`` /
``refined_R1``. The model never enters.
"""
from __future__ import division, print_function

import json
import math
import os
import sys
import time
import traceback

# Fixed (not tuned).
REFINE_CYCLES = 4
REFINE_U_ISO = 0.05
# Seconds of the budget kept for the refinement stage (non-centrosymmetric
# only); CF stops this much earlier so refinement still fits in time_budget_s.
REFINE_TIME_RESERVE_S = 3.0


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


def _write_checkpoint(path, result):
    """Atomically write *result* as JSON (tmp file + rename)."""
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _refine_candidate(fo_sq_obs, symmetry, peaks, n_nonH, n_cycles):
    """Refine top-N peaks (isotropic C) against Fo^2; return (wR2, R1).

    smtbx least squares, unit weights (no sigmas are carried into ab_initio),
    overall scale optimised out. Same procedure for every candidate.
    """
    from cctbx import crystal, xray
    from scitbx.lstbx import normal_eqns_solving
    from smtbx.refinement import constraints, least_squares
    import smtbx.utils

    xs = xray.structure(crystal.special_position_settings(symmetry))
    for i, p in enumerate(peaks[: int(n_nonH)]):
        sc = xray.scatterer(
            "C%d" % (i + 1), tuple(float(x) for x in p["fract"]),
            u=REFINE_U_ISO, scattering_type="C")
        sc.flags.set_use_u_iso(True).set_use_u_aniso(False)
        sc.flags.set_grad_site(True).set_grad_u_iso(True)
        xs.add_scatterer(sc)
    if xs.scatterers().size() == 0:
        raise ValueError("no peaks to refine")
    reparam = constraints.reparametrisation(
        structure=xs, constraints=[],
        connectivity_table=smtbx.utils.connectivity_table(xs))
    ls = least_squares.crystallographic_ls(
        fo_sq_obs, reparam, weighting_scheme=least_squares.unit_weighting())
    normal_eqns_solving.levenberg_marquardt_iterations(
        ls, n_max_iterations=int(n_cycles), track_all=True)
    wr2 = float(ls.wR2(cutoff_factor=-1e30))
    r1 = float(ls.r1_factor()[0])
    if not (math.isfinite(wr2) and math.isfinite(r1)):
        raise ValueError("non-finite refinement statistics")
    return wr2, r1


def _refine_candidates(ma, candidates, n_nonH, deadline, n_cycles=None):
    """Fill refined_wR2 / refined_R1 on every candidate (in place).

    Stops starting new refinements after *deadline* (wall clock); candidates
    not refined keep ``refined_wR2 = None``. Returns a metadata dict.
    """
    from cctbx.array_family import flex

    n_cycles = int(REFINE_CYCLES if n_cycles is None else n_cycles)
    t_start = time.time()
    meta = {"n_cycles": n_cycles, "u_iso": REFINE_U_ISO,
            "weights": "unit", "target": "Fo^2", "n_refined": 0,
            "n_failed": 0, "n_skipped_deadline": 0, "errors": []}
    try:
        fo_sq = ma.customized_copy(
            data=ma.data() * ma.data(),
            sigmas=flex.double(ma.size(), 1.0),
        ).set_observation_type_xray_intensity()
        obs = fo_sq.as_xray_observations()
        symmetry = ma.crystal_symmetry()
    except Exception as e:
        meta["errors"].append("setup: %s: %s" % (type(e).__name__, e))
        meta["n_failed"] = len(candidates)
        meta["time_s"] = round(time.time() - t_start, 3)
        return meta
    for c in candidates:
        c["refined_wR2"] = None
        c["refined_R1"] = None
        if time.time() >= deadline:
            meta["n_skipped_deadline"] += 1
            continue
        try:
            wr2, r1 = _refine_candidate(
                obs, symmetry, c.get("peaks") or [], n_nonH, n_cycles)
            c["refined_wR2"] = wr2
            c["refined_R1"] = r1
            meta["n_refined"] += 1
        except Exception as e:
            c["refine_error"] = "%s: %s" % (type(e).__name__, e)
            meta["n_failed"] += 1
            meta["errors"].append(
                "%s: %s: %s" % (c.get("name"), type(e).__name__, e))
    meta["time_s"] = round(time.time() - t_start, 3)
    return meta


def _assemble_result(candidates, tangent_meta, cf_meta, cf_seed_list, n_nonH,
                     ma, is_centric, t0, partial, refine_meta=None):
    """Build the result dict (final, or a checkpoint when *partial*)."""
    if not candidates:
        return {
            "error": "both tangent and charge_flip failed",
            "tangent": tangent_meta,
            "charge_flip": cf_meta,
            "partial": bool(partial),
            "solver_time_s": round(time.time() - t0, 3),
        }
    # Diagnostic only: own-FOM "winner" (host re-selects)
    fom_winner = max(candidates, key=lambda c: float(c.get("fom", float("-inf"))))
    return {
        "error": None,
        "partial": bool(partial),
        "winner": fom_winner["name"],  # diagnostic; host overrides
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
        "n_trials": (tangent_meta or {}).get("n_trials"),
        "n_e_used": (tangent_meta or {}).get("n_e_used"),
        "n_triplets": (tangent_meta or {}).get("n_triplets"),
        "tangent": tangent_meta,
        "charge_flip": cf_meta,
        "refine": refine_meta,
        "cf_seed_list": list(cf_seed_list),
        "timed_out": bool((cf_meta or {}).get("timed_out")),
        "solver_time_s": round(time.time() - t0, 3),
    }


def _loop_cf_with_deadline(solving, deadline):
    """Iterate smtbx solving_iterator until finished or wall-clock deadline.

    Mirrors charge_flipping.loop: ``solving_iterator`` is iterable via
    ``__iter__`` (generator) but is *not* itself an iterator, so we must
    ``it = iter(solving)`` then ``next(it)``. Returns True if aborted because
    time ran out (best-so-far left on *solving*). Checks the deadline before
    each ``next`` so a new CF step is not started late.
    """
    it = iter(solving)
    timed_out = False
    try:
        while True:
            if time.time() >= deadline:
                timed_out = True
                break
            try:
                next(it)
            except StopIteration:
                timed_out = False
                break
            if solving.state is solving.finished:
                timed_out = False
                break
    finally:
        # Match for-break: close the generator so clean_up() runs.
        try:
            it.close()
        except Exception:
            pass
    return timed_out


def _cf_deadline_probe(req):
    """Short real smtbx CF pass through _loop_cf_with_deadline (test hook)."""
    from scitbx.random import set_random_seed
    from smtbx.ab_initio import charge_flipping

    t0 = time.time()
    cell = req["cell"]
    sg = req.get("space_group") or "P -1"
    seed = int(req.get("seed") or 20251102)
    budget = float(req.get("time_budget_s") or 20.0)
    set_random_seed(seed)
    ma = _build_miller(cell, sg, req["hkl"], req["amplitudes"])
    # Match production defaults when requested; default to quasi-E.
    amp_type = req.get("amplitude_type", "quasi-E")
    kw = dict(
        max_attempts_to_get_phase_transition=1,
        max_attempts_to_get_sharp_correlation_map=1,
        max_solving_iterations=int(req.get("max_solving_iterations") or 40),
        yield_solving_interval=int(req.get("yield_solving_interval") or 5),
        yield_during_delta_guessing=True,
    )
    if amp_type == "quasi-E":
        kw["normalisations_for"] = charge_flipping.amplitude_quasi_normalisations

    # Regression check: solving_iterator is iterable but not an iterator.
    flipping = charge_flipping.weak_reflection_improved_iterator(delta=None)
    solving = charge_flipping.solving_iterator(flipping, ma, **kw)
    try:
        iter(solving)  # must not fail
    except TypeError as e:
        return {
            "error": "iter(solving) failed: %s" % e,
            "first_yield_ok": False,
            "solver_time_s": round(time.time() - t0, 3),
        }
    # Prove next(iter(...)) works (the e1e309fa bug was next(solving)).
    flipping_a = charge_flipping.weak_reflection_improved_iterator(delta=None)
    solving_a = charge_flipping.solving_iterator(flipping_a, ma, **kw)
    it_a = iter(solving_a)
    try:
        first = next(it_a)
    except TypeError as e:
        it_a.close()
        return {
            "error": "next(iter(solving)) TypeError: %s" % e,
            "first_yield_ok": False,
            "solver_time_s": round(time.time() - t0, 3),
        }
    except Exception as e:
        it_a.close()
        return {
            "error": "first yield failed: %s: %s" % (type(e).__name__, e),
            "first_yield_ok": False,
            "solver_time_s": round(time.time() - t0, 3),
        }
    it_a.close()

    # Fresh CF run through the deadline helper (same path as production CF).
    flipping2 = charge_flipping.weak_reflection_improved_iterator(delta=None)
    solving2 = charge_flipping.solving_iterator(flipping2, ma, **kw)
    deadline = t0 + budget
    try:
        timed_out = _loop_cf_with_deadline(solving2, deadline)
    except TypeError as e:
        return {
            "error": "_loop_cf_with_deadline TypeError: %s" % e,
            "first_yield_ok": True,
            "first_yield_type": type(first).__name__,
            "solver_time_s": round(time.time() - t0, 3),
        }
    n_peaks = 0
    try:
        fl = solving2.flipping_iterator
        fft_map = fl.rho_map
        sites, heights = _peak_list(fft_map, 8, 1.2)
        n_peaks = len(sites)
    except Exception:
        n_peaks = 0
    return {
        "error": None,
        "timed_out": bool(timed_out),
        "first_yield_ok": True,
        "first_yield_type": type(first).__name__,
        "n_peaks": int(n_peaks),
        "had_phase_transition": bool(
            getattr(solving2, "had_phase_transition", False)),
        "n_solutions": len(getattr(solving2, "f_calc_solutions", []) or []),
        "solver_time_s": round(time.time() - t0, 3),
    }



def run(req):
    from cctbx import dmtbx, maptbx, miller
    from cctbx.array_family import flex
    from scitbx.random import set_random_seed
    from smtbx.ab_initio import charge_flipping
    from libtbx import group_args

    if req.get("action") == "cf_deadline_probe":
        return _cf_deadline_probe(req)

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
    # Refinement-based ranking is non-centrosymmetric only; centrosymmetric
    # selection (E-map check) and its time accounting are unchanged.
    refine_on = (not is_centric) and bool(req.get("refine", True))
    ckpt_path = req.get("checkpoint_path")
    input_indices = [(int(h), int(k), int(l)) for h, k, l in req["hkl"]]

    candidates = []
    tangent_meta = {"error": None}
    cf_meta = {"error": None}
    cf_seed_list = req.get("cf_seed_list") or [0, 1, 7, 42, 2025]

    def checkpoint():
        for c in candidates:
            if c.get("name") == "tangent":
                c["method"] = "tangent"
        try:
            _write_checkpoint(ckpt_path, _assemble_result(
                candidates, tangent_meta, cf_meta, cf_seed_list, n_nonH, ma,
                is_centric, t0, partial=True))
        except Exception:
            pass  # a failed checkpoint must never abort phasing

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
        checkpoint()
    except Exception as e:
        tangent_meta = {"error": "%s: %s" % (type(e).__name__, e),
                        "traceback": traceback.format_exc()}

    # ----------------------------------------------------------- charge flipping
    # Multi-seed CF: fixed documented seed list, within remaining wall time.
    cf_runs = []
    # hard stop inside overall budget; non-centro keeps a small slice for the
    # refinement stage so the whole worker stays inside time_budget_s
    cf_deadline = t0 + budget - (REFINE_TIME_RESERVE_S if refine_on else 0.0)
    try:
        for seed_i, cf_seed in enumerate(cf_seed_list):
            if time.time() >= cf_deadline:
                cf_runs.append({"cf_seed": cf_seed, "skipped": True,
                                "reason": "budget_exhausted"})
                continue
            try:
                set_random_seed(int(seed) + 17 + int(cf_seed))
                flipping = charge_flipping.weak_reflection_improved_iterator(
                    delta=None)
                # yield often enough that the wall-clock check can abort near
                # the deadline (flipping_interval stays the diagnostic step).
                flip_interval = int(cf_params.get("flipping_interval", 60))
                yield_every = int(cf_params.get("yield_solving_interval",
                                                 min(10, max(1, flip_interval))))
                extra = group_args(
                    max_attempts_to_get_phase_transition=int(
                        cf_params.get("max_attempts_to_get_phase_transition", 5)),
                    max_attempts_to_get_sharp_correlation_map=int(
                        cf_params.get(
                            "max_attempts_to_get_sharp_correlation_map", 5)),
                    max_solving_iterations=int(
                        cf_params.get("max_solving_iterations", 500)),
                    yield_solving_interval=yield_every,
                    yield_during_delta_guessing=True,
                )
                amp_type = cf_params.get("amplitude_type", "quasi-E")
                if amp_type == "quasi-E":
                    extra.normalisations_for = (
                        charge_flipping.amplitude_quasi_normalisations)
                solving = charge_flipping.solving_iterator(
                    flipping, ma, **extra.__dict__)
                seed_timed_out = _loop_cf_with_deadline(solving, cf_deadline)
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
                    "timed_out": bool(seed_timed_out),
                }
                if note:
                    cand["note"] = note
                candidates.append(cand)
                cf_meta = {"n_seeds_requested": len(cf_seed_list),
                           "n_seeds_ok": len(cf_runs) + 1,
                           "timed_out": bool(seed_timed_out),
                           "runs": list(cf_runs),
                           "checkpoint": True}
                checkpoint()
                cf_runs.append({
                    "cf_seed": int(cf_seed),
                    "fom": fom,
                    "skew": skew,
                    "had_phase_transition": had,
                    "n_solutions": n_sol,
                    "n_peaks": len(sites),
                    "cc_peak_height": cc_val,
                    "timed_out": bool(seed_timed_out),
                })
                if seed_timed_out:
                    # Stop starting further seeds; remaining wall time is gone.
                    for skip_seed in cf_seed_list[seed_i + 1:]:
                        cf_runs.append({
                            "cf_seed": skip_seed,
                            "skipped": True,
                            "reason": "budget_exhausted",
                        })
                    break
            except Exception as e:
                cf_runs.append({
                    "cf_seed": int(cf_seed),
                    "error": "%s: %s" % (type(e).__name__, e),
                    "traceback": traceback.format_exc(),
                })
        cf_meta = {
            "n_seeds_requested": len(cf_seed_list),
            "n_seeds_ok": sum(1 for r in cf_runs if "fom" in r),
            "timed_out": any(r.get("timed_out") for r in cf_runs)
            or any(r.get("skipped") for r in cf_runs),
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

    refine_meta = None
    if candidates and refine_on:
        refine_meta = _refine_candidates(
            ma, candidates, n_nonH, deadline=t0 + budget)
    return _assemble_result(
        candidates, tangent_meta, cf_meta, cf_seed_list, n_nonH, ma,
        is_centric, t0, partial=False, refine_meta=refine_meta)


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
    _write_checkpoint(out_path, result)  # atomic: never a half-written file


if __name__ == "__main__":
    main()
else:
    if globals().get("REQUEST_JSON"):
        main()
