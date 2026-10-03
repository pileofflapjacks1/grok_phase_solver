"""Single entry point a later caller can use. No agent loop. No torch at import."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from grok_phase_solver.generate.geometry import sample_random_fracs
from grok_phase_solver.generate.noise import SAMPLE_STEPS
from grok_phase_solver.generate.score import (
    AMPLITUDE_SOURCE_DEPOSITED_FCALC,
    assign_uniqueness,
    score_sites,
)

METHODS = ("random", "denoiser", "diffusion_hybrid")


def propose(
    elements: Sequence[str],
    cell: Sequence[float],
    space_group: str,
    z: int,
    n: int = 1,
    method: str = "denoiser",
    seed: int = 0,
    model: Any = None,
    reference_fracs: Optional[np.ndarray] = None,
    reference_elements: Optional[Sequence[str]] = None,
    hkl: Optional[np.ndarray] = None,
    amplitudes: Optional[np.ndarray] = None,
    amplitude_source: Optional[str] = None,
    count_reconstruction: bool = False,
    fit_examples: Optional[Sequence[Mapping]] = None,
    fit_steps: int = 0,
    diffusion_steps: int = 2,
    diffusion_starts: int = 1,
    diffusion_polish: int = 2,
    sample_steps: int = SAMPLE_STEPS,
) -> list[dict]:
    """Propose ``n`` fractional coordinate sets and score each one.

    Parameters
    ----------
    elements, cell, space_group, z :
        Known composition (asymmetric-unit elements), cell (6,), Hermann–Mauguin
        symbol, and formula-unit count. Z is recorded. It is not expanded into
        extra copies; the coordinate set is the asymmetric unit.
    method :
        ``random`` (uniform draws, retried only on clash), ``denoiser`` (the
        coordinate model), or ``diffusion_hybrid`` (experimental phase-path flag;
        not the default). ``diffusion_hybrid`` needs ``hkl`` and ``amplitudes``.
    model :
        A fitted denoiser. If omitted, ``fit_examples`` and ``fit_steps`` train
        one. Otherwise the denoiser is the zero-initialized network at ``seed``.
    count_reconstruction :
        Set the ``reconstructed`` flag. Use this only for a held-out structure
        that was not in ``fit_examples``.

    Returns
    -------
    list of dicts with ``fracs``, ``elements``, ``cell``, ``space_group``, ``z``,
    ``method``, and ``score``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    requested = [str(e) for e in elements]
    cell_arr = np.asarray(cell, dtype=np.float64).reshape(6)
    n_draw = int(n)
    if n_draw < 1:
        raise ValueError("n must be >= 1")

    if method == "random":
        drawn = _random_draws(requested, cell_arr, space_group, n_draw, int(seed))
        source = None
    elif method == "denoiser":
        drawn = _denoiser_draws(
            requested,
            cell_arr,
            n_draw,
            int(seed),
            model=model,
            fit_examples=fit_examples,
            fit_steps=int(fit_steps),
            sample_steps=int(sample_steps),
        )
        source = None
    else:
        if hkl is None or amplitudes is None:
            raise ValueError(
                "diffusion_hybrid needs hkl and amplitudes. It is the experimental "
                "phase-path flag, not the coordinate default."
            )
        drawn = _diffusion_draws(
            requested,
            cell_arr,
            space_group,
            np.asarray(hkl),
            np.asarray(amplitudes, dtype=np.float64),
            n_draw,
            int(seed),
            n_steps=int(diffusion_steps),
            n_starts=int(diffusion_starts),
            n_polish=int(diffusion_polish),
        )
        source = amplitude_source or AMPLITUDE_SOURCE_DEPOSITED_FCALC

    samples: list[dict] = []
    for fracs, els, extra in drawn:
        score = score_sites(
            fracs,
            els,
            requested,
            cell_arr,
            space_group,
            reference_fracs=reference_fracs,
            reference_elements=reference_elements,
            hkl=hkl,
            amplitudes=amplitudes,
            count_reconstruction=count_reconstruction,
        )
        row = {
            "fracs": np.asarray(fracs, dtype=np.float64).reshape(-1, 3),
            "elements": list(els),
            "cell": cell_arr.copy(),
            "space_group": str(space_group),
            "z": int(z),
            "method": method,
            "score": score,
            "amplitude_source": source,
        }
        row.update(extra)
        samples.append(row)
    assign_uniqueness(samples)
    return samples


def _random_draws(elements, cell, space_group, n, seed):
    rng = np.random.default_rng(int(seed))
    rows = []
    for _ in range(n):
        fracs, accepted = sample_random_fracs(
            len(elements), elements, cell, space_group, rng
        )
        rows.append((fracs, list(elements), {"clash_rejected_draw": not accepted}))
    return rows


def _denoiser_draws(elements, cell, n, seed, *, model, fit_examples, fit_steps, sample_steps):
    from grok_phase_solver.generate.model import build_denoiser, sample_denoiser, train_denoiser

    torch_seed_note = {"trained_on_call": False, "fit_steps": 0}
    if model is None and fit_examples and fit_steps > 0:
        model = train_denoiser(fit_examples, steps=fit_steps, seed=seed)
        torch_seed_note = {"trained_on_call": True, "fit_steps": int(fit_steps)}
    elif model is None:
        import torch

        torch.manual_seed(int(seed))
        model = build_denoiser()
        model.eval()
    coords = sample_denoiser(
        model, elements, cell, n=n, seed=seed, n_steps=sample_steps
    )
    rows = []
    for i in range(n):
        rows.append((coords[i], list(elements), dict(torch_seed_note)))
    return rows


def _diffusion_draws(
    elements,
    cell,
    space_group,
    hkl,
    amplitudes,
    n,
    seed,
    *,
    n_steps,
    n_starts,
    n_polish,
):
    from grok_phase_solver.models.diffusion_phase import diffusion_hybrid_solve
    from grok_phase_solver.pipeline.peaks import pick_density_peaks

    rows = []
    n_req = len(elements)
    for i in range(n):
        _ph, rho, info = diffusion_hybrid_solve(
            hkl,
            amplitudes,
            cell,
            n_steps=n_steps,
            n_starts=n_starts,
            n_polish=n_polish,
            polish="charge_flipping",
            use_learned_score=False,
            use_free_fom_gate=True,
            seed=int(seed) + i,
            d_min=1.5,
            verbose=False,
        )
        peaks = pick_density_peaks(
            rho,
            n_peaks=max(n_req, 1),
            min_fract_dist=0.05,
            min_sigma=0.5,
        )
        n_found = len(peaks)
        # Do not pad. A short peak list does not match the requested composition.
        if n_found <= 0:
            fracs = np.zeros((0, 3), dtype=np.float64)
            els: list[str] = []
        elif n_found < n_req:
            fracs = np.vstack([np.asarray(p.fract, dtype=np.float64) for p in peaks])
            els = list(elements)[:n_found]
        else:
            fracs = np.vstack(
                [np.asarray(p.fract, dtype=np.float64) for p in peaks[:n_req]]
            )
            els = list(elements)
        rows.append(
            (
                fracs,
                els,
                {
                    "experimental": True,
                    "oracle_amplitudes": True,
                    "n_peaks": n_found,
                    "diffusion_status": info.get("status", "experimental"),
                    "space_group_used_for_peaks": space_group,
                },
            )
        )
    return rows
