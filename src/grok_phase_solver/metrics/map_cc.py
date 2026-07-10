"""Real-space and Fourier-shell correlation metrics."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def map_correlation(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """Pearson correlation coefficient between two density maps (fixed origin)."""
    a = np.asarray(rho1, dtype=np.float64).ravel()
    b = np.asarray(rho2, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-16:
        return 0.0
    return float(np.dot(a, b) / den)


def map_correlation_origin_invariant(
    rho1: np.ndarray,
    rho2: np.ndarray,
    also_inverted: bool = True,
) -> Tuple[float, Tuple[int, int, int]]:
    """
    Maximum Pearson CC over integer origin shifts (and optional inversion).

    Uses FFT cross-correlation:
        C(t) = IFFT( FFT(ρ1) * conj(FFT(ρ2)) )
    Peak location is the optimal lattice translation aligning the maps.

    Returns (best_cc, (tx, ty, tz)).
    """
    a = np.asarray(rho1, dtype=np.float64)
    b = np.asarray(rho2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    def best_shift_cc(x, y):
        x0 = x - x.mean()
        y0 = y - y.mean()
        nx = np.linalg.norm(x0.ravel())
        ny = np.linalg.norm(y0.ravel())
        if nx < 1e-16 or ny < 1e-16:
            return 0.0, (0, 0, 0)
        corr = np.fft.ifftn(np.fft.fftn(x0) * np.conj(np.fft.fftn(y0)))
        corr = np.real(corr) / (nx * ny)
        idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
        return float(corr[idx]), tuple(int(i) for i in idx)

    cc, shift = best_shift_cc(a, b)
    if also_inverted:
        # Enantiomorph / inversion through origin: compare to ρ(-r)
        b_inv = b[::-1, ::-1, ::-1]
        # after reverse, re-roll so index 0 stays origin-ish
        b_inv = np.roll(b_inv, (1, 1, 1), axis=(0, 1, 2))
        cc2, shift2 = best_shift_cc(a, b_inv)
        if cc2 > cc:
            cc, shift = cc2, shift2
    return cc, shift


def fourier_shell_correlation(
    F1: np.ndarray,
    F2: np.ndarray,
    hkl: np.ndarray,
    cell: np.ndarray,
    n_shells: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fourier shell correlation (FSC) between two complex structure-factor sets.

    FSC(s) = Σ Re(F1 F2*) / sqrt(Σ|F1|² Σ|F2|²)  within resolution shells.

    Returns (shell_d_centers, fsc_values).
    """
    from grok_phase_solver.physics.reciprocal import d_spacing, resolution_shells

    F1 = np.asarray(F1, dtype=np.complex128)
    F2 = np.asarray(F2, dtype=np.complex128)
    d = d_spacing(hkl, cell)
    shell_id, _, centers = resolution_shells(d, n_shells=n_shells)
    fsc = np.zeros(n_shells)
    for s in range(n_shells):
        m = shell_id == s
        if not np.any(m):
            fsc[s] = np.nan
            continue
        num = np.sum(np.real(F1[m] * np.conj(F2[m])))
        den = np.sqrt(np.sum(np.abs(F1[m]) ** 2) * np.sum(np.abs(F2[m]) ** 2))
        fsc[s] = num / (den + 1e-16)
    return centers, fsc
