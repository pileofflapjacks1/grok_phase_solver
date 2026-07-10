"""
grok_phase_solver
=================

Open-source framework for ab initio solution of the crystallographic phase
problem: recover phases φ(hkl) given amplitudes |F(hkl)| under physical
constraints (positivity, atomicity, symmetry, Parseval).

Physics foundation
------------------
Electron density (inverse Fourier transform of structure factors)::

    ρ(r) = (1/V) Σ_h |F(h)| exp(i φ(h) − 2π i h·r)

Structure factors from atomic model::

    F(h) = Σ_j f_j(s) exp(2π i h·r_j) exp(−B_j s²)

where s = sin(θ)/λ.

Package layout
--------------
- ``io``        : CIF / HKL / reflection tables
- ``physics``   : form factors, structure factors, density, symmetry
- ``data``      : COD download, synthetic structure generation
- ``solvers``   : physics baselines (charge flipping, HIO) + ML hooks
- ``metrics``   : mean phase error, map CC, R-factors
- ``models``    : neural architectures (PhAI-style interface; Phase 2+)
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
