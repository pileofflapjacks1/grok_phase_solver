"""Coordinate proposal track (research). Not the ``gps-solve`` default.

Importing this package does not import torch. The denoiser lives in
``generate.model`` and is loaded only when a caller asks for it.
"""

from grok_phase_solver.generate.propose import propose

__all__ = ["propose"]
