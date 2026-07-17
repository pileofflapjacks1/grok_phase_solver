"""
Scientist-facing GUI for grok_phase_solver (Streamlit).

Launch::

    gps-gui
    # or
    python -m grok_phase_solver.gui
    python scripts/run_gui.py
"""

from __future__ import annotations

__all__ = ["run_gui"]


def run_gui(argv: list[str] | None = None) -> None:
    """Start the Streamlit app (spawns streamlit CLI)."""
    from grok_phase_solver.gui.launch import main

    main(argv)
