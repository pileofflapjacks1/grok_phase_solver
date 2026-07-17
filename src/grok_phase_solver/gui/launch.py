"""Launch Streamlit for the gps-solve GUI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def app_path() -> Path:
    return Path(__file__).resolve().parent / "app.py"


def main(argv: list[str] | None = None) -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is required for the GUI.\n"
            "  python -m pip install -e \".[gui]\"\n"
            "  # or:  python -m pip install streamlit",
            file=sys.stderr,
        )
        sys.exit(1)

    args = ["streamlit", "run", str(app_path()), "--browser.gatherUsageStats", "false"]
    if argv:
        args.extend(argv)
    # Forward optional streamlit flags after --
    raise SystemExit(subprocess.call(args))


if __name__ == "__main__":
    main()
