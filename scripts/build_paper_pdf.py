#!/usr/bin/env python3
"""
Build docs/paper/arxiv_draft.pdf from docs/arxiv_draft.md.

Requires:
  - pandoc on PATH, or PANDOC env, or /tmp/pandoc-*/bin/pandoc
  - tectonic on PATH, or TECTONIC env, or /tmp/tectonic

Usage:
  python scripts/build_paper_pdf.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_tool(name: str, env: str, fallbacks: list[str]) -> str:
    if os.environ.get(env):
        return os.environ[env]
    w = shutil.which(name)
    if w:
        return w
    for fb in fallbacks:
        p = Path(fb)
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"{name} not found. Install pandoc + tectonic, or set {env}=/path/to/{name}"
    )


def preprocess(md: str) -> str:
    """Make Unicode safe for pdflatex/tectonic default fonts."""
    parts = re.split(r"(```.*?```)", md, flags=re.S)
    out: list[str] = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            lines = []
            for line in p.splitlines():
                while len(line) > 88:
                    lines.append(line[:88] + "\\")
                    line = "  " + line[88:]
                lines.append(line)
            out.append("\n".join(lines))
        else:
            p = (
                p.replace("φ", r"$\varphi$")
                .replace("≥", r"$\geq$")
                .replace("≤", r"$\leq$")
                .replace("≈", r"$\approx$")
                .replace("–", "--")
                .replace("—", "---")
                .replace("·", " / ")
            )
            out.append(p)
    return "".join(out)


def main() -> int:
    src = ROOT / "docs" / "arxiv_draft.md"
    out_pdf = ROOT / "docs" / "paper" / "arxiv_draft.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "docs" / "paper" / "_arxiv_draft_tex.md"

    pandoc = find_tool(
        "pandoc",
        "PANDOC",
        [
            "/tmp/pandoc-3.6.4-x86_64/bin/pandoc",
            str(Path.home() / "bin" / "pandoc"),
        ],
    )
    tectonic = find_tool(
        "tectonic",
        "TECTONIC",
        ["/tmp/tectonic", str(Path.home() / "bin" / "tectonic")],
    )

    tmp.write_text(preprocess(src.read_text()), encoding="utf-8")
    cmd = [
        pandoc,
        str(tmp),
        "-o",
        str(out_pdf),
        f"--resource-path={ROOT / 'docs'}:{ROOT / 'docs' / 'figures'}:{ROOT}",
        f"--pdf-engine={tectonic}",
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=11pt",
        "--toc",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=blue",
        "-V",
        "urlcolor=blue",
        "--metadata",
        "title=Toward an Open Physics/AI Framework for the Crystallographic Phase Problem",
        "--metadata",
        "author=Grok (xAI) and Joe",
        "-V",
        "linestretch=1.15",
    ]
    print("Building", out_pdf.relative_to(ROOT), "…")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode == 0 and out_pdf.exists():
        print(f"OK  {out_pdf}  ({out_pdf.stat().st_size // 1024} KB)")
        return 0
    print("PDF build failed", file=sys.stderr)
    return r.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
