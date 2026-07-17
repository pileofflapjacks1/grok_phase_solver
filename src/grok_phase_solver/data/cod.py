"""
Crystallography Open Database (COD) download helpers.

CIF:  https://www.crystallography.net/cod/{id}.cif
HKL:  https://www.crystallography.net/cod/hkl/{id}.hkl
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List, Optional, Union

PathLike = Union[str, Path]

COD_BASE = "https://www.crystallography.net/cod"

# Curated Phase-1 samples
COD_SAMPLE_IDS = {
    "2100301": {
        "name": "pyridine-3,5-dicarboxylic acid (dinicotinic acid)",
        "formula": "C7 H5 N O4",
        "space_group": "P21/c",
        "notes": "Small organic, P21/c — PhAI-relevant SG; neutron structure + COD HKL",
        "has_hkl": True,
    },
    "2016452": {
        "name": "small organic (PhAI COD sample)",
        "formula": "—",
        "space_group": "P21/c",
        "notes": "PhAI hybrid benchmark + experimental Fobs HKL in COD",
        "has_hkl": True,
    },
    "2017775": {
        "name": "roxithromycin",
        "formula": "C41 H76 N2 O15",
        "space_group": "P212121",
        "notes": "Larger macrolide with experimental Fobs HKL in COD",
        "has_hkl": True,
    },
}


def download_cod_cif(
    cod_id: Union[int, str],
    dest_dir: PathLike = "data/raw/cod",
    overwrite: bool = False,
) -> Path:
    """Download a COD CIF by numeric ID."""
    cod_id = str(cod_id).strip()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{cod_id}.cif"
    if out.exists() and not overwrite:
        return out
    url = f"{COD_BASE}/{cod_id}.cif"
    urllib.request.urlretrieve(url, out)
    return out


def download_cod_hkl(
    cod_id: Union[int, str],
    dest_dir: PathLike = "data/raw/cod",
    overwrite: bool = False,
) -> Path:
    """Download COD structure-factor file (.hkl CIF) if available."""
    cod_id = str(cod_id).strip()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{cod_id}.hkl"
    if out.exists() and not overwrite:
        return out
    url = f"{COD_BASE}/hkl/{cod_id}.hkl"
    urllib.request.urlretrieve(url, out)
    return out


def download_phase1_samples(dest_dir: PathLike = "data/raw/cod") -> List[Path]:
    """Download all curated Phase-1 COD entries (CIF + HKL when available)."""
    paths: List[Path] = []
    for cod_id, meta in COD_SAMPLE_IDS.items():
        p = download_cod_cif(cod_id, dest_dir=dest_dir)
        paths.append(p)
        if meta.get("has_hkl"):
            try:
                paths.append(download_cod_hkl(cod_id, dest_dir=dest_dir))
            except Exception as exc:
                print(f"Warning: HKL download failed for {cod_id}: {exc}")
    return paths
