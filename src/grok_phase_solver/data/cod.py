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

# Curated samples for experimental / Vol-band panels (v0.13+)
COD_SAMPLE_IDS = {
    "2100301": {
        "name": "pyridine-3,5-dicarboxylic acid (dinicotinic acid)",
        "formula": "C7 H5 N O4",
        "space_group": "P21/c",
        "notes": "Small organic, P21/c — PhAI-relevant SG; Vol < 1000 Å³",
        "has_hkl": True,
        "vol_band": "vol_lt_1000",
    },
    "2016452": {
        "name": "small organic (PhAI COD sample)",
        "formula": "—",
        "space_group": "P21/c",
        "notes": "PhAI hybrid + hard-path Fobs panel; Vol < 1000 Å³",
        "has_hkl": True,
        "vol_band": "vol_lt_1000",
    },
    "2017775": {
        "name": "roxithromycin",
        "formula": "C41 H76 N2 O15",
        "space_group": "P212121",
        "notes": "Larger macrolide; Vol > 3500 Å³ hard ab initio control",
        "has_hkl": True,
        "vol_band": "vol_gt_3500",
    },
    # Vol 1000–3500 Å³ band (AI-PhaSeed / Carrozzini hybrid-friendly)
    "2012000": {
        "name": "COD 2012000 (mid-volume organic)",
        "formula": "—",
        "space_group": "P21",
        "notes": "Vol ~1027 Å³ — stratified panel mid band",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "2013000": {
        "name": "COD 2013000 (mid-volume, P-1)",
        "formula": "—",
        "space_group": "P-1",
        "notes": "Vol ~1015 Å³ — centrosymmetric mid band",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "2200000": {
        "name": "COD 2200000 (small, P21)",
        "formula": "—",
        "space_group": "P21",
        "notes": "Vol ~665 Å³ — extra small-cell Fobs control",
        "has_hkl": True,
        "vol_band": "vol_lt_1000",
    },
    # Mid-band expansion (Vol 1000–3500 Å³, CHNOF, deposited Fobs).
    # Molecule size ~26 non-H. Not a 1505-structure panel.
    "1543089": {
        "name": "5-(2-hydroxybenzoyl)-2-(indol-3-yl)pyridine-3-carbonitrile",
        "formula": "C21 H13 N3 O2",
        "space_group": "P21/c",
        "notes": "Vol ~1638 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "1543227": {
        "name": "5,6-dipropylphthalazino[2,3-a]cinnoline-8,13-dione",
        "formula": "C22 H22 N2 O2",
        "space_group": "P21/n",
        "notes": "Vol ~1786 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "1544230": {
        "name": "(E)-4-methoxy-N'-(2,3,4-trimethoxybenzylidene)benzohydrazide monohydrate",
        "formula": "C18 H22 N2 O6",
        "space_group": "P21/c",
        "notes": "Vol ~1878 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "1544651": {
        "name": "(3S,4S)-1-benzyl-3-hydroxy-4-phenyl-tetrahydro-1,5-benzodiazepin-2-one",
        "formula": "C22 H20 N2 O2",
        "space_group": "P21/c",
        "notes": "Vol ~1752 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "1549607": {
        "name": "1-methyl-4-phenyl-3-[4-(trifluoromethyl)phenyl]pyrazolo[3,4-d]pyrimidine",
        "formula": "C19 H13 F3 N4",
        "space_group": "P21/c",
        "notes": "Vol ~1628 Å³ — light-atom organic Fobs (max Z = F)",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "1550274": {
        "name": "3-hydroxy-3-methylisochroman-1-one--2-(carboxymethyl)benzoic acid (1/1)",
        "formula": "C19 H18 O7",
        "space_group": "P21/c",
        "notes": "Vol ~1689 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "2016430": {
        "name": "bis(2,2,2-trinitroethyl) carbonate",
        "formula": "C5 H4 N6 O15",
        "space_group": "Pbca",
        "notes": "Vol ~2611 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "2221836": {
        "name": "2-(4-methylphenyl)-1H-anthraceno[1,2-d]imidazole-6,11-dione",
        "formula": "C22 H14 N2 O2",
        "space_group": "Pbca",
        "notes": "Vol ~3176 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "2227862": {
        "name": "acridinium 3-carboxypyrazine-2-carboxylate",
        "formula": "C19 H13 N3 O4",
        "space_group": "Pbca",
        "notes": "Vol ~3077 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
    },
    "2233297": {
        "name": "acridin-10-ium 6-carboxypyridine-2-carboxylate",
        "formula": "C20 H14 N2 O4",
        "space_group": "C2/c",
        "notes": "Vol ~3160 Å³ — light-atom organic Fobs",
        "has_hkl": True,
        "vol_band": "vol_1000_3500",
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
