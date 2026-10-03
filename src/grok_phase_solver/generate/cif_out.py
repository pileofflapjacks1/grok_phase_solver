"""Write proposed asymmetric units as a multi-block CIF."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def write_samples_cif(path: Path | str, samples: Sequence[dict]) -> None:
    """One ``data_`` block per sample. ``load_cif`` reads a single block only."""
    blocks: list[str] = []
    for i, sample in enumerate(samples, start=1):
        cell = np.asarray(sample["cell"], dtype=np.float64).reshape(6)
        sg = str(sample["space_group"])
        lines = [
            f"data_gen_{i}",
            f"_symmetry_space_group_name_H-M   '{sg}'",
            f"_cell_formula_units_Z            {int(sample.get('z', 1))}",
            f"_cell_length_a                   {cell[0]:.6f}",
            f"_cell_length_b                   {cell[1]:.6f}",
            f"_cell_length_c                   {cell[2]:.6f}",
            f"_cell_angle_alpha                {cell[3]:.6f}",
            f"_cell_angle_beta                 {cell[4]:.6f}",
            f"_cell_angle_gamma                {cell[5]:.6f}",
            f"_gps_generate_method             {sample.get('method', 'denoiser')}",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
        ]
        fracs = np.asarray(sample["fracs"], dtype=np.float64).reshape(-1, 3)
        elements = list(sample["elements"])
        for j, (el, frac) in enumerate(zip(elements, fracs), start=1):
            lines.append(
                f"{el}{j} {el} {frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f} 1.0"
            )
        blocks.append("\n".join(lines))
    Path(path).write_text("\n\n".join(blocks) + "\n")
