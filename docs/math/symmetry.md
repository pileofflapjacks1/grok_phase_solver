# Space-group symmetry helpers

## Module

`physics/symmetry.py` — gemmi-backed utilities used by the pipeline and seeding.

| Function | Role |
|----------|------|
| `parse_space_group` | HM → centrosymmetry, ops count, crystal system |
| `expand_fractional_coords` | ASU atoms → unit-cell via GroupOps |
| `apply_centro_phase_constraint` | snap φ to {0, π} when centro |
| `filter_systematic_absences` | report / filter absences |
| `space_group_diagnostics` | report.md block |

## Integration

- `gps-solve` records SG diagnostics and notes systematic absences (without
  silently dropping reflections from the export index).
- `partial_phaseed` / `--predicted-model` can expand fragment ASU by SG ops.
- Centrosymmetric auto-routing still uses centro phase discretization in
  `phai_phaseed` when gemmi flags centro.

## Honest limits

- Not a full SHELXL MERGE / twin refinement engine.
- Origin choice in non-P1: prefer origin-invariant mapCC and free FOM.
- Full reciprocal-space symmetry averaging of noisy Fobs remains optional future work.

## Related

- `io/cif.expand_asymmetric_unit` — structure-level expansion
- gemmi `SpaceGroup` / `GroupOps`
