# Demo: end-to-end solve without lab data

Synthetic small-molecule style data packaged as SHELX `.hkl` + `.ins`.

```bash
# from repo root, after: pip install -e .
gps-solve \
  --hkl examples/demo_solve/demo.hkl \
  --ins examples/demo_solve/demo.ins \
  --method charge_flipping \
  --n-iter 100 \
  --out examples/demo_solve/out
```

Then open:

- `examples/demo_solve/out/report.md`
- `examples/demo_solve/out/density_slice.png`
- `examples/demo_solve/out/peaks.csv`

Replace `demo.hkl` / `demo.ins` with **your** experimental files using the same command.
