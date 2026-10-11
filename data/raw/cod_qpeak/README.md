# COD inputs for the Q-peak benchmark

CIF + Fobs HKL for COD 2200239, 2200001 and 2230859, used only by
`scripts/run_qpeak_benchmark.py`. They live outside `data/raw/cod/` so they do
not join the default panel of `scripts/run_cod_stratified_bench.py`.

Source: https://www.crystallography.net/cod/<id>.cif and `<id>.hkl`, byte-identical
to the COD downloads on 2026-10-11. COD data are dedicated to the public domain
(CC0); please acknowledge the original authors (cited in each CIF).

To re-fetch:

```bash
for id in 2200239 2200001 2230859; do
  for ext in cif hkl; do
    curl -fsSL -o data/raw/cod_qpeak/$id.$ext https://www.crystallography.net/cod/$id.$ext
  done
done
```
