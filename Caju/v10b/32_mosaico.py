"""32 (v10b) — Raster ÚNICO dos 25 municípios (EPSG:32724, 10 m): classe (0/1, 255 = fora) e probabilidade final
(0-100, 255 = fora), em COG comprimido. Saída: predictions/caju_2025_25mun_{classe,prob}.tif
"""
from pathlib import Path

import rasterio
from rasterio.merge import merge

ROOT = Path(__file__).parent
OD = ROOT / "predictions"
for suf, nome in (("class", "classe"), ("final", "prob")):
    fs = sorted((OD / "2025").glob(f"m*_{suf}.tif"))
    srcs = [rasterio.open(f) for f in fs]
    arr, tr = merge(srcs, nodata=255, method="first")  # municípios não se sobrepõem (recorte pelo limite)
    prof = dict(driver="COG", height=arr.shape[1], width=arr.shape[2], count=1, dtype="uint8", crs=srcs[0].crs,
                transform=tr, nodata=255, compress="DEFLATE", predictor=2, overviews="AUTO", BIGTIFF="IF_SAFER")
    out = OD / f"caju_2025_25mun_{nome}.tif"
    with rasterio.open(out, "w", **prof) as d:
        d.write(arr[0], 1)
    for s in srcs:
        s.close()
    print(out, len(fs), "municípios", arr.shape, flush=True)
print("MOSAICO_OK")
