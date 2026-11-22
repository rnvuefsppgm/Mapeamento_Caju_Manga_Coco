"""17 (v1) — Raster ÚNICO dos 25 municípios (EPSG:32724, 10 m). GeoTIFF esparso escrito por janelas: a extensão total
(Jaíba-MG a Ipanguaçu-RN) é grande demais para montar na RAM. Saídas: dados/predicoes/manga_2025_25mun_{classe,prob}.tif
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

D = Path(__file__).parent / "dados/predicoes"
for suf, nome in (("class", "classe"), ("prob", "prob")):
    fs = sorted((D / "v1").glob(f"m*_{suf}.tif"))
    srcs = [rasterio.open(f) for f in fs]
    b = np.array([tuple(s.bounds) for s in srcs]); res = srcs[0].res[0]
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    tr = from_origin(x0, y1, res, res)
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=srcs[0].crs, transform=tr, nodata=255,
                tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=2, sparse_ok=True, BIGTIFF="YES")
    out = D / f"manga_2025_25mun_{nome}.tif"
    with rasterio.open(out, "w+", **prof) as dst:
        for s in srcs:
            w = from_bounds(*s.bounds, transform=tr).round_offsets().round_lengths()
            a = s.read(1)
            cur = dst.read(1, window=w)
            dst.write(np.where(a != 255, a, cur), 1, window=w)
            s.close()
    print(out.name, len(fs), "municípios", (H, W), flush=True)
print("MOSAICO_OK")
