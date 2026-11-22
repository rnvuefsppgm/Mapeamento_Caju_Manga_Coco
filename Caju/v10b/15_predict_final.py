"""15 (v10a-ft) — Predição final 2025 dos 25 municípios: 8 membros (3 sementes finais + 5 folds do CV v10a-ft),
TTA x4, fusão com XGBoost regional (se reports/fusao.json escolheu a < 1), limiar escolhido, pós-processamento
(manchas e buracos < 1 ha). Saídas em predictions/2025: <rid>_prob.tif (U-Net, 0-100), <rid>_final.tif (prob.
final 0-100), <rid>_class.tif (0/1). Relatório: reports/area_caju_2025_v10.json (caju_ha = final).
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

from src import pred_a as P

ROOT = Path(__file__).parent
OD = ROOT / "predictions/2025"; OD.mkdir(parents=True, exist_ok=True)
fu = json.loads((ROOT / "reports/fusao.json").read_text(encoding="utf-8"))["escolha"]
A, THR = float(fu["a"]), float(fu["limiar"])
files = sorted((ROOT / "models").glob("final_seed*.pt"))  # v10b: só as 5 sementes finais (folds pararam cedo)
mem = P.load_members(files)
print(f"membros={len(mem)} a={A} limiar={THR}", flush=True)
mun = gpd.read_file(ROOT.parent / "dados_externos/municipios25/municipios25.gpkg")
rep = ROOT / "reports/area_caju_2025_v10.json"
res = json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else []
done = {r["regiao"] for r in res}
for v in sorted((ROOT.parent / "v8/data/ee_exports/2025/embeddings").glob("m*.vrt")):
    rid = v.stem
    if rid in done and (OD / f"{rid}_class.tif").exists():
        continue
    U, prof = P.predict(v, mem)
    g = mun[mun["cod"].astype(str) == rid[1:8]].to_crs(prof["crs"])
    out = geometry_mask(g.geometry, U.shape, prof["transform"]) if len(g) else np.zeros(U.shape, bool)
    U[out] = np.nan
    XR = rasterio.open(ROOT.parent / f"v10a/data/fontes_a/{rid}_xr.tif").read(1).astype("float32")
    XR[XR == 255] = np.nan; XR /= 100
    F = U if A >= 1 else np.where(np.isfinite(XR), A * U + (1 - A) * XR, U)
    nod = ~np.isfinite(F)
    cls_raw = (np.nan_to_num(F, nan=0) >= THR) & ~nod
    cls = P.pos_proc(cls_raw) & ~nod
    pr = dict(driver="GTiff", height=U.shape[0], width=U.shape[1], count=1, dtype="uint8", crs=prof["crs"],
              transform=prof["transform"], nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    for arr, suf in ((U, "prob"), (F, "final")):
        with rasterio.open(OD / f"{rid}_{suf}.tif", "w", **pr) as d:
            d.write(np.where(np.isfinite(arr), np.clip(np.round(arr * 100), 0, 100), 255).astype("uint8"), 1)
    with rasterio.open(OD / f"{rid}_class.tif", "w", **pr) as d:
        d.write(np.where(nod, 255, cls).astype("uint8"), 1)
    r = {"regiao": rid, "caju_ha": round(cls.sum() / 100, 1), "caju_ha_sem_pos": round(cls_raw.sum() / 100, 1),
         "caju_ha_unet": round(float(((np.nan_to_num(U) >= THR) & np.isfinite(U)).sum()) / 100, 1), "a": A, "limiar": THR}
    res.append(r); print(r, flush=True)
    rep.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("PREDICT_V10A_OK", len(res))
