"""16 (v1) — Predição 2025 nos 25 municípios (motor do caju: src/pred_a, TTA x2) com as 5 sementes finais.
Classe = prob >= limiar (CV), só em classe MapBiomas compatível, pós-processamento 1 ha, recorte pelo limite municipal.
Uso: python 16_predicao.py <limiar> -> dados/predicoes/v1/<rid>_{prob,class}.tif ; reports/v1_area.json. Retomável.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

from src import pred_a as P
from src.config import load_config

cfg = load_config(None); R = Path(cfg["_output_root"]); D = R / "dados"
OD = D / "predicoes/v1"; OD.mkdir(parents=True, exist_ok=True)
COMPAT = cfg["amostragem"]["auto_classificacao"]["classes_compativeis"]
THR = float(sys.argv[1])
mem = P.load_members(sorted((R / "modelos").glob("final_seed*.pt")))
mun = gpd.read_file(D / "pam/municipios_selecionados.gpkg")
rep = R / "reports/v1_area.json"
res = json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else []
feitos = {r["regiao"] for r in res}
print(f"membros={len(mem)} limiar={THR}", flush=True)
for v in sorted((D / "embeddings").glob("m*.vrt")):
    rid = v.stem
    if rid in feitos and (OD / f"{rid}_class.tif").exists():
        continue
    U, prof = P.predict(v, mem)
    g = mun[mun["cod"].astype(str) == rid[1:8]].to_crs(prof["crs"])
    U[geometry_mask(g.geometry, U.shape, prof["transform"])] = np.nan
    with rasterio.open(D / f"mapbiomas/{rid}_mb.tif") as m:
        mb = m.read(1)
    nod = ~np.isfinite(U)
    raw = (np.nan_to_num(U) >= THR) & ~nod
    cls = P.pos_proc(raw & np.isin(mb, COMPAT)) & ~nod
    pr = dict(driver="GTiff", height=U.shape[0], width=U.shape[1], count=1, dtype="uint8", crs=prof["crs"],
              transform=prof["transform"], nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    with rasterio.open(OD / f"{rid}_prob.tif", "w", **pr) as d:
        d.write(np.where(nod, 255, np.clip(np.round(np.nan_to_num(U) * 100), 0, 100)).astype("uint8"), 1)
    with rasterio.open(OD / f"{rid}_class.tif", "w", **pr) as d:
        d.write(np.where(nod, 255, cls).astype("uint8"), 1)
    r = {"regiao": rid, "manga_ha": round(cls.sum() / 100, 1), "manga_ha_sem_mascara": round(raw.sum() / 100, 1)}
    res.append(r); print(r, flush=True)
    rep.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("PREDICT_V1_OK", len(res))
