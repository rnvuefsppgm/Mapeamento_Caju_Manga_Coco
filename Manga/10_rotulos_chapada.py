"""10 (v1) — Rótulos da Chapada para a rede (receita caju v10b, só com as fontes mais seguras).

manga (1)  = polígonos de campo (código 1, Livramento) + manchas confiáveis do mapa automático A5
não (0)    = classes seguras MapBiomas + prob A5 <= p_neg_xgb em classe compatível + NAO_MANGA da A4a
             + Livramento código 3 (área amostrada) — exceto onde a A5 diz manga (conflito -> ignorado)
ignorado   = resto + faixa de 100 m dos polígonos (código 2) + anel de 2 px em volta da manga + fora do município
Saída: dados/rotulos/<rid>_v1.tif ; resumo no log (<= 10 linhas).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage

from src.config import load_config, resolve_path, setup_logging

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("10_rotulos_chapada", cfg)
T, AM = cfg["treino_v1"], cfg["amostragem"]
SEG, COMPAT = AM["rotulos"]["classes_seguras_nao_coco_mb"], AM["auto_classificacao"]["classes_compativeis"]
D = Path(cfg["_output_root"]) / "dados"
rids = {int(r[1:8]): r for r in (D / "aoi/regioes.txt").read_text().split() if r}
nao = gpd.read_file(D / "rotulos/candidatos/rodada1_auto.gpkg")
nao = nao[nao.pasta == "NAO_MANGA"]

for cod in T["municipios"]:
    rid = rids[cod]
    with rasterio.open(D / f"embeddings/{rid}.vrt") as e:
        H, W, tr, crs, prof = e.height, e.width, e.transform, e.crs, e.profile
        nod = e.read(1) == 0
    rd = lambda p: rasterio.open(p).read(1)
    mb = rd(D / f"mapbiomas/{rid}_mb.tif")
    pa = rd(D / f"predicoes/xgb_auto/{rid}_prob.tif")
    cf = rd(D / f"predicoes/xgb_auto/{rid}_conf.tif") == 1
    dentro = (pa != 255) & ~nod
    L = np.full((H, W), 255, np.uint8)
    L[dentro & np.isin(mb, SEG)] = 0
    L[dentro & np.isin(mb, COMPAT) & (pa <= T["p_neg_xgb"])] = 0
    n_sub = nao[nao.regiao == rid]
    if len(n_sub):
        L[rasterize([(g, 1) for g in n_sub.to_crs(crs).geometry], (H, W), transform=tr).astype(bool) & dentro] = 0
    campo = D / f"rotulos/{rid}_campo.tif"
    if campo.exists():
        K = rd(campo)
        L[(K == 3) & ~cf & (pa < 50)] = 0
        L[(K == 3) & (cf | (pa >= 50))] = 255      # área "amostrada" onde o mapa vê manga: conflito
        L[K == 2] = 255
        L[K == 1] = 1
    L[cf & dentro] = 1
    anel = ndimage.binary_dilation(L == 1, iterations=T["anel_px"]) & (L != 1)
    L[anel] = 255
    L[~dentro] = 255
    p = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs=crs, transform=tr, nodata=255,
             tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    with rasterio.open(D / f"rotulos/{rid}_v1.tif", "w", **p) as d:
        d.write(L, 1)
    log.info(f"RESUMO {rid}: manga {(L == 1).sum() / 100:,.0f} ha | não-manga {(L == 0).sum() / 100:,.0f} ha | "
             f"ignorado {((L == 255) & dentro).sum() / 100:,.0f} ha")
