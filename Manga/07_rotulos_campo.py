"""07 — Raster de rótulos de campo (Livramento de Nossa Senhora) na grade 10 m dos embeddings, em CÓDIGOS.

Regras do caju (v9/v10/v10b) guardadas sem decidir ainda o que é negativo:
  1 = manga (polígono, erosão de 1 px na borda, src.labels.rasterize_labels)
  2 = faixa ignorada: anel de `ignore_buffer_m` (25 m) em volta e até `neg_distancia_m` (100 m) dos polígonos
  3 = negativo "denso": fora dos polígonos, > 100 m, dentro da área amostrada (envoltória convexa + 500 m)
  4 = negativo confiável MapBiomas: classe fora de C (compatíveis com manga), > 100 m, em qualquer lugar
  5 = desconhecido: classe C fora da área amostrada (pode ser manga não mapeada)
255 = sem dado (embedding NoData ou fora do município + 1 km)
Saída: dados/rotulos/<rid>_campo.tif ; resumo em logs/07_rotulos_campo.log
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from src.config import load_config, resolve_path, setup_logging
from src.labels import rasterize_labels

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("07_rotulos_campo", cfg)
M, A, RT = cfg["manga"], cfg["manga"]["amostras"], cfg["manga"]["rotulos"]


def main():
    p = gpd.read_file(resolve_path(cfg, "amostras") / "manga_poligonos.gpkg").to_crs(M["crs"])
    regs = gpd.read_file(resolve_path(cfg, "pam") / "municipios_selecionados.gpkg").to_crs(M["crs"])
    for cod in sorted(set(p.cod.astype(int))):
        rid = next(r for r in (resolve_path(cfg, "aoi") / "regioes.txt").read_text().split() if r.startswith(f"m{cod}_"))
        pp = p[p.cod.astype(int) == cod]
        with rasterio.open(resolve_path(cfg, "embeddings") / f"{rid}.vrt") as e:
            sh, tr, crs = (e.height, e.width), e.transform, e.crs
            nod = e.read(1) == 0  # banda 1 = 0 -> NoData (uint8 deslocado)
        lab = rasterize_labels(pp, sh, tr, cfg)  # 1 manga (erodido), 0 fundo, 255 borda erodida
        code = np.full(sh, 5, "uint8")
        perto = rasterize(((g, 1) for g in pp.buffer(A["neg_distancia_m"]).geometry), sh, transform=tr).astype(bool)
        amostr = rasterize([(pp.union_all().convex_hull.buffer(RT["area_amostrada_buffer_m"]), 1)], sh,
                           transform=tr).astype(bool)
        mb = np.zeros(sh, "uint8")
        with rasterio.open(M["fontes"]["mapbiomas_tif"]) as m:
            reproject(rasterio.band(m, 1), mb, dst_transform=tr, dst_crs=crs, dst_nodata=0,
                      resampling=Resampling.nearest)
        compat = np.isin(mb, RT["classes_mb_compativeis"])
        code[amostr] = 3
        code[~compat & (mb > 0)] = 4
        code[perto] = 2
        code[lab == 1] = 1
        munb = rasterize([(g.buffer(M["regioes"]["buffer_municipio_m"]), 1) for g in regs[regs.cod == cod].geometry],
                         sh, transform=tr).astype(bool)
        code[nod | ~munb] = 255
        prof = dict(driver="GTiff", height=sh[0], width=sh[1], count=1, dtype="uint8", crs=crs, transform=tr,
                    nodata=255, tiled=True, blockxsize=512, blockysize=512, compress="deflate")
        with rasterio.open(resolve_path(cfg, "rotulos") / f"{rid}_campo.tif", "w", **prof) as d:
            d.write(code, 1)
        v, c = np.unique(code, return_counts=True)
        log.info(f"RESUMO {rid}: " + " | ".join(f"código {a}: {b / 100:,.0f} ha" for a, b in zip(v, c)))


if __name__ == "__main__":
    main()
