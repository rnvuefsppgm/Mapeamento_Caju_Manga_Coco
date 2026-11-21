"""06 — Pós-processamento dos mapas classificados (ETAPAS 39-40).

Aplica unidade mínima de mapeamento e filtro de maioria, e opcionalmente
vetoriza para GeoPackage.

Uso:
    python 06_postprocess.py --vectorize
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.cog_utils import write_cog
from src.postprocessing import (apply_min_mapping_unit, majority_filter,
                                vectorize)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pós-processamento das classificações.")
    p.add_argument("--config", default=None)
    p.add_argument("--vectorize", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("06_postprocess", cfg)

    pred_dir = resolve_path(cfg, "predictions")
    res = cfg["harmonization"]["resolution"]
    min_ha = cfg["postprocessing"]["min_mapping_unit_ha"]

    class_files = sorted(pred_dir.glob("*_cashew_classification.tif"))
    if not class_files:
        log.error("Nenhuma classificação encontrada em %s.", pred_dir)
        return

    # Barra de progresso: um passo por mapa de classificação.
    for cf in tqdm(class_files, desc="Pós-processamento", unit="mapa"):
        with rasterio.open(cf) as ds:
            cls = ds.read(1)
            transform, crs = ds.transform, ds.crs

        nodata = cls == 255
        cls_clean = apply_min_mapping_unit(np.where(nodata, 0, cls).astype("uint8"),
                                           res, min_ha)
        if cfg["postprocessing"]["apply_majority_filter"]:
            cls_clean = majority_filter(cls_clean,
                                        cfg["postprocessing"]["majority_kernel"])
        cls_clean[nodata] = 255

        out = cf.with_name(cf.stem + "_clean.tif")
        write_cog(out, cls_clean[None], transform, crs, nodata=255,
                  band_names=["cashew_class"], dtype="uint8")
        log.info("Pós-processado: %s", out.name)

        if args.vectorize:
            mun = cf.stem.replace("_cashew_classification", "")
            gdf = vectorize(cls_clean, transform, crs, min_ha, res,
                            attrs={"municipality": mun,
                                   "year": cfg["temporal"]["start_date"][:4],
                                   "source_model": cfg["experiment"]["name"]})
            if len(gdf):
                gpkg = pred_dir / f"{mun}_cashew_polygons.gpkg"
                gdf.to_file(gpkg, driver="GPKG")
                log.info("  -> %d polígonos: %s", len(gdf), gpkg.name)


if __name__ == "__main__":
    main()
