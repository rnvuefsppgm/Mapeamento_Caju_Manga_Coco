"""Escrita e validação de Cloud Optimized GeoTIFF (COG) — ETAPA 16.

Preserva CRS, transform, resolução, NoData, bandas e escala. Usa overviews
internos e tiling. Validação leve verifica tiling + overviews.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

_COG_PROFILE = {
    "driver": "GTiff",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "compress": "deflate",
    "predictor": 2,          # 2 p/ inteiros; 3 p/ float (ajustado abaixo)
    "BIGTIFF": "IF_SAFER",
}


def write_cog(
    path: str | Path,
    array: np.ndarray,
    transform: Affine,
    crs: Any,
    nodata: Optional[float] = None,
    band_names: Optional[Sequence[str]] = None,
    dtype: Optional[str] = None,
    overview_levels: Sequence[int] = (2, 4, 8, 16),
) -> Path:
    """Escreve um array (C,H,W) ou (H,W) como COG com overviews internos.

    Returns:
        Caminho do arquivo escrito.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        array = array[np.newaxis, ...]
    count, height, width = array.shape
    out_dtype = dtype or str(array.dtype)

    profile = dict(_COG_PROFILE)
    profile["predictor"] = 3 if np.issubdtype(np.dtype(out_dtype), np.floating) else 2
    profile.update(
        dtype=out_dtype, count=count, height=height, width=width,
        transform=transform, crs=crs, nodata=nodata,
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(out_dtype))
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, str(name))
        dst.build_overviews(list(overview_levels), Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    logger.info("COG escrito: %s (%d bandas, %dx%d).", path.name, count, width, height)
    return path


def is_valid_cog(path: str | Path) -> bool:
    """Validação leve: verifica se o arquivo é tiled e possui overviews."""
    with rasterio.open(path) as ds:
        tiled = ds.profile.get("tiled", False)
        has_ovr = len(ds.overviews(1)) > 0 if ds.count >= 1 else False
    ok = bool(tiled and has_ovr)
    if not ok:
        logger.warning("Arquivo pode não ser COG válido: %s (tiled=%s, ovr=%s).",
                       path, tiled, has_ovr)
    return ok
