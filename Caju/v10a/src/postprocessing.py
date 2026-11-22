"""Pós-processamento do mapa classificado (ETAPAS 39-40).

Operações justificadas pela resolução (10 m) e pelo tamanho mínimo esperado de
uma plantação de caju:
    * Minimum Mapping Unit (MMU): remove componentes conexos menores que um limiar
      em hectares (ruído de sal-e-pimenta).
    * Majority filter: suaviza a classificação por voto de vizinhança.
Filtros NÃO são aplicados arbitrariamente; parâmetros vêm do config.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def apply_min_mapping_unit(
    classified: np.ndarray, resolution_m: float, min_ha: float
) -> np.ndarray:
    """Remove objetos de caju menores que ``min_ha`` hectares.

    Args:
        classified: array (H,W) uint8 {0,1} (255 pode representar NoData).
        resolution_m: resolução do pixel em metros.
        min_ha: unidade mínima de mapeamento em hectares.
    """
    pixel_area_ha = (resolution_m ** 2) / 1e4
    min_pixels = max(1, int(round(min_ha / pixel_area_ha)))
    mask = classified == 1
    labeled, n = ndimage.label(mask)
    if n == 0:
        return classified
    sizes = ndimage.sum(np.ones_like(labeled), labeled, index=np.arange(1, n + 1))
    remove = np.where(sizes < min_pixels)[0] + 1
    out = classified.copy()
    out[np.isin(labeled, remove)] = 0
    logger.info("MMU: removidos %d objetos < %.2f ha (%d px).",
                len(remove), min_ha, min_pixels)
    return out


def majority_filter(classified: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Filtro de maioria (moda) por janela quadrada ``kernel``."""
    from scipy.ndimage import generic_filter

    def _mode(v):
        vals, counts = np.unique(v, return_counts=True)
        return vals[np.argmax(counts)]

    out = generic_filter(classified.astype("float32"), _mode,
                         size=kernel, mode="nearest")
    return out.astype(classified.dtype)


def vectorize(
    classified: np.ndarray, transform, crs, min_ha: float, resolution_m: float,
    attrs: Optional[Dict[str, Any]] = None,
):
    """Vetoriza a classe caju para GeoDataFrame de polígonos (ETAPA 40)."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    mask = classified == 1
    geoms = []
    for geom, val in shapes(classified.astype("uint8"), mask=mask, transform=transform):
        if val == 1:
            geoms.append(shape(geom))
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if len(gdf):
        gdf["area_ha"] = gdf.geometry.area / 1e4
        gdf = gdf[gdf["area_ha"] >= min_ha].reset_index(drop=True)
        gdf["polygon_id"] = [f"poly_{i:06d}" for i in range(len(gdf))]
    if attrs:
        for k, v in attrs.items():
            gdf[k] = v
    logger.info("Vetorização: %d polígonos de caju.", len(gdf))
    return gdf
