"""Rasterização de labels de caju e geração de background (ETAPAS 18-19).

Diferença crítica (PONTO CRÍTICO SOBRE AS AMOSTRAS):
    Só rasterize como caju (=1) polígonos que representem plantações CONFIRMADAS.
    Se o vetor for apenas AOI de download, ele NÃO é label. Os scripts controlam
    isso via ``--role {ground_truth|aoi}``; aqui apenas implementamos a
    rasterização quando o papel é ground truth.

Erosão de borda:
    Polígonos digitalizados têm imprecisão geométrica; pixels de borda são mistos.
    Aplicamos erosão opcional para converter a moldura de borda em classe
    "ignore" (255), que a loss desconsidera — reduz ruído de rótulo sem perder o
    núcleo do alvo.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage

logger = logging.getLogger(__name__)


def rasterize_labels(
    gdf: gpd.GeoDataFrame,
    out_shape: Tuple[int, int],
    transform: Affine,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Rasteriza polígonos de caju em uma máscara alinhada ao grid de features.

    Args:
        gdf: polígonos de caju CONFIRMADO, no MESMO CRS do grid de features.
        out_shape: (height, width) do grid.
        transform: affine do grid (mesmo do composto de features).
        cfg: configuração (valores de classe, erosão).

    Returns:
        Máscara uint8 com valores {0 background, 1 caju, 255 ignore}.
    """
    pos = cfg["chips"]["label_positive_value"]
    bg = cfg["chips"]["label_background_value"]
    ignore = cfg["chips"]["label_ignore_value"]
    erosion_px = cfg["chips"]["edge_erosion_px"]

    shapes = ((geom, pos) for geom in gdf.geometry if geom is not None)
    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=bg,
        dtype="uint8",
        all_touched=False,
    )

    if erosion_px > 0:
        core = ndimage.binary_erosion(
            mask == pos, iterations=erosion_px, border_value=0
        )
        border = (mask == pos) & (~core)
        mask[border] = ignore  # moldura de borda vira "ignore"
        logger.info("Erosão de borda: %d px marcados como ignore.", int(border.sum()))

    logger.info("Labels rasterizadas: caju=%d, bg=%d, ignore=%d px.",
                int((mask == pos).sum()), int((mask == bg).sum()),
                int((mask == ignore).sum()))
    return mask


def mark_nodata_as_ignore(
    label: np.ndarray, valid_mask: np.ndarray, ignore_value: int = 255
) -> np.ndarray:
    """Marca como 'ignore' pixels sem dado de imagem (NoData nas features)."""
    label = label.copy()
    label[~valid_mask] = ignore_value
    return label
