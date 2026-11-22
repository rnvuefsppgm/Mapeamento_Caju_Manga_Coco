"""Índices espectrais Sentinel-2 para diferenciação de cajueiros.

Cada índice foi escolhido por justificativa biofísica (ETAPA 7), não para inflar
dimensionalidade. Fórmulas usam reflectância [0,1]. Bandas S2 (10-20 m):
    B02 Blue, B03 Green, B04 Red, B05/B06/B07 Red Edge, B08 NIR (10m),
    B8A NIR narrow, B11 SWIR1, B12 SWIR2.

Justificativas resumidas:
    NDVI  (B08,B04) — vigor/biomassa verde; separa vegetação de solo/água.
    NDRE  (B08,B05) — Red Edge: sensível a clorofila em dossel denso; cajueiro
                       perene tende a saturar menos no NDRE que no NDVI.
    NDMI  (B08,B11) — conteúdo de água foliar; discrimina dossel arbóreo úmido
                       (cajueiro) de pastagem/agricultura sazonalmente seca.
    EVI   (B08,B04,B02) — reduz saturação e efeito de solo/atmosfera em dossel denso.
    SAVI  (B08,B04)     — corrige influência do solo em coberturas esparsas.
    NBR   (B08,B12)     — sensível a estrutura/senescência; ajuda a separar
                       silvicultura/queimadas de cultivo perene.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import xarray as xr

logger = logging.getLogger(__name__)

_L_SAVI = 0.5  # fator de ajuste de solo do SAVI


def _b(da: xr.DataArray, name: str) -> xr.DataArray:
    return da.sel(band=name)


def compute_indices(reflectance: xr.DataArray, indices: List[str]) -> xr.DataArray:
    """Calcula os índices pedidos a partir de um composto de reflectância.

    Args:
        reflectance: DataArray (band, y, x) de reflectância (já composto no tempo).
        indices: lista de nomes (ex.: ['NDVI','NDRE',...]).

    Returns:
        DataArray (band, y, x) onde ``band`` são os nomes dos índices.
    """
    funcs: Dict[str, callable] = {
        "NDVI": lambda d: _norm_diff(_b(d, "B08"), _b(d, "B04")),
        "NDRE": lambda d: _norm_diff(_b(d, "B08"), _b(d, "B05")),
        "GNDVI": lambda d: _norm_diff(_b(d, "B08"), _b(d, "B03")),
        "NDMI": lambda d: _norm_diff(_b(d, "B08"), _b(d, "B11")),
        "NDWI": lambda d: _norm_diff(_b(d, "B03"), _b(d, "B08")),
        "NBR": lambda d: _norm_diff(_b(d, "B08"), _b(d, "B12")),
        "SAVI": lambda d: _savi(_b(d, "B08"), _b(d, "B04")),
        "EVI": lambda d: _evi(_b(d, "B08"), _b(d, "B04"), _b(d, "B02")),
        # --- v3: índices adicionais p/ separar melhor o cajueiro ---
        # Red-edge (clorofila do dossel perene: cajueiro tende a se destacar):
        "NDRE2": lambda d: _norm_diff(_b(d, "B08"), _b(d, "B06")),
        "CIre": lambda d: _ratio(_b(d, "B08"), _b(d, "B05")) - 1.0,   # Chlorophyll Index red-edge
        # Umidade/água foliar (dossel arbóreo úmido vs pasto/solo seco):
        "MSI": lambda d: _ratio(_b(d, "B11"), _b(d, "B08")),          # Moisture Stress Index
        # Vigor por razão simples (realça contraste vegetação x fundo):
        "RVI": lambda d: _ratio(_b(d, "B08"), _b(d, "B04")),         # Ratio Vegetation Index
    }
    out = []
    names = []
    for name in indices:
        if name not in funcs:
            logger.warning("Índice desconhecido ignorado: %s", name)
            continue
        # reset_coords(drop=True): remove coordenadas escalares herdadas das
        # bandas de origem (ex.: o EVI retém 'band'='B02'), que fariam o concat
        # falhar ("coordinate 'band' not present in all datasets"). O
        # expand_dims fixa o índice como a banda de saída com o nome correto.
        arr = funcs[name](reflectance).reset_coords(drop=True).expand_dims(band=[name])
        out.append(arr)
        names.append(name)
    result = xr.concat(out, dim="band")
    logger.info("Índices espectrais calculados: %s", names)
    return result


def _norm_diff(a: xr.DataArray, b: xr.DataArray, eps: float = 1e-6) -> xr.DataArray:
    return (a - b) / (a + b + eps)


def _ratio(a: xr.DataArray, b: xr.DataArray, eps: float = 1e-6) -> xr.DataArray:
    """Razão simples a/b (com piso no denominador para estabilidade)."""
    return a / (b + eps)


def _savi(nir: xr.DataArray, red: xr.DataArray, eps: float = 1e-6) -> xr.DataArray:
    return ((nir - red) / (nir + red + _L_SAVI + eps)) * (1 + _L_SAVI)


def _evi(nir: xr.DataArray, red: xr.DataArray, blue: xr.DataArray) -> xr.DataArray:
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1.0)
