"""Pré-processamento Sentinel-2 L2A (reflectância de superfície).

Decisões metodológicas implementadas:
    * Máscara de qualidade baseada na SCL (Scene Classification Layer), não apenas
      no ``eo:cloud_cover`` do tile (ETAPA 6).
    * Harmonização do offset de baseline >= 04.00: dados processados após
      ~2022-01-25 têm ``BOA_ADD_OFFSET = -1000`` embutido; para uma série
      temporal consistente, subtrai-se 1000 dos DN dessas cenas (correção POR
      CENA, aplicada a TODOS os pixels da cena) antes de aplicar a escala.
      (Sem isso, cenas antigas e novas ficam em referenciais diferentes.)
    * Fração de pixels VÁLIDOS calculada DENTRO da AOI, não no tile inteiro.

Escala: reflectância = (DN + offset) * 0.0001, com clip em [0, 1].
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import stackstac
import xarray as xr

logger = logging.getLogger(__name__)

# Data de corte da Processing Baseline 04.00 do Sentinel-2 L2A: cenas com
# datetime a partir daqui trazem o BOA_ADD_OFFSET (-1000) embutido nos DN.
_BASELINE_04_CUTOFF = np.datetime64("2022-01-25")


def load_s2_stack(
    items: List[Any],
    bands: List[str],
    bounds: tuple[float, float, float, float],
    epsg: int,
    resolution: float,
    scl_asset: str = "SCL",
) -> xr.DataArray:
    """Carrega um stack S2 (lazy, via stackstac) recortado ao grid de referência.

    O stack é criado JÁ no CRS/resolução/bounds finais (10 m), garantindo o
    alinhamento espacial com o S1. A leitura é remota (COG + range requests);
    nenhum tile completo é baixado (ETAPA 15, opção B/C).

    Args:
        items: STAC Items S2 (assinados).
        bands: bandas espectrais a carregar (ex.: ['B02', ...]).
        bounds: (minx, miny, maxx, maxy) no CRS de destino.
        epsg: código EPSG do grid de destino.
        resolution: resolução em metros.
        scl_asset: nome do asset SCL.

    Returns:
        DataArray (time, band, y, x) com bandas + SCL, ainda em DN (int).
    """
    assets = list(bands) + [scl_asset]
    stack = stackstac.stack(
        items,
        assets=assets,
        epsg=epsg,
        resolution=resolution,
        bounds=bounds,
        dtype="float32",
        # NaN em float32 (e não np.nan=float64): o stackstac 0.5.1 valida
        # np.can_cast(type(fill_value), dtype); float64 -> float32 daria erro.
        fill_value=np.float32(np.nan),
        rescale=False,      # NÃO aplicar rescale automático; controlamos manualmente
        chunksize=2048,
    )
    logger.info("Stack S2: %s", dict(stack.sizes))
    return stack


def apply_scl_mask_and_scale(
    stack: xr.DataArray,
    bands: List[str],
    cfg: Dict[str, Any],
    scl_asset: str = "SCL",
) -> xr.DataArray:
    """Mascara pixels inválidos via SCL, harmoniza baseline e converte a reflectância.

    Returns:
        DataArray (time, band, y, x) apenas com ``bands``, em reflectância [0,1],
        com pixels inválidos = NaN.
    """
    s2cfg = cfg["sentinel2"]
    invalid = s2cfg["scl_invalid_classes"]
    scale = s2cfg["scale"]
    offset = s2cfg["boa_offset_baseline_04"]

    scl = stack.sel(band=scl_asset)
    valid = ~scl.isin(invalid)              # True onde pixel é utilizável

    optical = stack.sel(band=bands)

    # Harmonização de baseline >= 04.00 (CORREÇÃO metodológica): o BOA_ADD_OFFSET
    # (-1000) é uma propriedade da CENA (Processing Baseline), não do pixel.
    # Aplica-se o offset a TODOS os pixels das cenas processadas a partir de
    # ~2022-01-25 (uniforme por cena), e não condicionalmente por valor de DN.
    # Reflectâncias negativas resultantes (pixels escuros) são clipadas a 0.
    if "time" in optical.coords:
        post_baseline = optical["time"] >= _BASELINE_04_CUTOFF
        per_scene_offset = xr.where(post_baseline, offset, 0.0)
        n_post = int(np.asarray(post_baseline).sum())
        logger.info("Harmonização baseline 04.00: offset %s aplicado a %d cena(s) "
                    "pós-2022 (uniforme por cena).", offset, n_post)
    else:
        # Sem dimensão temporal explícita: assume janela pós-baseline (offset em tudo).
        per_scene_offset = offset
        logger.info("Harmonização baseline 04.00: offset %s aplicado (sem eixo "
                    "temporal; assumido pós-2022).", offset)

    optical = optical + per_scene_offset
    reflectance = (optical * scale).clip(0.0, 1.0)
    reflectance = reflectance.where(valid)   # aplica máscara (invalida -> NaN)

    logger.info("S2: SCL aplicada (classes inválidas=%s) e reflectância escalada.",
                invalid)
    return reflectance


def valid_pixel_fraction(masked: xr.DataArray) -> xr.DataArray:
    """Fração de pixels válidos por cena (média sobre band/y/x).

    Base para selecionar cenas com qualidade suficiente DENTRO da AOI (ETAPA 6).
    """
    ref_band = masked.isel(band=0)
    frac = (~ref_band.isnull()).mean(dim=["y", "x"])
    return frac


def select_valid_scenes(
    masked: xr.DataArray, min_valid_fraction: float
) -> xr.DataArray:
    """Mantém apenas cenas cuja fração de pixels válidos na AOI >= limiar."""
    frac = valid_pixel_fraction(masked).compute()
    # Máscara booleana PURA (numpy) + indexação POR POSIÇÃO (isel): evita o
    # conflito de coordenada 'band' que ocorre ao indexar com um DataArray
    # booleano que herdou a coord escalar 'band' de valid_pixel_fraction.
    keep = np.asarray(frac.values) >= min_valid_fraction
    n_keep = int(keep.sum())
    n_total = masked.sizes.get("time", int(keep.size))
    logger.info("S2: %d/%d cenas com fração válida >= %.2f.",
                n_keep, n_total, min_valid_fraction)
    if n_keep == 0:
        logger.warning("Nenhuma cena S2 válida na AOI para a janela temporal!")
    return masked.isel(time=np.where(keep)[0])
