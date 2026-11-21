"""Pré-processamento Sentinel-1 RTC (gamma0, terreno-corrigido).

O que JÁ vem aplicado no produto ``sentinel-1-rtc`` do Planetary Computer
(validado na doc oficial — ver README/ETAPA 9). Marcação SIM/NÃO:

    Orbit correction ............... SIM (não refazer)
    Thermal noise removal .......... SIM (não refazer)
    Radiometric calibration ........ SIM -> gamma0 (não refazer)
    Border noise removal ........... SIM (não refazer)
    Terrain correction (RTC) ....... SIM (Copernicus DEM) (não refazer)
    Escala ......................... LINEAR de potência (gamma0), NÃO em dB

    Speckle filtering .............. NÃO -> tratado por COMPOSIÇÃO TEMPORAL (ETAPA 10)
    Conversão para dB .............. NÃO -> feita aqui, opcionalmente (to_db)

DECISÃO (ETAPA 10): em vez de filtro espacial de speckle (Lee/Refined Lee etc.),
que borra bordas e pequenas plantações, reduzimos o speckle pela MEDIANA temporal
multitemporal — mais adequado a alvos pequenos e preserva feições estruturais.
"""
from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import stackstac
import xarray as xr

logger = logging.getLogger(__name__)


def load_s1_stack(
    items: List[Any],
    polarizations: List[str],
    bounds: tuple[float, float, float, float],
    epsg: int,
    resolution: float,
) -> xr.DataArray:
    """Carrega um stack S1 RTC (lazy) no mesmo grid do S2 (alinhamento espacial).

    Reamostrar o RTC (~10-20 m nativo) para o grid comum de 10 m usa reamostragem
    bilinear e NÃO cria informação nova — apenas garante coincidência de células
    com o S2 (ETAPA 15).

    Returns:
        DataArray (time, band, y, x) com polarizações em POTÊNCIA LINEAR.
    """
    stack = stackstac.stack(
        items,
        assets=polarizations,
        epsg=epsg,
        resolution=resolution,
        bounds=bounds,
        dtype="float32",
        # NaN em float32 (e não np.nan=float64): o stackstac 0.5.1 valida
        # np.can_cast(type(fill_value), dtype); float64 -> float32 daria erro.
        fill_value=np.float32(np.nan),
        rescale=False,
        chunksize=2048,
    )
    logger.info("Stack S1: %s", dict(stack.sizes))
    return stack


def linear_to_db(da: xr.DataArray, floor: float = 1e-6) -> xr.DataArray:
    """Converte potência linear (gamma0) para decibéis: 10*log10(x).

    Aplica um piso para evitar log(0)/log(negativo). Valores <= 0 (artefatos de
    reamostragem) são tratados como NaN.
    """
    da = da.where(da > 0)
    return 10.0 * np.log10(da.clip(min=floor))


def preprocess_s1(
    stack: xr.DataArray, to_db: bool = True
) -> xr.DataArray:
    """Aplica (opcionalmente) conversão para dB e devolve o stack pronto.

    DECISÃO metodológica: usar dB para VV/VH torna as distribuições mais próximas
    de gaussianas (melhor para normalização z-score/robusta e para a CNN). As
    features derivadas (razões) são calculadas em ``sar_features`` levando em conta
    a unidade escolhida aqui.
    """
    if to_db:
        db = linear_to_db(stack)
        logger.info("S1: convertido para dB (10*log10).")
        return db
    logger.info("S1: mantido em potência linear.")
    return stack
