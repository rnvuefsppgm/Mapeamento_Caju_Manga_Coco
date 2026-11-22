"""Composição temporal robusta (ETAPA 12 / questão single-date vs multitemporal).

DECISÃO para a Fase 1 (prova de conceito):
    Nem cena única, nem série temporal completa. Usamos DESCRITORES TEMPORAIS
    robustos por pixel dentro de uma janela sazonal:
        - mediana  -> composto robusto a nuvens residuais e speckle;
        - p25/p75  -> capturam variabilidade fenológica sem custo de série completa.
    Justificativa: cajueiro é perene com sazonalidade; a mediana reduz speckle do
    S1 (substitui filtro espacial) e nuvens residuais do S2, enquanto os
    percentis adicionam informação de dinâmica temporal barata. Série temporal
    completa fica para a Fase 2 (ConvLSTM/Transformers temporais).

Todos os redutores ignoram NaN (nanmedian/nanpercentile via skipna=True).
"""
from __future__ import annotations

import logging
from typing import List

import xarray as xr

logger = logging.getLogger(__name__)

_REDUCERS = {
    "mean": lambda da: da.mean(dim="time", skipna=True),
    "median": lambda da: da.median(dim="time", skipna=True),
    "min": lambda da: da.min(dim="time", skipna=True),
    "max": lambda da: da.max(dim="time", skipna=True),
    "std": lambda da: da.std(dim="time", skipna=True),
    "p10": lambda da: da.quantile(0.10, dim="time", skipna=True).drop_vars("quantile"),
    "p25": lambda da: da.quantile(0.25, dim="time", skipna=True).drop_vars("quantile"),
    "p75": lambda da: da.quantile(0.75, dim="time", skipna=True).drop_vars("quantile"),
    "p90": lambda da: da.quantile(0.90, dim="time", skipna=True).drop_vars("quantile"),
    "amplitude": lambda da: da.max(dim="time", skipna=True) - da.min(dim="time", skipna=True),
}


def temporal_composite(
    stack: xr.DataArray, reducers: List[str], min_observations: int = 3
) -> xr.DataArray:
    """Reduz o eixo temporal para descritores robustos, por banda.

    Args:
        stack: DataArray (time, band, y, x).
        reducers: lista de redutores (ex.: ['median','p25','p75']).
        min_observations: nº mínimo de observações válidas por pixel; abaixo disso
            o pixel vira NaN (evita compostos baseados em 1-2 cenas).

    Returns:
        DataArray (band, y, x) onde ``band`` = '<banda>_<redutor>'.
    """
    n_obs = (~stack.isnull()).sum(dim="time")
    enough = n_obs >= min_observations

    composites = []
    new_names = []
    for red in reducers:
        if red not in _REDUCERS:
            logger.warning("Redutor desconhecido ignorado: %s", red)
            continue
        reduced = _REDUCERS[red](stack)
        reduced = reduced.where(enough)  # invalida pixels com poucas observações
        for band_name in stack.band.values:
            new_names.append(f"{band_name}_{red}")
        composites.append(reduced)

    result = xr.concat(composites, dim="band")
    result = result.assign_coords(band=new_names)
    logger.info("Composto temporal: %d bandas de saída (redutores=%s).",
                len(new_names), reducers)
    return result
