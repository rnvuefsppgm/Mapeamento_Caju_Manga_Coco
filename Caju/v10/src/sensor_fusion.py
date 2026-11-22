"""Fusão multissensor óptico-SAR (ETAPA 14).

DECISÃO para a Fase 1: EARLY FUSION como baseline multissensor.
    Empilhamos, no eixo de bandas, todas as camadas já harmonizadas espacialmente
    (mesmo grid 10 m, mesmo CRS, mesmos bounds):
        [ S2 bandas compostas | índices | S1 VV/VH compostos | SAR features ]
    Vantagens: simples, reprodutível, sem duplicar encoders, funciona bem com o
    volume de amostras esperado numa PoC. A comparação de modalidades (E1/E2/E3)
    é controlada por ``modality`` no config — o mesmo código monta S2-only,
    S1-only ou fusão apenas selecionando quais grupos entram na pilha.

Mid/late fusion ficam para a Fase 2 (encoders separados / ensemble de probs).

Este módulo NÃO lê nuvem/rede: recebe DataArrays já compostos e os concatena,
garantindo grid idêntico. A ordem/nomes das bandas é registrada para reprodução.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _align(reference: xr.DataArray, other: xr.DataArray) -> xr.DataArray:
    """Garante que ``other`` compartilhe exatamente o grid (y,x) da referência."""
    if other.sizes.get("y") != reference.sizes.get("y") or \
       other.sizes.get("x") != reference.sizes.get("x"):
        other = other.interp(y=reference.y, x=reference.x, method="nearest")
    return other.assign_coords(y=reference.y, x=reference.x)


def build_feature_stack(
    modality: str,
    s2_bands: Optional[xr.DataArray] = None,
    s2_indices: Optional[xr.DataArray] = None,
    s1_bands: Optional[xr.DataArray] = None,
    sar_features: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Monta a pilha de features conforme a modalidade do experimento.

    Args:
        modality: 's2' | 's1' | 'fusion'.
        s2_bands: composto de bandas S2 (band,y,x).
        s2_indices: composto de índices (band,y,x) ou None.
        s1_bands: composto de VV/VH (band,y,x).
        sar_features: features SAR derivadas (band,y,x) ou None.

    Returns:
        DataArray (band, y, x) com todas as camadas selecionadas, e atributo
        ``.attrs['band_names']`` com a ordem canônica.
    """
    layers: List[xr.DataArray] = []

    if modality in ("s2", "fusion"):
        if s2_bands is None:
            raise ValueError("modality inclui S2 mas s2_bands é None.")
        layers.append(s2_bands)
        if s2_indices is not None and s2_indices.size > 0:
            layers.append(_align(s2_bands, s2_indices))

    if modality in ("s1", "fusion"):
        if s1_bands is None:
            raise ValueError("modality inclui S1 mas s1_bands é None.")
        ref = layers[0] if layers else s1_bands
        layers.append(_align(ref, s1_bands))
        if sar_features is not None and sar_features.size > 0:
            layers.append(_align(ref, sar_features))

    if not layers:
        raise ValueError(f"Modalidade inválida: {modality}")

    # Mantém apenas as coordenadas de DIMENSÃO (band/y/x) em cada camada. As
    # camadas brutas (S2/S1 via stackstac) trazem coords não-dimensionais como
    # 'description', 'title', 'proj:code', 'epsg'... que NÃO existem nas camadas
    # derivadas (índices/SAR); sem esta limpeza, o concat falha com
    # "coordinate '<nome>' not present in all datasets".
    def _only_dim_coords(da: xr.DataArray) -> xr.DataArray:
        drop = [c for c in da.coords if c not in da.dims]
        return da.drop_vars(drop, errors="ignore")

    layers = [_only_dim_coords(l) for l in layers]
    stack = xr.concat(layers, dim="band")
    band_names = [str(b) for b in stack.band.values]
    stack.attrs["band_names"] = band_names
    stack.attrs["modality"] = modality
    logger.info("Fusão (%s): %d canais -> %s", modality, len(band_names), band_names)
    return stack


def to_numpy_chw(stack: xr.DataArray) -> np.ndarray:
    """Converte o stack (band,y,x) para array numpy (C,H,W) float32.

    NaNs são preservados aqui; o tratamento (fill/normalização) ocorre no dataset.
    """
    arr = stack.transpose("band", "y", "x").values.astype("float32")
    return arr
