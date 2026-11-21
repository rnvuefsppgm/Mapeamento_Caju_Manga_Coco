"""Features SAR derivadas de VV/VH + TEXTURA (ETAPA 8).

Duas famílias de features, calculadas sobre a mediana temporal de VV/VH (em dB,
já com o speckle atenuado pela composição temporal):

1) Radiométricas (por pixel):
   - Cuidado com a unidade (dB vs linear). Em dB, a razão vira DIFERENÇA
     (log(a/b)=log a - log b); por isso ``ratio_vh_vv`` = VH_dB - VV_dB.

2) TEXTURA (espacial, em janela local) — ajuda a separar o dossel estruturado do
   cajueiro de superfícies lisas (pasto, solo, água):
   - Estatísticas locais: desvio-padrão e amplitude (max-min) em janela.
   - GLCM/Haralick (implementação eficiente por diferenças deslocadas + média
     local, equivalente a P(i,j) agregada localmente):
       * contrast      = E[(i-j)^2]
       * dissimilarity = E[|i-j|]
       * homogeneity   = E[1/(1+(i-j)^2)]
     calculadas sobre a imagem quantizada e mediadas em 4 orientações
     (0°, 90°, 45°, 135°) para reduzir anisotropia.

Este módulo assume que o stack S1 recebido está em dB (padrão do pipeline).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import xarray as xr
from scipy import ndimage

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Texturas (numpy/scipy) — rápidas o suficiente para rasters grandes
# --------------------------------------------------------------------------- #
def _local_mean(a: np.ndarray, w: int) -> np.ndarray:
    return ndimage.uniform_filter(a, size=w, mode="reflect")


def _quantize(a: np.ndarray, levels: int, valid: np.ndarray) -> np.ndarray:
    """Quantiza a imagem (dB) em ``levels`` níveis, robusto a outliers (p1-p99)."""
    if not valid.any():
        return np.zeros_like(a, dtype="float32")
    lo, hi = np.percentile(a[valid], [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    q = np.clip((a - lo) / (hi - lo), 0.0, 1.0) * (levels - 1)
    return np.rint(np.nan_to_num(q, nan=0.0)).astype("float32")


def _texture_for_pol(
    arr2d: np.ndarray, w: int, levels: int,
    want_stats: List[str], want_glcm: List[str],
) -> Dict[str, np.ndarray]:
    """Calcula as features de textura de uma polarização (array 2D em dB)."""
    valid = np.isfinite(arr2d)
    fill = float(np.nanmean(arr2d)) if valid.any() else 0.0
    filled = np.nan_to_num(arr2d, nan=fill).astype("float32")
    feats: Dict[str, np.ndarray] = {}

    # --- estatísticas locais ---
    if "std" in want_stats:
        m = _local_mean(filled, w)
        msq = _local_mean(filled * filled, w)
        feats["std"] = np.sqrt(np.clip(msq - m * m, 0.0, None))
    if "range" in want_stats:
        feats["range"] = (ndimage.maximum_filter(filled, size=w, mode="reflect")
                          - ndimage.minimum_filter(filled, size=w, mode="reflect"))

    # --- GLCM (Haralick) eficiente por diferenças deslocadas ---
    if want_glcm:
        Q = _quantize(arr2d, levels, valid)
        offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 0°, 90°, 45°, 135°
        con = np.zeros_like(Q); dis = np.zeros_like(Q); hom = np.zeros_like(Q)
        for dy, dx in offsets:
            Qs = np.roll(np.roll(Q, -dy, axis=0), -dx, axis=1)
            d = Q - Qs
            con += _local_mean(d * d, w)
            dis += _local_mean(np.abs(d), w)
            hom += _local_mean(1.0 / (1.0 + d * d), w)
        n = float(len(offsets))
        if "contrast" in want_glcm:
            feats["contrast"] = con / n
        if "dissimilarity" in want_glcm:
            feats["dissimilarity"] = dis / n
        if "homogeneity" in want_glcm:
            feats["homogeneity"] = hom / n

    # invalida (NaN) onde não havia dado
    for k in list(feats):
        feats[k] = np.where(valid, feats[k], np.nan).astype("float32")
    return feats


# --------------------------------------------------------------------------- #
# API principal
# --------------------------------------------------------------------------- #
def compute_sar_features(s1_db: xr.DataArray, cfg: Dict[str, Any]) -> xr.DataArray:
    """Calcula features SAR (razões/diferenças + textura) a partir de VV/VH em dB.

    Args:
        s1_db: DataArray (band, y, x) com bandas 'vv','vh' em dB (já composto).
        cfg: configuração completa. Usa ``cfg['sar_features']`` (lista de razões)
            e, se presente, ``cfg['sar_texture']`` (bloco de textura).

    Returns:
        DataArray (band, y, x) com as features nomeadas (band = nome da feature).
    """
    features = cfg.get("sar_features", []) or []
    tcfg = cfg.get("sar_texture", {}) or {}

    vv = s1_db.sel(band="vv")
    vh = s1_db.sel(band="vh")

    layers: List[xr.DataArray] = []
    names: List[str] = []

    # 1) radiométricas (dB): razão em linear = diferença em dB
    available = {
        "ratio_vh_vv": vh - vv,   # ~ log(VH/VV): espalhamento de volume
        "diff_vv_vh": vv - vh,    # inverso do acima
        "cross_diff": vh - vv,    # diferença cross-pol bruta
    }
    for f in features:
        if f not in available:
            logger.warning("Feature SAR desconhecida ignorada: %s", f)
            continue
        layers.append(available[f])
        names.append(f)

    # 2) textura (estatísticas locais + GLCM), por polarização
    if tcfg.get("enabled", False):
        w = int(tcfg.get("window", 5))
        levels = int(tcfg.get("glcm_levels", 32))
        want_stats = list(tcfg.get("local_stats", []) or [])
        want_glcm = list(tcfg.get("glcm", []) or [])
        for pol, band in (("vv", vv), ("vh", vh)):
            feats = _texture_for_pol(band.values, w, levels, want_stats, want_glcm)
            for key, arr in feats.items():
                da = xr.DataArray(
                    arr, dims=band.dims,
                    coords={d: band.coords[d] for d in band.dims if d in band.coords},
                )
                layers.append(da)
                names.append(f"{pol}_{key}")

    if not layers:
        return xr.DataArray()

    # Limpa coords escalares herdadas e fixa o nome de cada feature como banda,
    # evitando o erro "coordinate ... not present in all datasets" no concat.
    cleaned = [
        l.drop_vars([c for c in l.coords if c not in l.dims], errors="ignore")
         .expand_dims(band=[n])
        for l, n in zip(layers, names)
    ]
    result = xr.concat(cleaned, dim="band")
    logger.info("Features SAR (%d, incl. textura): %s", len(names), names)
    return result
