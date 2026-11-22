"""Inferência por janela deslizante com blending (ETAPA 35).

Reconstrói o mapa de probabilidade do município a partir de chips sobrepostos.
O blending gaussiano pondera mais o centro de cada chip, reduzindo artefatos de
costura nas bordas — problema clássico de sliding-window em segmentação.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


def _gaussian_weight(size: int, sigma_frac: float = 0.25) -> np.ndarray:
    """Kernel gaussiano 2D (size x size) para ponderar chips no blending."""
    ax = np.linspace(-1, 1, size)
    gx = np.exp(-(ax ** 2) / (2 * sigma_frac ** 2))
    w = np.outer(gx, gx)
    return (w / w.max()).astype("float32")


def sliding_window_predict(
    features: np.ndarray,          # (C,H,W) já normalizado, NaN->0
    valid_mask: np.ndarray,        # (H,W) bool: onde há dado
    model,
    cfg: Dict[str, Any],
    device: str = "cpu",
) -> np.ndarray:
    """Gera o mapa de probabilidade P(caju) para um raster inteiro.

    Returns:
        Array float32 (H,W) com probabilidades em [0,1]; NoData onde ~valid_mask.
    """
    if not _TORCH:
        raise RuntimeError("PyTorch não está instalado.")
    size = cfg["inference"]["chip_size"]
    overlap = cfg["inference"]["overlap"]
    stride = size - overlap
    blending = cfg["inference"]["blending"]
    batch_size = cfg["inference"]["batch_size"]

    c, h, w = features.shape
    prob_acc = np.zeros((h, w), dtype="float32")
    weight_acc = np.zeros((h, w), dtype="float32")
    weight = _gaussian_weight(size) if blending == "gaussian" else np.ones((size, size), "float32")

    coords = []
    for r in range(0, max(1, h - size + 1), stride):
        for cc in range(0, max(1, w - size + 1), stride):
            coords.append((r, cc))
    if (h - size) % stride != 0:
        coords += [(h - size, cc) for cc in range(0, max(1, w - size + 1), stride)]
    if (w - size) % stride != 0:
        coords += [(r, w - size) for r in range(0, max(1, h - size + 1), stride)]

    model.eval()
    model.to(device)
    n_batches = (len(coords) + batch_size - 1) // batch_size
    with torch.no_grad():
        # Barra de progresso: janelas deslizantes deste raster (não persiste).
        for i in tqdm(range(0, len(coords), batch_size), total=n_batches,
                      desc="Janelas", unit="batch", leave=False):
            batch_coords = coords[i:i + batch_size]
            batch = np.stack([features[:, r:r + size, cc:cc + size]
                              for r, cc in batch_coords])
            x = torch.from_numpy(batch).float().to(device)
            logits = model(x)
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            for (r, cc), p in zip(batch_coords, probs):
                prob_acc[r:r + size, cc:cc + size] += p * weight
                weight_acc[r:r + size, cc:cc + size] += weight

    weight_acc[weight_acc == 0] = 1.0
    prob = prob_acc / weight_acc
    prob[~valid_mask] = np.nan
    logger.info("Inferência sliding-window concluída (%d janelas).", len(coords))
    return prob.astype("float32")


def probability_entropy(prob: np.ndarray) -> np.ndarray:
    """Mapa de incerteza por entropia binária (ETAPA 37).

    Alta entropia (~0.5 de probabilidade) => baixa confiança => candidato a
    active learning.
    """
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return ent.astype("float32")
