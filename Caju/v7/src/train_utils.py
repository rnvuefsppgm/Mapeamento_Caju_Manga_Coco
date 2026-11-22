"""Utilidades de treino: seed, AMP, early stopping, checkpoint, threshold (ETAPA 27).

Suporta GPU (CUDA) com fallback para CPU, mixed precision, scheduler, retomada
via checkpoint e busca de threshold ótimo no conjunto de VALIDAÇÃO (ETAPA 36).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False


def set_seed(seed: int) -> None:
    """Fixa sementes para reprodutibilidade (ETAPA 27)."""
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Retorna 'cuda' se disponível, senão 'cpu' (fallback)."""
    if _TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_checkpoint(path: Path, model, optimizer, epoch: int,
                    best_metric: float, cfg: Dict[str, Any]) -> None:
    """Salva estado completo para retomar treino."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": cfg,
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, map_location: str = "cpu"):
    """Carrega checkpoint; devolve (epoch, best_metric)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", -1.0)


class EarlyStopping:
    """Para o treino se a métrica monitorada não melhora por ``patience`` épocas."""

    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def find_best_threshold(
    probs: np.ndarray, targets: np.ndarray, cfg: Dict[str, Any],
    ignore_index: int = 255,
) -> Tuple[float, Dict[str, float]]:
    """Busca o threshold ótimo no conjunto de VALIDAÇÃO (ETAPA 36).

    Critério configurável: maior F1, maior IoU ou melhor recall sob precisão
    mínima. Nunca usa o conjunto de teste.
    """
    from .metrics import SegMeter

    criterion = cfg["inference"]["threshold_criterion"]
    min_prec = cfg["inference"].get("min_precision", 0.8)
    best_t, best_score, best_metrics = 0.5, -1.0, {}
    for t in np.linspace(0.05, 0.95, 19):
        meter = SegMeter(ignore_index)
        meter.update(probs, targets, threshold=float(t))
        m = meter.compute()
        if criterion == "f1":
            score = m["f1_cashew"]
        elif criterion == "iou":
            score = m["iou_cashew"]
        else:  # min_precision -> maximiza recall sob precisão mínima
            score = m["recall_cashew"] if m["precision_cashew"] >= min_prec else -1.0
        if score > best_score:
            best_score, best_t, best_metrics = score, float(t), m
    logger.info("Threshold ótimo (%s) = %.2f | %s", criterion, best_t, best_metrics)
    return best_t, best_metrics
