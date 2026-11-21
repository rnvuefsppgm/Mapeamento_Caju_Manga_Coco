"""Losses e métricas de segmentação (ETAPAS 29-30), focadas na classe caju.

Loss recomendada para a PoC: BCE + Dice.
    * BCE estabiliza o treino; Dice combate o forte desbalanceamento caju/fundo.
    * Focal+Dice é alternativa para desbalanceamento extremo.
Pixels com label == ignore_value (255) são desconsiderados em loss e métricas.

Métricas priorizam a classe caju: IoU/F1/Precision/Recall do positivo.
Overall Accuracy NÃO é usada isoladamente (inflada pelo fundo dominante).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
    _Module = nn.Module
except Exception:  # pragma: no cover
    _TORCH = False

    class _Module:  # fallback: permite importar o módulo sem torch (só p/ SegMeter)
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise RuntimeError("PyTorch não está instalado; losses indisponíveis.")


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class DiceLoss(_Module):  # type: ignore
    def __init__(self, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        prob = torch.sigmoid(logits.squeeze(1))
        valid = target != self.ignore_index
        t = (target == 1).float() * valid
        p = prob * valid
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(_Module):  # type: ignore
    """BCE mascarada + Dice (padrão recomendado)."""

    def __init__(self, ignore_index: int = 255, pos_weight: float | None = None) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.dice = DiceLoss(ignore_index)
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        logits1 = logits.squeeze(1)
        valid = (target != self.ignore_index)
        t = (target == 1).float()
        pw = (torch.tensor(self.pos_weight, device=logits.device)
              if self.pos_weight else None)
        bce = F.binary_cross_entropy_with_logits(
            logits1, t, reduction="none", pos_weight=pw)
        bce = (bce * valid).sum() / valid.sum().clamp(min=1)
        return bce + self.dice(logits, target)


class FocalDiceLoss(_Module):  # type: ignore
    def __init__(self, ignore_index: int = 255, gamma: float = 2.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.dice = DiceLoss(ignore_index)

    def forward(self, logits, target):
        logits1 = logits.squeeze(1)
        valid = (target != self.ignore_index)
        t = (target == 1).float()
        bce = F.binary_cross_entropy_with_logits(logits1, t, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma * bce)
        focal = (focal * valid).sum() / valid.sum().clamp(min=1)
        return focal + self.dice(logits, target)


def get_loss(name: str, ignore_index: int = 255):
    """Fábrica de loss por nome (config ``training.loss``)."""
    name = name.lower()
    if name in ("bce_dice", "bcedice"):
        return BCEDiceLoss(ignore_index)
    if name == "dice":
        return DiceLoss(ignore_index)
    if name in ("focal_dice", "focaldice"):
        return FocalDiceLoss(ignore_index)
    if name in ("bce", "wbce"):
        return BCEDiceLoss(ignore_index, pos_weight=5.0 if name == "wbce" else None)
    raise ValueError(f"Loss não suportada: {name}")


# --------------------------------------------------------------------------- #
# Métricas (numpy, acumuláveis)
# --------------------------------------------------------------------------- #
class SegMeter:
    """Acumula TP/FP/FN/TN para a classe caju e computa métricas ao final."""

    def __init__(self, ignore_index: int = 255) -> None:
        self.ignore_index = ignore_index
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> None:
        valid = target != self.ignore_index
        pred = (prob >= threshold) & valid
        pos = (target == 1) & valid
        self.tp += int(np.sum(pred & pos))
        self.fp += int(np.sum(pred & ~pos & valid))
        self.fn += int(np.sum(~pred & pos))
        self.tn += int(np.sum(~pred & ~pos & valid))

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)           # = Producer's Accuracy do caju
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        users_acc = precision                    # User's Accuracy do caju
        return {
            "precision_cashew": precision,
            "recall_cashew": recall,
            "f1_cashew": f1,
            "iou_cashew": iou,
            "dice_cashew": dice,
            "producers_acc": recall,
            "users_acc": users_acc,
        }
