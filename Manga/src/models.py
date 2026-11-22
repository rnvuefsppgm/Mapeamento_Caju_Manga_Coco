"""Construção de modelos de segmentação (ETAPA 24).

Recomendação da PoC:
    * PRINCIPAL:  U-Net (encoder ResNet-34) via segmentation-models-pytorch.
      Motivo: robusta, fácil de treinar, ótima com poucos-médios dados, aceita
      qualquer nº de canais (ideal para fusão), baixa VRAM.
    * BENCHMARK 1: U-Net++ (mesmo encoder) — skip connections densas, ganho em
      bordas/alvos pequenos ao custo de +VRAM.
    * BENCHMARK 2: DeepLabV3+ — contexto multiescala (ASPP), bom p/ heterogeneidade.
    * BASELINE tradicional: Random Forest por pixel (ver ml_baseline.py).

Canais > 3: o encoder pré-treinado (ImageNet, RGB) tem conv de entrada de 3
canais. O smp adapta automaticamente ``in_channels``; os pesos RGB são repetidos/
médios para os canais extras. Isso dá um "warm start" imperfeito porém útil.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import segmentation_models_pytorch as smp
    import torch.nn as nn
    _SMP = True
except Exception:  # pragma: no cover
    _SMP = False


_ARCH = {
    "unet": "Unet",
    "unetpp": "UnetPlusPlus",
    "deeplabv3plus": "DeepLabV3Plus",
}


def build_model(cfg: Dict[str, Any], in_channels: int):
    """Instancia o modelo de segmentação conforme o config.

    Args:
        cfg: configuração.
        in_channels: nº de canais de entrada (depende da modalidade/features).

    Returns:
        ``nn.Module`` de segmentação (saída: logits, 1 canal).
    """
    if not _SMP:
        raise RuntimeError("segmentation-models-pytorch não está instalado.")
    m = cfg["model"]
    arch_key = m["architecture"]
    if arch_key not in _ARCH:
        raise ValueError(f"Arquitetura não suportada: {arch_key}")
    arch_cls = getattr(smp, _ARCH[arch_key])

    model = arch_cls(
        encoder_name=m["encoder"],
        encoder_weights=m.get("encoder_weights") or None,
        in_channels=in_channels,
        classes=m["num_classes"],
        activation=None,  # devolve logits; sigmoid aplicado na loss/inferência
    )
    logger.info("Modelo: %s | encoder=%s | in_channels=%d | classes=%d",
                arch_key, m["encoder"], in_channels, m["num_classes"])
    return model
