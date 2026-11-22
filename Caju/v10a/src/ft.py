"""v10a-ft — ajuste fino a partir dos pesos da v10 (ativado pela variável de ambiente FT_INIT).

FT_INIT = pasta com fold{k}.pt (CV: cada fold parte do MESMO fold da v10, sem vazamento) ou arquivo .pt
(modelo final). Inclui: aquecimento 2 épocas + cosseno; encoder a 0,1x nas 5 primeiras épocas; EMA dos
pesos (0,998); perda + 0,5 x Lovász; sorteio 3x dos chips dos 5 municípios fracos.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch

FRACOS = ("Icapui", "Fortim", "CerroCora", "MonsenhorHipolito", "Buique")
WARM, ENC_SLOW, ENC_F, EMA_D, LOV_W, PESO_FRACO = 2, 5, 0.1, 0.998, 0.5, 3.0


def load_init(fold=None, log=None):
    p = os.environ.get("FT_INIT")
    if not p:
        return None
    p = Path(p)
    f = p / f"fold{fold}.pt" if p.is_dir() else p
    ck = torch.load(f, map_location="cpu", weights_only=False)
    if log:
        log.info("FT: pesos iniciais de %s (blocos de teste da origem: %s)", f, ck.get("test_blocks", "-"))
    return ck


def sampler(meta, ids):
    m = meta.set_index("chip_id").loc[ids, "municipality"].astype(str)
    w = torch.tensor([PESO_FRACO if any(k in x for k in FRACOS) else 1.0 for x in m], dtype=torch.double)
    return torch.utils.data.WeightedRandomSampler(w, num_samples=len(ids), replacement=True)


def with_lovasz(base, ignore):
    import segmentation_models_pytorch as smp
    lov = smp.losses.LovaszLoss(mode="binary", ignore_index=ignore, from_logits=True)

    def f(logits, y):
        return base(logits, y) + LOV_W * lov(logits.float(), y.float())
    return f


def _enc(n):
    return n.startswith("encoder.") or ".encoder." in n


def optimizer(model, cfg):
    tr = cfg["training"]
    enc = [p for n, p in model.named_parameters() if _enc(n)]
    dec = [p for n, p in model.named_parameters() if not _enc(n)]
    return torch.optim.AdamW([{"params": enc, "enc": True}, {"params": dec, "enc": False}],
                             lr=tr["learning_rate"], weight_decay=tr["weight_decay"])


def set_lr(opt, ep, epochs, cfg):
    base = cfg["training"]["learning_rate"]
    if ep < WARM:
        k = (ep + 1) / WARM
    else:
        k = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (ep - WARM) / max(1, epochs - WARM)))
    for g in opt.param_groups:
        g["lr"] = base * k * (ENC_F if g.get("enc") and ep < ENC_SLOW else 1.0)


def ema(model):
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_D), use_buffers=True)
