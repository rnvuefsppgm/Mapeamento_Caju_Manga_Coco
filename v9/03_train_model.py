"""03 — Treinamento e avaliação (ETAPAS 21-33).

Fluxo:
    1. split espacial (previne data leakage);
    2. normalização estimada só no treino;
    3. treino da CNN (AMP, early stopping, checkpoint, scheduler);
    4. escolha de threshold na validação;
    5. avaliação no teste (métricas focadas em caju);
    6. baseline Random Forest com o MESMO split (comparação justa);
    7. salva relatório de métricas e o melhor modelo.

Uso:
    python 03_train_model.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.dataset import (CashewChipDataset, compute_norm_stats, spatial_split)
from src.metrics import SegMeter, get_loss
from src.ml_baseline import train_rf_baseline
from src.train_utils import (EarlyStopping, find_best_threshold, get_device,
                             load_checkpoint, save_checkpoint, set_seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treinamento do modelo de caju.")
    p.add_argument("--config", default=None)
    p.add_argument("--resume", default=None, help="Checkpoint para retomar.")
    p.add_argument("--skip-baseline", action="store_true")
    return p.parse_args()


def _run_epoch(model, loader, loss_fn, optimizer, device, scaler, train: bool,
               desc: str = "treino"):
    import torch
    model.train() if train else model.eval()
    total = 0.0
    meter = SegMeter()
    ctx = torch.enable_grad() if train else torch.no_grad()
    # Barra de progresso por batch (não persiste entre épocas: leave=False).
    iterator = tqdm(loader, desc=desc, unit="batch", leave=False)
    with ctx:
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=(scaler is not None)):
                logits = model(x)
                loss = loss_fn(logits, y)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            batch_loss = float(loss.item())
            total += batch_loss
            iterator.set_postfix(loss=f"{batch_loss:.4f}")
            prob = torch.sigmoid(logits.squeeze(1)).detach().cpu().numpy()
            meter.update(prob, y.cpu().numpy())
    return total / max(1, len(loader)), meter.compute()


def _collect_probs(model, loader, device, desc: str = "coletando probs"):
    """Coleta probs/targets achatados (para threshold e avaliação final)."""
    import torch
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, unit="batch", leave=False):
            logits = model(x.to(device))
            probs.append(torch.sigmoid(logits.squeeze(1)).cpu().numpy().reshape(-1))
            targets.append(y.numpy().reshape(-1))
    return np.concatenate(probs), np.concatenate(targets)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("03_train_model", cfg)
    set_seed(cfg["project"]["random_seed"])

    import torch
    from torch.utils.data import DataLoader
    from src.models import build_model

    meta = pd.read_parquet(resolve_path(cfg, "metadata") / "chips_metadata.parquet")
    log.info("Chips carregados: %d.", len(meta))

    split = spatial_split(meta, cfg, seed=cfg["project"]["random_seed"])
    stats = compute_norm_stats(meta, split["train"], cfg)

    ds_tr = CashewChipDataset(meta, split["train"], stats, cfg,
                              augment=cfg["training"]["augment"])
    ds_va = CashewChipDataset(meta, split["val"], stats, cfg, augment=False)
    ds_te = CashewChipDataset(meta, split["test"], stats, cfg, augment=False)

    device = get_device()
    log.info("Dispositivo: %s", device)
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=nw)

    in_ch = len(stats["per_channel"])
    model = build_model(cfg, in_channels=in_ch).to(device)
    loss_fn = get_loss(cfg["training"]["loss"], cfg["chips"]["label_ignore_value"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = None
    if cfg["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"])
    scaler = torch.cuda.amp.GradScaler() if (cfg["training"]["mixed_precision"]
                                             and device == "cuda") else None

    start_epoch, best = 0, -1.0
    ckpt_path = resolve_path(cfg, "models") / f"{cfg['experiment']['name']}_best.pt"
    if args.resume:
        start_epoch, best = load_checkpoint(Path(args.resume), model, optimizer, device)
        log.info("Retomado da época %d (best=%.4f).", start_epoch, best)

    stopper = EarlyStopping(cfg["training"]["patience"], mode="max")
    monitor = cfg["training"]["monitor_metric"]

    # Barra de progresso de nível superior: uma por época de treino.
    epoch_bar = tqdm(range(start_epoch, cfg["training"]["epochs"]),
                     desc="Épocas", unit="época")
    for epoch in epoch_bar:
        tr_loss, tr_m = _run_epoch(model, dl_tr, loss_fn, optimizer, device, scaler,
                                   True, desc=f"época {epoch} [treino]")
        va_loss, va_m = _run_epoch(model, dl_va, loss_fn, optimizer, device, None,
                                   False, desc=f"época {epoch} [val]")
        if scheduler:
            scheduler.step()
        log.info("Época %d | train_loss=%.4f IoU=%.3f | val_loss=%.4f IoU=%.3f",
                 epoch, tr_loss, tr_m["iou_cashew"], va_loss, va_m["iou_cashew"])
        epoch_bar.set_postfix(val_IoU=f"{va_m['iou_cashew']:.3f}",
                              best=f"{max(best, 0.0):.3f}")
        improved = stopper.step(va_m[monitor])
        if improved:
            best = va_m[monitor]
            save_checkpoint(ckpt_path, model, optimizer, epoch, best, cfg)
            log.info("  -> novo melhor (%s=%.4f), checkpoint salvo.", monitor, best)
        if stopper.should_stop:
            log.info("Early stopping na época %d.", epoch)
            break
    epoch_bar.close()

    # Threshold na validação + avaliação no teste
    load_checkpoint(ckpt_path, model, None, device)
    va_probs, va_targets = _collect_probs(model, dl_va, device, desc="val (threshold)")
    best_t, _ = find_best_threshold(va_probs.reshape(1, -1),
                                    va_targets.reshape(1, -1), cfg,
                                    cfg["chips"]["label_ignore_value"])
    te_probs, te_targets = _collect_probs(model, dl_te, device, desc="teste (avaliação)")
    meter = SegMeter(cfg["chips"]["label_ignore_value"])
    meter.update(te_probs.reshape(1, -1), te_targets.reshape(1, -1), threshold=best_t)
    test_metrics = meter.compute()
    log.info("TESTE (t=%.2f): %s", best_t, test_metrics)

    report = {
        "experiment": cfg["experiment"]["name"],
        "modality": cfg["experiment"]["modality"],
        "in_channels": in_ch,
        "best_threshold": best_t,
        "test_metrics": test_metrics,
        "n_train": len(split["train"]), "n_val": len(split["val"]),
        "n_test": len(split["test"]),
    }

    if not args.skip_baseline:
        try:
            rf = train_rf_baseline(meta, split, cfg)
            report["rf_baseline"] = rf["metrics"]
        except Exception as exc:  # noqa: BLE001
            log.warning("Baseline RF falhou: %s", exc)

    rep_path = (resolve_path(cfg, "reports")
                / f"metrics_{cfg['experiment']['name']}.json")
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Relatório salvo: %s", rep_path)


if __name__ == "__main__":
    main()
