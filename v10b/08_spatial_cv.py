"""08 — Validação cruzada ESPACIAL (leave-one-block-out) — número "definitivo".

Em vez de um único split treino/val/teste, roda K folds: em cada fold um grupo de
blocos espaciais fica como TESTE e o restante treina/valida. Todo bloco é testado
exatamente uma vez. Reporta média ± desvio-padrão das métricas do caju (IoU, F1,
precisão, recall) — bem mais confiável que medir em 1 bloco só.

Vale para a CNN e/ou para o baseline Random Forest (rápido). Cada fold da CNN é
um treino completo, então em CPU isto é MUITO custoso: prefira GPU e/ou reduza
`--epochs`, ou rode só o RF (`--model rf`) para uma estimativa espacial rápida.

Uso:
    python 08_spatial_cv.py --model both --epochs 30
    python 08_spatial_cv.py --model rf                 # rápido (sem CNN)
    python 08_spatial_cv.py --folds 5 --epochs 25      # 5 folds, 25 épocas/ fold
Saídas: reports/spatial_cv_<exp>.json e reports/spatial_cv_<exp>.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.dataset import (CashewChipDataset, compute_norm_stats, field_meta,
                         pseudo_ids_for_train)
from src.metrics import SegMeter, get_loss
from src.train_utils import (EarlyStopping, find_best_threshold, get_device,
                             set_seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validação cruzada espacial (LOBO-CV).")
    p.add_argument("--config", default=None)
    p.add_argument("--model", choices=["both", "cnn", "rf"], default="both",
                   help="Que modelo validar: CNN, RF (rápido) ou ambos.")
    p.add_argument("--folds", type=int, default=0,
                   help="Nº de folds. 0 = leave-one-block-out (um fold por bloco).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Épocas por fold (default: training.epochs do config).")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--channels", choices=["sentinel", "emb", "both"], default=None,
                   help="v7: canais do modelo (mesmos chips e folds para todos).")
    p.add_argument("--meta", default=None, help="v9: parquet de metadados (ex.: chips_metadata_B.parquet).")
    p.add_argument("--name", default=None, help="v7: nome do experimento (relatórios).")
    return p.parse_args()


def _make_folds(groups, n_folds, seed):
    """Divide os grupos (blocos) em n_folds partições espaciais."""
    rng = np.random.default_rng(seed)
    groups = list(groups)
    rng.shuffle(groups)
    if n_folds <= 0 or n_folds >= len(groups):
        return [[g] for g in groups]  # leave-one-block-out
    folds = [[] for _ in range(n_folds)]
    for i, g in enumerate(groups):
        folds[i % n_folds].append(g)
    return folds


def _ids_for(meta, blocks):
    return meta.loc[meta["block_id"].isin(set(blocks)), "chip_id"].tolist()


def _run_epoch(model, loader, loss_fn, optimizer, device, scaler, train, desc, ema=None):
    import torch
    model.train() if train else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc=desc, unit="batch", leave=False):
            x, y = x.to(device), y.to(device)
            import src.ft as _ft; x, y = _ft.gpu_prep(x, y, loader.dataset, train)  # v10b
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=(scaler is not None)):
                logits = model(x)
                loss = loss_fn(logits, y)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
                if ema is not None:  # v10a-ft: média móvel dos pesos
                    ema.update_parameters(model)
            total += float(loss.item())
    return total / max(1, len(loader))


def _collect_probs(model, loader, device, desc):
    import torch
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, unit="batch", leave=False):
            import src.ft as _ft; x, _ = _ft.gpu_prep(x.to(device), None, loader.dataset, False)  # v10b
            logits = model(x)
            probs.append(torch.sigmoid(logits.squeeze(1)).cpu().numpy().reshape(-1))
            targets.append(y.numpy().reshape(-1))
    return np.concatenate(probs), np.concatenate(targets)


def _train_cnn_fold(meta, train_ids, val_ids, test_ids, cfg, epochs, patience, log, fold=None):
    import torch
    from torch.utils.data import DataLoader
    from src.models import build_model
    import src.ft as ft  # v10a-ft: ajuste fino a partir da v10

    ck = ft.load_init(fold, log)
    stats = ck["stats"] if ck else compute_norm_stats(meta, train_ids, cfg)
    in_ch = len(stats["per_channel"])
    ds_tr = CashewChipDataset(meta, train_ids, stats, cfg, augment=cfg["training"]["augment"])
    ds_va = CashewChipDataset(meta, val_ids, stats, cfg, augment=False)
    ds_te = CashewChipDataset(meta, test_ids, stats, cfg, augment=False)

    device = get_device()
    bs, nw = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)  # v7: GPU alimentada
    smp = ft.sampler(meta, train_ids) if ck else None
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=smp is None, sampler=smp, drop_last=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, **kw)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, **kw)

    model = build_model(cfg, in_channels=in_ch)
    if ck:
        model.load_state_dict(ck["model_state"])
    model = model.to(device)
    loss_fn = get_loss(cfg["training"]["loss"], cfg["chips"]["label_ignore_value"])
    if ck:
        loss_fn = ft.with_lovasz(loss_fn, cfg["chips"]["label_ignore_value"])
        opt = ft.optimizer(model, cfg)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                weight_decay=cfg["training"]["weight_decay"])
    ema = ft.ema(model) if ck else None
    evm = ema.module if ema is not None else model
    scaler = torch.cuda.amp.GradScaler() if (cfg["training"]["mixed_precision"]
                                             and device == "cuda") else None
    stopper = EarlyStopping(patience, mode="max")
    monitor = cfg["training"]["monitor_metric"]
    best_state, best_val, best_epoch = None, -1.0, -1

    for epoch in tqdm(range(epochs), desc="épocas(fold)", unit="ep", leave=False):
        if ck:
            ft.set_lr(opt, epoch, epochs, cfg)
        _run_epoch(model, dl_tr, loss_fn, opt, device, scaler, True, f"ep{epoch}[tr]", ema=ema)
        va_p, va_t = _collect_probs(evm, dl_va, device, f"ep{epoch}[va]")
        m = SegMeter(cfg["chips"]["label_ignore_value"])
        m.update(va_p.reshape(1, -1), va_t.reshape(1, -1), threshold=0.5)
        va_iou = m.compute()[monitor]
        if stopper.step(va_iou):
            best_val, best_epoch = va_iou, epoch  # v9
            best_state = {k: v.detach().cpu().clone() for k, v in evm.state_dict().items()}
        if stopper.should_stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # threshold na validação, avaliação no teste
    va_p, va_t = _collect_probs(model, dl_va, device, "val(thr)")
    best_t, _ = find_best_threshold(va_p.reshape(1, -1), va_t.reshape(1, -1), cfg,
                                    cfg["chips"]["label_ignore_value"])
    te_p, te_t = _collect_probs(model, dl_te, device, "teste")
    meter = SegMeter(cfg["chips"]["label_ignore_value"])
    meter.update(te_p.reshape(1, -1), te_t.reshape(1, -1), threshold=best_t)
    res = meter.compute()
    res["threshold"] = best_t
    res["best_val_iou"] = float(best_val)
    res["best_epoch"] = int(best_epoch)  # v9: p/ épocas fixas no modelo final
    res["_oof"] = (te_p.astype("float16"), te_t.astype("uint8"))  # v9: limiar global
    res["_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    res["_stats"] = stats
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def _agg(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "n": len(vals)}
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.channels:
        cfg["experiment"]["channels"] = args.channels
    if args.name:
        cfg["experiment"]["name"] = args.name
    log = setup_logging("08_spatial_cv", cfg)
    log.info("Experimento %s | canais=%s", cfg["experiment"]["name"],
             cfg["experiment"].get("channels", "both"))
    set_seed(cfg["project"]["random_seed"])

    meta = pd.read_parquet(resolve_path(cfg, "metadata") / (args.meta or "chips_metadata.parquet"))
    # v6: folds definidos só por blocos com rótulo de campo; pseudo-chips vão
    # apenas para o treino de cada fold (fora dos blocos de val/teste e vizinhos).
    groups = sorted(field_meta(meta)["block_id"].dropna().unique().tolist())
    meta_all, meta = meta, field_meta(meta)
    log.info("Chips: %d | blocos espaciais: %d", len(meta), len(groups))
    if len(groups) < 3:
        log.error("Poucos blocos (%d) para CV espacial. Reduza block_size_km "
                  "e refaça o 02.", len(groups))
        return

    folds = _make_folds(groups, args.folds, cfg["project"]["random_seed"])
    epochs = args.epochs or cfg["training"]["epochs"]
    patience = args.patience or cfg["training"]["patience"]
    log.info("CV espacial: %d folds | épocas/fold=%d | modelo=%s",
             len(folds), epochs, args.model)

    # backup do norm_stats.json (a CNN o sobrescreve por fold)
    stats_path = Path(cfg["_output_root"]) / cfg["normalization"]["stats_file"]
    backup = stats_path.read_text() if stats_path.exists() else None

    cnn_rows, rf_rows, per_fold = [], [], []
    for i, test_blocks in enumerate(folds):
        remaining = [g for g in groups if g not in set(test_blocks)]
        # v6: bloco de validação varia por fold (antes era sempre o 1º)
        val_blocks = [remaining[i % len(remaining)]] if len(remaining) > 1 else remaining
        train_blocks = [g for g in remaining if g not in set(val_blocks)] or remaining
        tr_ids, va_ids, te_ids = (_ids_for(meta, train_blocks),
                                  _ids_for(meta, val_blocks), _ids_for(meta, test_blocks))
        tr_ids += pseudo_ids_for_train(meta_all, set(val_blocks) | set(test_blocks))
        log.info("Fold %d/%d | teste=%s (%d chips) | treino=%d | val=%d",
                 i + 1, len(folds), test_blocks, len(te_ids), len(tr_ids), len(va_ids))
        rec = {"fold": i + 1, "test_blocks": ",".join(map(str, test_blocks)),
               "n_test": len(te_ids), "n_train": len(tr_ids), "n_val": len(va_ids)}

        if args.model in ("cnn", "both") and te_ids and tr_ids:
            try:
                import time as _t
                _t0 = _t.time()
                c = _train_cnn_fold(meta_all, tr_ids, va_ids, te_ids, cfg, epochs,
                                    patience, log, fold=i + 1)
                c["minutes"] = round((_t.time() - _t0) / 60, 1)  # v7: tempo do fold
                _p, _t_ = c.pop("_oof")  # v9: predições fora do fold (teste)
                _od = resolve_path(cfg, "reports") / "oof"; _od.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(_od / f"{cfg['experiment']['name']}_fold{i + 1}.npz", p=_p, t=_t_)
                # v6: salva o modelo do fold (+ stats e limiar) para o ensemble do 05
                import torch
                ens_dir = (resolve_path(cfg, "models")
                           / f"ensemble_{cfg['experiment']['name']}")
                ens_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": c.pop("_state"), "stats": c.pop("_stats"),
                            "threshold": c["threshold"], "test_blocks": test_blocks,
                            "channels": cfg["experiment"].get("channels", "both")},
                           ens_dir / f"fold{i + 1}.pt")
                cnn_rows.append(c)
                rec.update({f"cnn_{k}": v for k, v in c.items()})
                log.info("  CNN fold %d: IoU=%.3f F1=%.3f P=%.3f R=%.3f (t=%.2f)",
                         i + 1, c["iou_cashew"], c["f1_cashew"],
                         c["precision_cashew"], c["recall_cashew"], c["threshold"])
            except Exception as exc:  # noqa: BLE001
                log.warning("  CNN fold %d falhou: %s", i + 1, exc)

        if args.model in ("rf", "both") and te_ids and tr_ids:
            try:
                from src.ml_baseline import train_rf_baseline
                split = {"train": tr_ids + va_ids, "test": te_ids}
                rf = train_rf_baseline(meta, split, cfg)["metrics"]
                rf_rows.append(rf)
                rec.update({f"rf_{k}": v for k, v in rf.items()})
                log.info("  RF  fold %d: IoU=%.3f F1=%.3f", i + 1,
                         rf["iou_cashew"], rf["f1_cashew"])
            except Exception as exc:  # noqa: BLE001
                log.warning("  RF fold %d falhou: %s", i + 1, exc)

        per_fold.append(rec)

    # restaura o norm_stats.json original
    if backup is not None:
        stats_path.write_text(backup, encoding="utf-8")

    keys = ["iou_cashew", "f1_cashew", "precision_cashew", "recall_cashew", "dice_cashew"]
    report = {
        "experiment": cfg["experiment"]["name"],
        "modality": cfg["experiment"]["modality"],
        "n_folds": len(folds),
        "epochs_per_fold": epochs,
        "cnn_cv": _agg(cnn_rows, keys) if cnn_rows else None,
        "rf_cv": _agg(rf_rows, keys) if rf_rows else None,
        "per_fold": per_fold,
    }
    reports = resolve_path(cfg, "reports")
    (reports / f"spatial_cv_{cfg['experiment']['name']}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(per_fold).to_csv(
        reports / f"spatial_cv_{cfg['experiment']['name']}.csv", index=False)

    if report["cnn_cv"]:
        c = report["cnn_cv"]["iou_cashew"]
        log.info("== CNN CV: IoU_caju = %.3f ± %.3f (n=%d folds) ==",
                 c["mean"], c["std"], c["n"])
    if report["rf_cv"]:
        r = report["rf_cv"]["iou_cashew"]
        log.info("== RF  CV: IoU_caju = %.3f ± %.3f (n=%d folds) ==",
                 r["mean"], r["std"], r["n"])
    log.info("Relatório salvo: %s", reports / f"spatial_cv_{cfg['experiment']['name']}.json")


if __name__ == "__main__":
    main()
