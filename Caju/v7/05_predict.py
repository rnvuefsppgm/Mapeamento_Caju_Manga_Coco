"""05 — Inferência por janela deslizante (ETAPAS 35-37).

Carrega o melhor modelo e o mesmo ``norm_stats`` do treino, gera para cada
município os rasters de probabilidade e classificação, além de um mapa de
incerteza (entropia). Tudo exportado como COG.

Uso:
    python 05_predict.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.cog_utils import write_cog
from src.dataset import select_channels
from src.embeddings import stack_with_embeddings
from src.dataset import apply_norm
from src.inference_utils import probability_entropy, sliding_window_predict
from src.train_utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferência de caju por município.")
    p.add_argument("--config", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Threshold; se ausente, usa o do relatório de treino.")
    p.add_argument("--ensemble", default=None,
                   help="v7: nome do experimento; usa models/ensemble_<nome>/fold*.pt (do 08).")
    return p.parse_args()


def _load_threshold(cfg, log) -> float:
    if cfg["inference"]["threshold"] is not None:
        return float(cfg["inference"]["threshold"])
    rep = resolve_path(cfg, "reports") / f"metrics_{cfg['experiment']['name']}.json"
    if rep.exists():
        t = json.loads(rep.read_text()).get("best_threshold", 0.5)
        log.info("Threshold do relatório de treino: %.2f", t)
        return float(t)
    log.warning("Threshold não encontrado; usando 0.5.")
    return 0.5


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("05_predict", cfg)

    import torch
    from src.models import build_model

    device = get_device()
    # v6: --ensemble usa os modelos dos folds do 08 (models/ensemble/fold*.pt),
    # cada um com sua normalização; probabilidade = média; limiar = média dos folds.
    members = []
    if args.ensemble:
        for f in sorted((resolve_path(cfg, "models") / f"ensemble_{args.ensemble}").glob("fold*.pt")):
            ck = torch.load(f, map_location="cpu", weights_only=False)
            m = build_model(cfg, in_channels=len(ck["stats"]["per_channel"]))
            m.load_state_dict(ck["model_state"])
            members.append((m.to(device).eval(), ck["stats"], ck["threshold"]))
            cfg["experiment"]["channels"] = ck.get("channels", "both")  # v7
        if not members:
            log.error("Nenhum modelo em models/ensemble. Rode o 08 antes.")
            return
        log.info("Ensemble com %d modelos.", len(members))
        threshold = (args.threshold if args.threshold is not None
                     else float(np.mean([t for _, _, t in members])))
    else:
        stats = json.loads((Path(cfg["_output_root"])
                            / cfg["normalization"]["stats_file"]).read_text())
        model = build_model(cfg, in_channels=len(stats["per_channel"]))
        ckpt = resolve_path(cfg, "models") / f"{cfg['experiment']['name']}_best.pt"
        load_checkpoint(ckpt, model, None, device)
        members.append((model.to(device), stats, None))
        threshold = (args.threshold if args.threshold is not None
                     else _load_threshold(cfg, log))

    prov_path = resolve_path(cfg, "metadata") / "prediction_provenance.parquet"
    if not prov_path.exists():
        log.error("Rode 04_download_prediction_data.py primeiro.")
        return
    prov = pd.read_parquet(prov_path)
    pred_dir = resolve_path(cfg, "predictions")

    # Barra de progresso: um passo por município (a inferência interna de cada
    # município tem sua própria barra de janelas — ver inference_utils).
    for _, row in tqdm(prov.iterrows(), total=len(prov),
                       desc="Inferência (municípios)", unit="mun"):
        mun = row["municipality"]
        cog = row["features_cog"]
        log.info("Inferindo município: %s", mun)
        with rasterio.open(cog) as ds:
            feats = ds.read().astype("float32")   # (C,H,W) NaN=NoData
            transform, crs = ds.transform, ds.crs

        feats = select_channels(stack_with_embeddings(feats, cog, cfg), cfg)  # v7
        valid = np.isfinite(feats).all(axis=0)
        feats = np.nan_to_num(feats, nan=0.0)
        prob = None
        for model, stats, _ in members:
            p = sliding_window_predict(apply_norm(feats, stats), valid, model, cfg, device)
            prob = p if prob is None else prob + p
        prob = prob / len(members)
        del feats
        classification = np.where(np.isfinite(prob), (prob >= threshold).astype("uint8"),
                                  255).astype("uint8")
        entropy = probability_entropy(np.nan_to_num(prob, nan=0.5))

        base = "".join(ch if ch.isalnum() else "_" for ch in str(mun))
        write_cog(pred_dir / f"{base}_cashew_probability.tif",
                  np.nan_to_num(prob, nan=-1.0)[None], transform, crs, nodata=-1.0,
                  band_names=["P_cashew"], dtype="float32")
        write_cog(pred_dir / f"{base}_cashew_classification.tif",
                  classification[None], transform, crs, nodata=255,
                  band_names=["cashew_class"], dtype="uint8")
        write_cog(pred_dir / f"{base}_cashew_uncertainty.tif",
                  entropy[None], transform, crs, nodata=-1.0,
                  band_names=["entropy"], dtype="float32")
        log.info("  -> mapas salvos (threshold=%.2f).", threshold)


if __name__ == "__main__":
    main()
