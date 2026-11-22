"""07 — Estatísticas municipais (ETAPA 41).

Calcula área de caju por município a partir dos mapas classificados (limpos, se
existirem) e exporta CSV + Parquet.

Uso:
    python 07_export_statistics.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from src.config import ensure_dirs, load_config, resolve_path, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estatísticas municipais de caju.")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("07_export_statistics", cfg)

    pred_dir = resolve_path(cfg, "predictions")
    res = cfg["harmonization"]["resolution"]
    pixel_ha = (res ** 2) / 1e4

    rows = []
    class_files = sorted(pred_dir.glob("*_cashew_classification*.tif"))
    # Barra de progresso: um passo por mapa de classificação avaliado.
    for cf in tqdm(class_files, desc="Estatísticas municipais", unit="mapa"):
        # prioriza a versão *_clean.tif se existir
        if cf.name.endswith("_clean.tif"):
            base = cf.name.replace("_cashew_classification_clean.tif", "")
        else:
            clean = cf.with_name(cf.stem + "_clean.tif")
            if clean.exists():
                continue  # a versão clean será processada
            base = cf.name.replace("_cashew_classification.tif", "")

        with rasterio.open(cf) as ds:
            cls = ds.read(1)
        valid = cls != 255
        cashew_px = int(np.sum(cls == 1))
        mun_px = int(np.sum(valid))
        cashew_ha = cashew_px * pixel_ha
        mun_ha = mun_px * pixel_ha
        prob_file = pred_dir / f"{base}_cashew_probability.tif"
        mean_prob = None
        if prob_file.exists():
            with rasterio.open(prob_file) as pds:
                p = pds.read(1)
                p = p[(p >= 0) & (cls == 1)]
                mean_prob = float(p.mean()) if p.size else None

        rows.append({
            "municipality": base,
            "cashew_area_ha": round(cashew_ha, 2),
            "municipality_area_ha": round(mun_ha, 2),
            "municipality_percentage": round(100 * cashew_ha / mun_ha, 3)
            if mun_ha else 0.0,
            "mean_probability": mean_prob,
            "year": cfg["temporal"]["start_date"][:4],
            "model_version": cfg["experiment"]["name"],
        })

    if not rows:
        log.error("Nenhuma classificação para estatística.")
        return

    df = pd.DataFrame(rows)
    reports = resolve_path(cfg, "reports")
    df.to_csv(reports / "municipal_statistics.csv", index=False)
    df.to_parquet(reports / "municipal_statistics.parquet", index=False)
    log.info("Estatísticas municipais salvas (%d municípios).", len(df))
    log.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
