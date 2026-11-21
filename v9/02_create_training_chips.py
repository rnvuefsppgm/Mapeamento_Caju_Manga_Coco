"""02 — Criação dos chips de treinamento (ETAPAS 17-20).

Para cada COG de features gerado no passo 01, rasteriza as labels de caju
(quando role=ground_truth) alinhadas ao grid, corta chips com QC e acumula a
tabela de metadados (Parquet). Inclui geração implícita de negativos (chips sem
caju) e suporte a incorporação de hard negatives.

Uso:
    python 02_create_training_chips.py --vector data/vectors/training_polygons.shp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from src.chip_generator import generate_chips, save_metadata
from src.embeddings import stack_with_embeddings
from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.geo_utils import group_by_spatial_blocks, read_vector
from src.labels import mark_nodata_as_ignore, rasterize_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Criação de chips de treino.")
    p.add_argument("--vector", required=True, help="Polígonos de caju (ground truth).")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("02_create_training_chips", cfg)

    meta_dir = resolve_path(cfg, "metadata")
    chips_dir = resolve_path(cfg, "chips")
    # v6: junta a proveniência de todos os anos (training_provenance_<ano>.parquet)
    prov_files = sorted(meta_dir.glob("training_provenance_*.parquet"))
    if not prov_files:
        log.error("Rode 01_download_training_data.py primeiro (proveniência ausente).")
        return
    prov = pd.concat([pd.read_parquet(p) for p in prov_files], ignore_index=True)
    log.info("Proveniência: %d COGs de %d ano(s).", len(prov), len(prov_files))

    # Mesma resolução do 01: procura o vetor em paths.vectors se necessário.
    _vec = Path(args.vector)
    if not _vec.is_absolute() and not _vec.exists():
        _vec = resolve_path(cfg, "vectors") / _vec.name
    log.info("Vetor (labels): %s", _vec)
    gdf = read_vector(str(_vec), cfg["project"]["crs_working"])
    gdf = group_by_spatial_blocks(gdf, cfg["split"]["block_size_km"])

    all_records = []
    # Barra de progresso: um passo por COG de features do passo 01.
    for _, row in tqdm(prov.iterrows(), total=len(prov),
                       desc="COGs -> chips", unit="COG"):
        cog = row["features_cog"]
        with rasterio.open(cog) as ds:
            features = ds.read().astype("float32")  # (C,H,W); NaN=NoData
            transform = ds.transform
            out_shape = (ds.height, ds.width)
        features = stack_with_embeddings(features, cog, cfg)  # v7: [Sentinel ; emb]

        if row["role"] == "ground_truth":
            label = rasterize_labels(gdf, out_shape, transform, cfg)
        else:
            # AOI de download: NÃO há label -> tudo background/ignore.
            log.warning("Cluster %s é AOI (sem label); chips serão só negativos.",
                        row["cluster_id"])
            label = np.zeros(out_shape, dtype="uint8")

        valid_mask = np.isfinite(features).all(axis=0)
        label = mark_nodata_as_ignore(label, valid_mask,
                                      cfg["chips"]["label_ignore_value"])

        # block_id do cluster (fallback). O block_id DEFINITIVO de cada chip é
        # recalculado dentro de generate_chips pela geolocalização do chip, no
        # tamanho de bloco de `split.block_size_km` (evita que o split herde o
        # bloco de 40 km do clustering de consulta). Ver CORREÇÃO ponto B.
        block_id = str(row["block_ids"]).split(",")[0] if row.get("block_ids") else ""
        meta_common = {
            "municipality": row.get("municipality", ""),
            "date_start": row["date_start"], "date_end": row["date_end"],
            "crs": row["crs"], "block_id": block_id,
            "s2_item_ids": row.get("s2_item_ids", ""),
            "s1_item_ids": row.get("s1_item_ids", ""),
            "orbit_direction": row.get("orbit_direction", ""),
            "year": int(row.get("year", 0) or 0), "source": "field",
        }
        recs = generate_chips(features, label, cfg, chips_dir,
                              prefix=f"{row['cluster_id']}_{row.get('year', '')}",
                              meta_common=meta_common, transform=transform)
        all_records.extend(recs)
        del features, label, valid_mask  # v6: libera RAM entre COGs

    if not all_records:
        log.error("Nenhum chip gerado. Verifique QC e alinhamento label/features.")
        return

    df = save_metadata(all_records, meta_dir / "chips_metadata.parquet")
    n_pos = int((~df["is_negative"]).sum())
    log.info("Chips: %d total | %d com caju | %d negativos.",
             len(df), n_pos, len(df) - n_pos)


if __name__ == "__main__":
    main()
