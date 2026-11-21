"""04 — Aquisição de imagens para predição por município (ETAPA 34).

Usa EXATAMENTE o mesmo ``build_fused_composite`` do treino (mesmo pré-processo,
mesmas features, mesma composição temporal, mesmo grid). Gera um COG de features
por município, pronto para inferência.

Uso:
    python 04_download_prediction_data.py --vector data/vectors/municipios.shp \
        --name-field NM_MUN
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from tqdm import tqdm

from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.cog_utils import write_cog
from src.geo_utils import read_vector
from src.pipeline import build_fused_composite
from src.planetary_computer_utils import open_catalog
from src.sensor_fusion import to_numpy_chw
from src.stac_utils import build_provenance_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aquisição de imagens por município.")
    p.add_argument("--vector", required=True, help="Shapefile de municípios.")
    p.add_argument("--name-field", default="municipality",
                   help="Campo com o nome do município.")
    p.add_argument("--config", default=None)
    return p.parse_args()


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("04_download_prediction_data", cfg)

    catalog = open_catalog(cfg)
    # Resolve o vetor em paths.vectors se não vier absoluto/existente.
    _vec = Path(args.vector)
    if not _vec.is_absolute() and not _vec.exists():
        _vec = resolve_path(cfg, "vectors") / _vec.name
    log.info("Vetor de municípios (download): %s", _vec)
    gdf = read_vector(str(_vec), cfg["project"]["crs_working"])
    name_field = args.name_field if args.name_field in gdf.columns else None
    if name_field is None:
        gdf["municipality"] = [f"mun_{i}" for i in range(len(gdf))]
        name_field = "municipality"

    pred_dir = resolve_path(cfg, "predictions")
    meta_dir = resolve_path(cfg, "metadata")
    prov_records = []

    # Barra de progresso: um passo por município.
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf),
                         desc="Municípios (predição)", unit="mun"):
        mun = row[name_field]
        sub = gdf.iloc[[idx]].copy()
        log.info("Município %s (%d/%d).", mun, idx + 1, len(gdf))

        # Resumível: pula se já existe.
        out_path = pred_dir / f"features_{_safe(mun)}.tif"
        if out_path.exists():
            log.info("  já existe, pulando: %s", out_path.name)
            continue

        stack, prov = build_fused_composite(catalog, sub, cfg)
        if stack is None:
            log.warning("  sem composto válido para %s; pulando.", mun)
            continue

        arr = to_numpy_chw(stack)
        write_cog(out_path, arr, prov["grid"]["transform"],
                  cfg["project"]["crs_working"], nodata=float("nan"),
                  band_names=prov.get("band_names"))
        prov_records.append({
            "municipality": str(mun), "features_cog": str(out_path),
            "s2_item_ids": ",".join(prov["s2_items"]),
            "s1_item_ids": ",".join(prov["s1_items"]),
            "orbit_direction": prov.get("orbit_direction", ""),
            "n_channels": arr.shape[0],
            "date_start": cfg["temporal"]["start_date"],
            "date_end": cfg["temporal"]["end_date"],
            "crs": cfg["project"]["crs_working"],
        })

    if prov_records:
        df = build_provenance_table(prov_records)
        df.to_parquet(meta_dir / "prediction_provenance.parquet", index=False)
        log.info("Proveniência de predição salva: %d municípios.", len(df))


if __name__ == "__main__":
    main()
