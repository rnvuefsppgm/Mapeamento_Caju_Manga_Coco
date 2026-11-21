"""01 — Aquisição dos dados de treinamento (ETAPAS 2-16).

Lê os polígonos de treino, valida geometrias, agrupa por clusters espaciais para
consultas STAC eficientes, constrói o composto multissensor fundido (10 m) por
cluster e salva como COG intermediário + tabela de proveniência.

Uso:
    python 01_download_training_data.py --vector data/vectors/training_polygons.shp \
        --role ground_truth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from tqdm import tqdm

from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.cog_utils import write_cog
from src.geo_utils import (classify_vector_role, cluster_bboxes_for_query,
                           group_by_spatial_blocks, read_vector)
from src.pipeline import build_fused_composite
from src.planetary_computer_utils import (discover_sentinel_collections,
                                          open_catalog)
from src.sensor_fusion import to_numpy_chw
from src.stac_utils import build_provenance_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aquisição de dados de treino S1+S2.")
    p.add_argument("--vector", required=True, help="Shapefile/GPKG de treino.")
    p.add_argument("--role", choices=["ground_truth", "aoi"], default="ground_truth",
                   help="Papel do vetor: label de caju ou apenas AOI de download.")
    p.add_argument("--config", default=None)
    p.add_argument("--year", type=int, default=None,
                   help="v6: ano do composto (usa temporal.season_start/end MM-DD).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("01_download_training_data", cfg)

    # v6: limita threads do dask (outros processos rodando na máquina).
    import dask
    dask.config.set(scheduler="threads",
                    num_workers=int(cfg.get("resources", {}).get("dask_threads", 4)))

    # v6: multi-ano — sobrescreve a janela temporal e marca as saídas com o ano.
    year = args.year or int(str(cfg["temporal"]["start_date"])[:4])
    if args.year:
        t = cfg["temporal"]
        t["start_date"] = f"{year}-{t.get('season_start', '05-01')}"
        t["end_date"] = f"{year}-{t.get('season_end', '10-31')}"
    log.info("Ano %d: janela %s a %s", year, cfg["temporal"]["start_date"],
             cfg["temporal"]["end_date"])

    catalog = open_catalog(cfg)
    discover_sentinel_collections(catalog)  # valida IDs em runtime (metodologia)

    # Resolve o vetor: usa o caminho como veio se for absoluto/existente; caso
    # contrário procura pelo NOME do arquivo em paths.vectors
    # (D:/Modelo_Caju/v3/data/vectors). Garante o download a partir dessa pasta.
    _vec = Path(args.vector)
    if not _vec.is_absolute() and not _vec.exists():
        _vec = resolve_path(cfg, "vectors") / _vec.name
    log.info("Vetor de treino (download): %s", _vec)
    gdf = read_vector(str(_vec), cfg["project"]["crs_working"])
    classify_vector_role(gdf)
    log.info("Papel declarado do vetor: %s", args.role)

    gdf = group_by_spatial_blocks(gdf, cfg["split"]["block_size_km"])
    if "municipality" not in gdf.columns:
        gdf["municipality"] = "NA"

    clusters = cluster_bboxes_for_query(gdf, cluster_size_km=40.0)
    processed_dir = resolve_path(cfg, "processed")
    meta_dir = resolve_path(cfg, "metadata")

    prov_records = []
    # Barra de progresso: um passo por cluster de consulta STAC.
    for cluster_id, _bbox, sub in tqdm(clusters, desc="Clusters (treino)",
                                       unit="cluster"):
        log.info("Cluster %s: %d feições.", cluster_id, len(sub))
        out_path = processed_dir / f"features_{cluster_id}_{year}.tif"
        if out_path.exists() and (meta_dir / f"training_provenance_{year}.parquet").exists():
            log.info("  já existe, pulando: %s", out_path.name)  # v6: retomável
            continue
        stack, prov = build_fused_composite(catalog, sub, cfg)
        if stack is None:
            log.warning("Cluster %s sem composto válido; pulando.", cluster_id)
            continue

        arr = to_numpy_chw(stack)  # (C,H,W)
        grid = prov["grid"]
        write_cog(
            out_path, arr, grid["transform"], cfg["project"]["crs_working"],
            nodata=float("nan"), band_names=prov.get("band_names"),
        )
        prov_records.append({
            "cluster_id": cluster_id,
            "features_cog": str(out_path),
            "role": args.role,
            "municipality": ",".join(sorted(sub["municipality"].unique().astype(str))),
            "block_ids": ",".join(sorted(sub["block_id"].unique())),
            "s2_item_ids": ",".join(prov["s2_items"]),
            "s1_item_ids": ",".join(prov["s1_items"]),
            "orbit_direction": prov.get("orbit_direction", ""),
            "date_start": cfg["temporal"]["start_date"],
            "date_end": cfg["temporal"]["end_date"],
            "n_channels": arr.shape[0],
            "crs": cfg["project"]["crs_working"],
            "resolution": cfg["harmonization"]["resolution"],
            "year": year,
        })
        del stack, arr  # v6: libera RAM entre clusters

    if prov_records:
        df = build_provenance_table(prov_records)
        df.to_parquet(meta_dir / f"training_provenance_{year}.parquet", index=False)
        log.info("Proveniência salva: %d clusters.", len(df))
    else:
        log.error("Nenhum composto de treino gerado. Verifique AOI/janela temporal.")


if __name__ == "__main__":
    main()
