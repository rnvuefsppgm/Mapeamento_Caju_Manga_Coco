"""02b (v6) — Chips de PSEUDO-RÓTULO a partir de mapas já gerados (v4).

Ideia: aproveitar áreas já mapeadas com ALTA confiança como amostras extras.
    p >= pseudo_labels.pos_threshold  -> caju (1)
    p <= pseudo_labels.neg_threshold  -> não-caju (0)
    entre os dois / NoData            -> ignorado (255, não entra na loss)
Polígonos de campo têm prioridade (dentro deles o rótulo é 1).

Regras anti-vazamento (aplicadas em src/dataset.py):
    * chips marcados ``source="pseudo"`` entram SÓ no treino;
    * pseudo-chips em blocos de val/teste (ou vizinhos) são descartados.

Rodar DEPOIS do 02 (este script acrescenta linhas ao chips_metadata.parquet).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from rasterio.warp import Resampling, reproject

from src.chip_generator import generate_chips, save_metadata
from src.embeddings import stack_with_embeddings
from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.geo_utils import read_vector
from src.labels import rasterize_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chips de pseudo-rótulo (v6).")
    p.add_argument("--vector", default="Pol_Caju.shp", help="Polígonos de campo.")
    p.add_argument("--config", default=None)
    return p.parse_args()


def _review_points(prob, transform, crs, mun, cfg, log, n=150, seed=42):
    """v7: pontos para revisão manual (QGIS) nas zonas INCERTAS (0,3 <= p <= 0,7).

    Saída: data/labels/review_points.gpkg (camada por município) com a
    probabilidade v4 e um campo ``classe_revisada`` vazio para preencher.
    """
    import geopandas as gpd
    rows, cols = np.where((prob >= 0.3) & (prob <= 0.7))
    if rows.size == 0:
        return
    idx = np.random.default_rng(seed).choice(rows.size, min(n, rows.size), replace=False)
    xs, ys = rasterio.transform.xy(transform, rows[idx], cols[idx])
    g = gpd.GeoDataFrame({"municipio": mun, "p_v4": prob[rows[idx], cols[idx]],
                          "classe_revisada": ""},
                         geometry=gpd.points_from_xy(xs, ys), crs=crs)
    out = resolve_path(cfg, "labels") / "review_points.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out, layer=mun, driver="GPKG")
    log.info("%s: %d pontos de revisão -> %s", mun, len(g), out.name)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("02b_pseudo_label_chips", cfg)
    pcfg = cfg["pseudo_labels"]
    ign = cfg["chips"]["label_ignore_value"]

    meta_path = resolve_path(cfg, "metadata") / "chips_metadata.parquet"
    if not meta_path.exists():
        log.error("Rode o 02 primeiro (chips_metadata.parquet ausente).")
        return
    meta = pd.read_parquet(meta_path)
    if "source" in meta.columns:
        meta = meta[meta["source"] != "pseudo"]  # reexecução idempotente
    n_ch_field = int(meta["n_channels"].mode().iloc[0])

    vec = Path(args.vector)
    if not vec.is_absolute() and not vec.exists():
        vec = resolve_path(cfg, "vectors") / vec.name
    gdf = read_vector(str(vec), cfg["project"]["crs_working"])

    cfg_ps = copy.deepcopy(cfg)
    cfg_ps["chips"]["stride"] = int(pcfg["stride"])
    src_dir = Path(pcfg["source_dir"])
    chips_dir = resolve_path(cfg, "chips")
    records = []
    # v7: alvo = COGs dos CLUSTERS DE TREINO de 2025 (área de treino apenas);
    # probabilidade = mapas v4 (2023) dos municípios, reprojetados e mosaicados.
    prob_files = [Path(pcfg["prob_dir"]) / f"{m}_cashew_probability.tif"
                  for m in pcfg["municipalities"]]
    prob_files = [f for f in prob_files if f.exists()]
    cogs = sorted(resolve_path(cfg, "processed").glob(f"features_*_{pcfg['year']}.tif"))
    for f_feat in tqdm(cogs, desc="Pseudo-rótulos", unit="cluster"):
        mun = f_feat.stem.replace("features_", "")
        with rasterio.open(f_feat) as ds:
            transform, shape, crs = ds.transform, (ds.height, ds.width), ds.crs
            feats = ds.read().astype("float32")
        feats = stack_with_embeddings(feats, f_feat, cfg)  # v7
        if feats.shape[0] != n_ch_field:
            log.error("%s: %d canais != %d dos chips de campo; pulando.",
                      mun, feats.shape[0], n_ch_field)
            continue
        log.info("%s: %dx%d px, %d canais (~%.1f GB float32).", mun, shape[0],
                 shape[1], feats.shape[0], feats.nbytes / 1e9)
        # v7: probabilidade (v4/2023) reprojetada para a grade das features 2025
        prob = np.full(shape, -1.0, dtype="float32")
        for f_prob in prob_files:
            tmp = np.full(shape, -1.0, dtype="float32")
            with rasterio.open(f_prob) as ds:
                reproject(ds.read(1), tmp, src_transform=ds.transform, src_crs=ds.crs,
                          src_nodata=-1.0, dst_transform=transform, dst_crs=crs,
                          dst_nodata=-1.0, resampling=Resampling.nearest)
            prob = np.where(prob < 0, tmp, prob)
        log.info("%s: %.0f%% da área com probabilidade v4.", mun, 100 * (prob >= 0).mean())
        _review_points(prob, transform, crs, mun, cfg, log)
        label = np.full(shape, ign, dtype="uint8")
        label[(prob >= 0) & (prob <= pcfg["neg_threshold"])] = 0
        label[prob >= pcfg["pos_threshold"]] = 1
        field = rasterize_labels(gdf, shape, transform, cfg)
        label[field == cfg["chips"]["label_positive_value"]] = 1
        label[~np.isfinite(feats).all(axis=0)] = ign
        del prob, field
        log.info("%s: caju=%.1f%% | não-caju=%.1f%% | ignorado=%.1f%%", mun,
                 100 * (label == 1).mean(), 100 * (label == 0).mean(),
                 100 * (label == ign).mean())
        recs = generate_chips(
            feats, label, cfg_ps, chips_dir, prefix=f"pseudo_{mun}",
            meta_common={"municipality": mun, "crs": cfg["project"]["crs_working"],
                         "date_start": f"{pcfg['year']}-05-01",
                         "date_end": f"{pcfg['year']}-10-31",
                         "year": pcfg["year"], "source": "pseudo"},
            transform=transform)
        records.extend(recs)
        del feats, label  # libera RAM antes do próximo município

    if not records:
        log.error("Nenhum pseudo-chip gerado.")
        return
    tmp = resolve_path(cfg, "metadata") / "_pseudo_tmp.parquet"
    ps = save_metadata(records, tmp)
    tmp.unlink(missing_ok=True)
    out = pd.concat([meta, ps], ignore_index=True)
    out.to_parquet(meta_path, index=False)
    log.info("Pseudo-chips: %d (com caju: %d). Total de chips: %d.", len(ps),
             int((~ps["is_negative"]).sum()), len(out))


if __name__ == "__main__":
    main()
