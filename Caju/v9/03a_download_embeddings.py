"""03a (v7) — Baixa os embeddings AlphaEarth na grade de cada COG Sentinel.

Sem tarefas no Earth Engine (computePixels síncrono, em tiles). Retomável.
Alvos: COGs de treino (data/processed/*_<ano>.tif) e de predição (data/predictions/features_*.tif).

    python 03a_download_embeddings.py --project SEU_PROJETO_GCP
"""
from __future__ import annotations

import argparse

from src.config import ensure_dirs, load_config, resolve_path, setup_logging
from src.embeddings import download_embeddings_for


def main() -> None:
    p = argparse.ArgumentParser(description="Embeddings AlphaEarth alinhados (v7).")
    p.add_argument("--config", default=None)
    p.add_argument("--project", default=None, help="Projeto GCP registrado no Earth Engine.")
    args = p.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = setup_logging("03a_download_embeddings", cfg)
    import ee
    project = args.project or cfg["embeddings"].get("project") or None
    ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    y = int(cfg["embeddings"]["year"])
    n = (ee.ImageCollection(cfg["embeddings"]["collection"])
         .filterDate(f"{y}-01-01", f"{y + 1}-01-01").size().getInfo())
    if n == 0:
        log.error("Sem embeddings para %d. Ajuste embeddings.year no config.", y)
        raise SystemExit(2)
    log.info("Embeddings %d: %d imagens na coleção.", y, n)
    targets = sorted(resolve_path(cfg, "processed").glob(f"features_*_{cfg['temporal']['years'][0]}.tif"))
    targets += sorted(resolve_path(cfg, "predictions").glob("features_*.tif"))
    log.info("%d COGs alvo.", len(targets))
    for cog in targets:
        download_embeddings_for(cog, cfg)
    log.info("Embeddings concluídos.")


if __name__ == "__main__":
    main()
