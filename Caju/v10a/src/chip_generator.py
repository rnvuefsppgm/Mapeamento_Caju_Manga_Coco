"""Geração de chips multissensores com controle de qualidade (ETAPAS 17-20).

Cada chip guarda correspondência perfeita entre features (S2+S1) e label, pois
ambos vêm do MESMO grid alinhado. Chips são salvos como .npz (features float32 +
label uint8) e uma tabela de metadados é acumulada (salva em Parquet).

QC descarta chips com fração de pixels válidos abaixo do limiar (muito NoData/
nuvem). A geração de negativos é feita amostrando chips fora dos polígonos de
caju (background), incluindo áreas heterogêneas — hard negatives podem ser
adicionados depois via ``02_create_training_chips.py --hard-negatives``.

SPLIT ESPACIAL (CORREÇÃO ponto B): o ``block_id`` de cada chip é derivado da
GEOLOCALIZAÇÃO do centro do chip, no tamanho de bloco de ``split.block_size_km``
(ex.: 20 km), usando a mesma convenção de ``geo_utils.group_by_spatial_blocks``.
Assim o split treino/val/teste respeita blocos reais de 20 km — e não o bloco de
40 km do clustering de consulta STAC. Requer o ``transform`` do COG (CRS métrico).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ChipRecord:
    """Metadados de um chip (linha da tabela de metadados — ETAPA 20)."""

    chip_id: str
    path: str
    municipality: str
    date_start: str
    date_end: str
    valid_fraction: float
    cashew_fraction: float
    background_fraction: float
    n_channels: int
    chip_size: int
    resolution: int
    crs: str
    block_id: str = ""
    s2_item_ids: str = ""
    s1_item_ids: str = ""
    orbit_direction: str = ""
    is_negative: bool = False
    year: int = 0                 # v6: ano do composto (multi-ano)
    source: str = "field"         # v6: "field" (polígonos) | "pseudo" (pseudo-rótulo)
    extra: Dict[str, Any] = field(default_factory=dict)


def _tiles(height: int, width: int, size: int, stride: int):
    """Gera janelas (row0, col0) cobrindo o raster com passo ``stride``."""
    for r in range(0, max(1, height - size + 1), stride):
        for c in range(0, max(1, width - size + 1), stride):
            yield r, c
    # garante cobertura das bordas inferior/direita
    if (height - size) % stride != 0:
        for c in range(0, max(1, width - size + 1), stride):
            yield height - size, c
    if (width - size) % stride != 0:
        for r in range(0, max(1, height - size + 1), stride):
            yield r, width - size


def _spatial_block_id(transform, r0: int, c0: int, size: int,
                      block_size_m: float) -> str:
    """block_id do chip pela geolocalização do seu centro (bloco de N km).

    Usa a mesma convenção de ``geo_utils.group_by_spatial_blocks`` (floor das
    coordenadas métricas divididas pelo tamanho do bloco), de modo que chips
    vizinhos do mesmo bloco de 20 km caem sempre na mesma partição do split.
    """
    cx, cy = transform * (c0 + size / 2.0, r0 + size / 2.0)
    ix = int(np.floor(cx / block_size_m))
    iy = int(np.floor(cy / block_size_m))
    return f"blk_{ix}_{iy}"


def generate_chips(
    features: np.ndarray,           # (C, H, W) float32, NaN = inválido
    label: np.ndarray,              # (H, W) uint8 {0,1,255}
    cfg: Dict[str, Any],
    out_dir: Path,
    prefix: str,
    meta_common: Dict[str, Any],
    transform: Optional[Any] = None,   # affine do COG (CRS métrico) p/ block_id
) -> List[ChipRecord]:
    """Extrai chips de um par (features, label) alinhados e aplica QC.

    Returns:
        Lista de ChipRecord dos chips aprovados no QC.
    """
    size = cfg["chips"]["size"]
    stride = cfg["chips"]["stride"]
    min_valid = cfg["chips"]["min_valid_fraction"]
    pos_val = cfg["chips"]["label_positive_value"]
    bg_val = cfg["chips"]["label_background_value"]
    ignore_val = cfg["chips"]["label_ignore_value"]
    block_size_m = cfg["split"]["block_size_km"] * 1000.0

    out_dir.mkdir(parents=True, exist_ok=True)
    c_channels, height, width = features.shape
    records: List[ChipRecord] = []

    # Materializa as janelas para exibir total na barra de progresso.
    tiles = list(_tiles(height, width, size, stride))
    for idx, (r0, c0) in enumerate(
            tqdm(tiles, desc=f"Chips {prefix}", unit="tile", leave=False)):
        fchip = features[:, r0:r0 + size, c0:c0 + size]
        lchip = label[r0:r0 + size, c0:c0 + size]
        if fchip.shape[1:] != (size, size):
            continue

        valid = np.isfinite(fchip).all(axis=0)
        valid_fraction = float(valid.mean())
        if valid_fraction < min_valid:
            continue  # QC: muito NoData/nuvem

        cashew_fraction = float((lchip == pos_val).mean())
        bg_fraction = float((lchip == bg_val).mean())
        is_negative = cashew_fraction == 0.0

        # substitui NaN por 0 após registrar a máscara de validade
        # v6: float16 (metade do disco/IO); dataset volta para float32 ao ler.
        fchip = np.nan_to_num(fchip, nan=0.0).astype("float16")
        if meta_common.get("source") == "pseudo":
            # pseudo-chip só vale se tiver pixels confiáveis suficientes
            if float((lchip != ignore_val).mean()) < cfg.get("pseudo_labels", {}).get(
                    "min_confident_fraction", 0.5):
                continue

        # block_id espacial real do chip (bloco de split.block_size_km); se o
        # transform não for fornecido, cai no block_id do cluster (fallback).
        if transform is not None:
            block_id = _spatial_block_id(transform, r0, c0, size, block_size_m)
        else:
            block_id = meta_common.get("block_id", "")

        chip_id = f"{prefix}_{idx:06d}"
        chip_path = out_dir / f"{chip_id}.npz"
        np.savez(  # v6: sem compressão -> leitura muito mais rápida no treino
            chip_path,
            features=fchip,
            label=lchip.astype("uint8"),
            valid=valid.astype("uint8"),
        )

        records.append(ChipRecord(
            chip_id=chip_id,
            path=str(chip_path),
            municipality=meta_common.get("municipality", ""),
            date_start=meta_common.get("date_start", ""),
            date_end=meta_common.get("date_end", ""),
            valid_fraction=valid_fraction,
            cashew_fraction=cashew_fraction,
            background_fraction=bg_fraction,
            n_channels=c_channels,
            chip_size=size,
            resolution=cfg["harmonization"]["resolution"],
            crs=meta_common.get("crs", ""),
            block_id=block_id,
            s2_item_ids=meta_common.get("s2_item_ids", ""),
            s1_item_ids=meta_common.get("s1_item_ids", ""),
            orbit_direction=meta_common.get("orbit_direction", ""),
            is_negative=is_negative,
            year=int(meta_common.get("year", 0) or 0),
            source=meta_common.get("source", "field"),
        ))

    logger.info("%s: %d chips aprovados no QC.", prefix, len(records))
    return records


def save_metadata(records: List[ChipRecord], path: Path) -> pd.DataFrame:
    """Salva a tabela de metadados dos chips em Parquet (ETAPA 20)."""
    rows = []
    for rec in records:
        d = rec.__dict__.copy()
        d.pop("extra", None)
        rows.append(d)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Metadados dos chips salvos: %s (%d linhas).", path.name, len(df))
    return df
