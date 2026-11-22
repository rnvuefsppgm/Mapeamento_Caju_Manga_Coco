"""Utilidades comuns: config e caminhos (nada de caminho fixo nos scripts)."""
from pathlib import Path

import yaml

RAIZ_CODIGO = Path(__file__).resolve().parents[1]


def cfg():
    with open(RAIZ_CODIGO / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ext(c, *partes):
    """Pasta compartilhada D:/Modelo_Coco/dados_externos/..."""
    p = Path(c["data_root"], "dados_externos", *partes)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ver(c, *partes):
    """Pasta da versão D:/Modelo_Coco/<versao>/..."""
    p = Path(c["data_root"], c["project"]["version"], *partes)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
