"""Ponte para os scripts portados do coco: cfg() devolve a seção `amostragem` do config.yaml do projeto manga e
ver() traduz os caminhos do coco (v1/data/...) para os do manga (dados/...)."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
_MAPA = {("data", "embeddings", "2025"): ("dados", "embeddings"), ("data", "aoi", "regions"): ("dados", "aoi", "regioes"),
         ("data",): ("dados",)}


def cfg():
    with open(RAIZ / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["amostragem"]


def ver(c, *partes):
    for k, v in _MAPA.items():
        if partes[:len(k)] == k:
            partes = v + partes[len(k):]
            break
    p = RAIZ.joinpath(*partes)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
