"""Utilidades comuns: config e caminhos (nada de caminho fixo nos scripts)."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]


def cfg():
    with open(RAIZ / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(*partes):
    q = RAIZ.joinpath(*partes)
    q.parent.mkdir(parents=True, exist_ok=True)
    return q


def municipios_por_cultura(c):
    """{cultura: [cod IBGE int]} a partir das saídas por município de cada projeto."""
    out = {}
    for k, v in c["culturas"].items():
        if "municipios_dir" in v:
            out[k] = sorted({int(f.name[1:8]) for f in Path(v["municipios_dir"]).glob("m*_prob.tif")})
        else:
            out[k] = sorted({int(l.split()[0][:7]) for l in Path(v["municipios_lista"]).read_text().splitlines() if l.strip()})
    return out
