"""01 — Amostras de manga: KML/KMZ de `entrada/` -> `dados/amostras/manga_poligonos.gpkg`.

Lê todos os .kml/.kmz (detecta zip pelo conteúdo, pois MANGA_3.kml é um KMZ), extrai os polígonos,
corrige geometrias, remove duplicatas (mesma geometria em mais de um arquivo), mede área em EPSG:32724
e associa o município (BR_Municipios_2024). Resumo em reports/01_amostras.md.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.validation import make_valid

from src.config import load_config, resolve_path, setup_logging

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("01_amostras", cfg)
M = cfg["manga"]


def kml_text(p: Path) -> str:
    raw = p.read_bytes()
    if raw[:2] == b"PK":
        z = zipfile.ZipFile(p)
        return z.read(next(n for n in z.namelist() if n.lower().endswith(".kml"))).decode("utf8", "ignore")
    return raw.decode("utf8", "ignore")


def coords(s: str):
    pts = [tuple(map(float, c.split(",")[:2])) for c in s.split()]
    return pts if len(pts) >= 4 else None


def parse(p: Path):
    t = kml_text(p)
    out = []
    for pm in re.findall(r"<Placemark.*?</Placemark>", t, re.S):
        nome = (re.search(r"<name>(.*?)</name>", pm, re.S) or [None, ""])[1].strip()
        for poly in re.findall(r"<Polygon.*?</Polygon>", pm, re.S):
            ob = re.search(r"<outerBoundaryIs>.*?<coordinates>(.*?)</coordinates>", poly, re.S)
            ins = re.findall(r"<innerBoundaryIs>.*?<coordinates>(.*?)</coordinates>", poly, re.S)
            ext = coords(ob.group(1)) if ob else None
            if ext:
                out.append({"arquivo": p.name, "nome": nome,
                            "geometry": Polygon(ext, [c for c in map(coords, ins) if c])})
    return out


def main():
    ent = resolve_path(cfg, "entrada")
    rows = []
    for p in sorted(list(ent.glob("*.kml")) + list(ent.glob("*.kmz"))):
        r = parse(p)
        log.info(f"{p.name}: {len(r)} polígonos")
        rows += r
    g = gpd.GeoDataFrame(rows, crs=4326)
    n0 = len(g)
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
    g = g[~g.geometry.is_empty]
    g = g.explode(index_parts=False)
    g = g[g.geom_type == "Polygon"].to_crs(M["crs"])
    # duplicatas: mesma geometria (arredondada a 0,1 m) em mais de um arquivo
    g["k"] = g.geometry.apply(lambda x: tuple(round(v, 1) for v in x.bounds) + (round(x.area, 0),))
    g = g.drop_duplicates("k").drop(columns="k")
    g["area_ha"] = g.area / 1e4
    mun = gpd.read_file(M["fontes"]["municipios_shp"]).to_crs(M["crs"])
    j = gpd.sjoin(g.assign(geometry=g.representative_point()), mun[["CD_MUN", "NM_MUN", "SIGLA_UF", "geometry"]],
                  how="left", predicate="within")
    g[["cod", "municipio", "uf"]] = j[["CD_MUN", "NM_MUN", "SIGLA_UF"]].values
    g = g.reset_index(drop=True)
    g["id"] = range(1, len(g) + 1)
    # sobreposição entre polígonos que sobraram
    uni = g.union_all().area / 1e4
    out = resolve_path(cfg, "amostras") / "manga_poligonos.gpkg"
    g.to_file(out, driver="GPKG")
    by = g.groupby(["cod", "municipio", "uf"]).agg(n=("id", "size"), area_ha=("area_ha", "sum")).reset_index()
    bb = g.total_bounds
    rep = resolve_path(cfg, "reports") / "01_amostras.md"
    rep.write_text(
        "# 01 — Amostras de manga\n\n"
        f"Lidos {n0} polígonos; {len(g)} após correção e remoção de duplicatas. "
        f"Área somada {g.area_ha.sum():,.1f} ha; área sem sobreposição {uni:,.1f} ha.\n\n"
        f"Área por polígono (ha): mín {g.area_ha.min():.2f} · mediana {g.area_ha.median():.2f} · máx {g.area_ha.max():.1f}\n\n"
        f"Extensão (EPSG:32724): {(bb[2]-bb[0])/1e3:.1f} × {(bb[3]-bb[1])/1e3:.1f} km\n\n"
        "| Arquivo | Polígonos lidos |\n|---|---|\n"
        + "".join(f"| {a} | {c} |\n" for a, c in pd.DataFrame(rows).arquivo.value_counts().items())
        + "\n| Município | UF | Polígonos | Área (ha) |\n|---|---|---|---|\n"
        + "".join(f"| {r.municipio} ({r.cod}) | {r.uf} | {r.n} | {r.area_ha:,.1f} |\n" for r in by.itertuples()),
        encoding="utf-8")
    log.info(f"RESUMO: {n0} lidos -> {len(g)} únicos; {g.area_ha.sum():,.0f} ha (sem sobreposição {uni:,.0f}); "
             f"municípios: {', '.join(by.municipio + ' ' + by.n.astype(str))}")


if __name__ == "__main__":
    main()
