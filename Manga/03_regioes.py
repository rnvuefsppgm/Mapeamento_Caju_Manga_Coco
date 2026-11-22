"""03 — Regiões de exportação (DatensEE): um GeoJSON de geometria pura (EPSG:4326) por município selecionado,
com buffer de 1 km (igual ao caju v8). As amostras estão dentro de Livramento, então não há região de treino à parte.
Também copia a expressão EE dos embeddings 2025 do caju (idêntica: uint8 = round(x*127)+128, 0 = NoData).
Saídas: dados/aoi/regioes/<id>.geojson, dados/aoi/regioes.txt (menor -> maior, p/ testar cedo), dados/aoi/expressions/
"""
from __future__ import annotations

import json
import shutil
import unicodedata
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping

from src.config import load_config, resolve_path, setup_logging

cfg = load_config(Path(__file__).parent / "config/config.yaml")
log = setup_logging("03_regioes", cfg)
M = cfg["manga"]
EXPR = Path("D:/Modelo_Caju/v8/expressions")


def rid(cod, nome):
    s = "".join(ch for ch in unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode() if ch.isalnum())
    return f"m{cod}_{s[:20]}"


def main():
    aoi = resolve_path(cfg, "aoi")
    rd = aoi / "regioes"
    g = gpd.read_file(resolve_path(cfg, "pam") / "municipios_selecionados.gpkg").to_crs(M["crs"])
    g["rid"] = [rid(c, n) for c, n in zip(g.cod, g.NM_MUN)]
    g["area_buf_km2"] = g.buffer(M["regioes"]["buffer_municipio_m"]).area / 1e6
    g = g.sort_values("area_buf_km2")
    for r in g.itertuples():
        geom = gpd.GeoSeries([r.geometry.buffer(M["regioes"]["buffer_municipio_m"])], crs=M["crs"]).to_crs(4326).iloc[0]
        (rd / f"{r.rid}.geojson").write_text(json.dumps(mapping(geom)), encoding="utf-8")
    (aoi / "regioes.txt").write_text("\n".join(g.rid) + "\n", encoding="utf-8")  # termina com \n (bug do caju v8)
    g.drop(columns="geometry").to_csv(aoi / "regioes.csv", index=False, encoding="utf-8-sig")
    (aoi / "expressions").mkdir(exist_ok=True)
    for f in ["embeddings_2025.json", "embeddings_2025_bands.json"]:
        shutil.copy(EXPR / f, aoi / "expressions" / f)
    tot = g.area_buf_km2.sum()
    log.info(f"RESUMO: {len(g)} regiões; {tot:,.0f} km² com buffer; embeddings estimados ~{tot * 0.6 / 1e3:,.0f} GB "
             f"(caju: 24.914 km² -> 15 GB); maior: {g.rid.iloc[-1]} {g.area_buf_km2.iloc[-1]:,.0f} km²")


if __name__ == "__main__":
    main()
