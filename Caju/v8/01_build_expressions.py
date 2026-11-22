"""01 (v8) — Expressões Earth Engine (2025) + regiões para exportação via DatensEE.

Gera (rodar com D:/Modelo_Caju/.venv, que tem earthengine-api):
  expressions/embeddings_2025.json  64 bandas uint8 (valor*127+128; NoData 0)
  expressions/sentinel_2025.json    49 bandas int16 (NoData -32768) — ver BANDAS abaixo
  expressions/*_bands.json          nomes e escalas das bandas
  data/aoi/regions/<id>.geojson     25 municípios (buffer 1 km) + 3 clusters de treino (v7)

Sentinel no EE (≠ v7/Planetary Computer: S1 é GRD, não RTC) -> modelos Sentinel são retreinados.
  S2 SR harmonizado, SCL válido, mai–out/2025: 10 bandas x p25/p50/p75 (x10000) = 30
  6 índices (mediana, x10000): NDVI, NDRE, NDMI, EVI, NBR, NDWI           = 6
  S1 GRD IW descendente (dB): VV, VH x p25/p50/p75 (x100) + VH-VV p50     = 7
  Textura GLCM (3x3) sobre VV/VH p50: contrast, idm, diss (x100)          = 6   -> 49
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

ROOT = Path(__file__).parent
Y, S0, S1_ = 2025, "2025-05-01", "2025-11-01"
PROJECT = "rnv-geodatin"
NB_I16 = -32768


def embeddings():
    img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filterDate(f"{Y}-01-01", f"{Y + 1}-01-01").mosaic())
    names = img.bandNames()
    # uint8 (DatensEE não aceita int8): v = round(x*127)+128 ; 0 = NoData ; decodificar (v-128)/127
    return img.multiply(127).round().clamp(-127, 127).add(128).unmask(0).toUint8(), names


def sentinel():
    b = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    def s2mask(i):
        scl = i.select("SCL")
        ok = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)) \
            .And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        r = i.select(b).multiply(0.0001).updateMask(ok)
        nd = lambda x, y: r.normalizedDifference([x, y])
        evi = r.expression("2.5*(N-R)/(N+6*R-7.5*B+1)",
                           {"N": r.select("B8"), "R": r.select("B4"), "B": r.select("B2")})
        idx = ee.Image.cat(nd("B8", "B4"), nd("B8", "B5"), nd("B8", "B11"), evi,
                           nd("B8", "B12"), nd("B3", "B8")).rename(
            ["NDVI", "NDRE", "NDMI", "EVI", "NBR", "NDWI"])
        return r.addBands(idx)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(S0, S1_)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)).map(s2mask))
    pct = s2.select(b).reduce(ee.Reducer.percentile([25, 50, 75]))
    idx = s2.select(["NDVI", "NDRE", "NDMI", "EVI", "NBR", "NDWI"]).median()
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(S0, S1_)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))
    s1p = s1.reduce(ee.Reducer.percentile([25, 50, 75]))
    ratio = s1p.select("VH_p50").subtract(s1p.select("VV_p50")).rename("VHmVV_p50")
    q = s1p.select(["VV_p50", "VH_p50"]).add(30).multiply(4).clamp(0, 255).toInt()
    tex = q.glcmTexture(size=1).select(
        ["VV_p50_contrast", "VV_p50_idm", "VV_p50_diss",
         "VH_p50_contrast", "VH_p50_idm", "VH_p50_diss"])
    groups = [(pct, 10000), (idx, 10000), (s1p, 100), (ratio, 100), (tex, 100)]
    out = ee.Image.cat([g.multiply(s) for g, s in groups])
    out = out.round().clamp(-32767, 32767).unmask(NB_I16).toInt16()
    return out, out.bandNames()


def _write_geom(geom, path):
    """DatensEE exige GeoJSON de geometria pura (Polygon/MultiPolygon), não FeatureCollection."""
    from shapely.geometry import mapping
    path.write_text(json.dumps(mapping(geom)), encoding="utf-8")


def regions():
    rd = ROOT / "data" / "aoi" / "regions"
    m = gpd.read_file(ROOT.parent / "dados_externos" / "municipios25" / "municipios25.gpkg")
    m = m.to_crs(32724)
    m["geometry"] = m.buffer(1000)
    ids = []
    for _, r in m.iterrows():
        rid = f"m{r['cod']}_" + "".join(ch for ch in unicodedata.normalize("NFKD", str(r["NM_MUN"])).encode("ascii", "ignore").decode() if ch.isalnum())[:20]
        _write_geom(gpd.GeoSeries([r.geometry], crs=32724).to_crs(4326).iloc[0], rd / f"{rid}.geojson")
        ids.append(rid)
    prov = pd.read_parquet(ROOT.parent / "v7" / "metadata" / "training_provenance_2025.parquet")
    for cog in prov["features_cog"]:
        with rasterio.open(cog) as d:
            g = box(*d.bounds)
        rid = "treino_" + Path(cog).stem.replace("features_", "").replace("_2025", "")
        _write_geom(gpd.GeoSeries([g], crs=32724).to_crs(4326).iloc[0], rd / f"{rid}.geojson")
        ids.insert(0, rid)
    (ROOT / "data" / "aoi" / "regions.txt").write_text("\n".join(ids), encoding="utf-8")
    return ids


def main():
    ee.Initialize(project=PROJECT)
    ed = ROOT / "expressions"
    for name, fn in [("embeddings", embeddings)]:  # v8: só embeddings (Sentinel EE em ../v8_sentinel)
        img, bn = fn()
        (ed / f"{name}_{Y}.json").write_text(ee.serializer.toJSON(img), encoding="utf-8")
        (ed / f"{name}_{Y}_bands.json").write_text(json.dumps(bn.getInfo()), encoding="utf-8")
        print(name, len(bn.getInfo()), "bandas")
    print(len(regions()), "regiões")


if __name__ == "__main__":
    main()
