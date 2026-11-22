"""01 (v9) — Expressão EE da MapBiomas Col.10 (classificação 2024) p/ negativos confiáveis. Rodar com .venv."""
import ee
from pathlib import Path
ee.Initialize(project="rnv-geodatin")
img = ee.Image("projects/mapbiomas-public/assets/brazil/lulc/collection10/"
               "mapbiomas_brazil_collection10_integration_v2").select("classification_2024")
Path(__file__).with_name("mapbiomas_2024.json").write_text(ee.serializer.toJSON(img.unmask(0).toUint8()), encoding="utf-8")
print("ok")
