# Legendas — mapa único caju + manga + coco 2025

Legendas do raster `Culturas/saidas/mapa/culturas_2025_mapbiomas_c11.tif` (MapBiomas Coleção 11 + culturas).
Nomes e cores das classes MapBiomas vêm da planilha oficial da Coleção 11; as classes novas não constam dela.

| Arquivo | Uso |
|---|---|
| `culturas_2025_mapbiomas_c11.clr` | colormap do ArcMap (`valor R G B`): Symbology → Colormap → Import |
| `culturas_2025_mapbiomas_c11_nomes.clr` | colormap com nomes (`valor R G B nome`) para ArcGIS Pro / QGIS |
| `culturas_2025_mapbiomas_c11.tif.vat.dbf` | tabela de atributos (Value, Count, Classe, Red, Green, Blue; ANSI 1252). Copie ao lado do .tif para ver os nomes no ArcMap (Unique Values pelo campo `Classe`) |
| `legenda_classes.csv` | código, nome (pt/en), cor hex/RGB e área no mapa (ha) |

| Código | Classe nova | Cor |
|---|---|---|
| 97 | Caju | `#c0392b` |
| 98 | Manga | `#f39c12` |
| 99 | Coco-da-baía | `#16a085` |

Contém as 26 classes MapBiomas presentes nos 58 municípios mapeados + as 3 culturas (29 classes).
Gerado por `Culturas/scripts/04_legendas.py`.
