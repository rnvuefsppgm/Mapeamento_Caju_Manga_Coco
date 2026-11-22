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

## Todas as classes

| Cor | Código | Classe | Class (en) | Hex | R, G, B | Área no mapa (ha) |
|---|---|---|---|---|---|---|
| ![#1f8d49](cores/3.png) | **3** | Formação Florestal | Forest Formation | `#1f8d49` | 31, 141, 73 | 335.556 |
| ![#7dc975](cores/4.png) | **4** | Formação Savânica | Savanna Formation | `#7dc975` | 125, 201, 117 | 5.531.949 |
| ![#04381d](cores/5.png) | **5** | Mangue | Mangrove | `#04381d` | 4, 56, 29 | 11.766 |
| ![#7a5900](cores/9.png) | **9** | Silvicultura | Forest Plantation | `#7a5900` | 122, 89, 0 | 1.208 |
| ![#519799](cores/11.png) | **11** | Campo Alagado e Área Pantanosa | Wetland | `#519799` | 81, 151, 153 | 520 |
| ![#d6bc74](cores/12.png) | **12** | Formação Campestre | Grassland Formation | `#d6bc74` | 214, 188, 116 | 51.344 |
| ![#edde8e](cores/15.png) | **15** | Pastagem | Pasture | `#edde8e` | 237, 222, 142 | 1.723.241 |
| ![#db7093](cores/20.png) | **20** | Cana | Sugar cane | `#db7093` | 219, 112, 147 | 29.442 |
| ![#ffefc3](cores/21.png) | **21** | Mosaico de Usos | Mosaic of Uses | `#ffefc3` | 255, 239, 195 | 557.227 |
| ![#ffa07a](cores/23.png) | **23** | Praia, Duna e Areal | Beach, Dune and Sand Spot | `#ffa07a` | 255, 160, 122 | 49.519 |
| ![#d4271e](cores/24.png) | **24** | Área Urbanizada | Urban Area | `#d4271e` | 212, 39, 30 | 71.777 |
| ![#db4d4f](cores/25.png) | **25** | Outras Áreas não Vegetadas | Other non Vegetated Areas | `#db4d4f` | 219, 77, 79 | 52.169 |
| ![#ad5100](cores/29.png) | **29** | Afloramento Rochoso | Rocky Outcrop | `#ad5100` | 173, 81, 0 | 12.713 |
| ![#9c0027](cores/30.png) | **30** | Mineração | Mining | `#9c0027` | 156, 0, 39 | 147 |
| ![#091077](cores/31.png) | **31** | Aquicultura | Aquaculture | `#091077` | 9, 16, 119 | 8.867 |
| ![#fc8114](cores/32.png) | **32** | Apicum | Hypersaline Tidal Flat | `#fc8114` | 252, 129, 20 | 2.010 |
| ![#2532e4](cores/33.png) | **33** | Rio, Lago e Oceano | River, Lake and Ocean | `#2532e4` | 37, 50, 228 | 359.474 |
| ![#f5b3c8](cores/39.png) | **39** | Soja | Soybean | `#f5b3c8` | 245, 179, 200 | 712 |
| ![#f54ca9](cores/41.png) | **41** | Outras Lavouras Temporárias | Other Temporary Crops | `#f54ca9` | 245, 76, 169 | 200.001 |
| ![#d68fe2](cores/46.png) | **46** | Café | Coffee | `#d68fe2` | 214, 143, 226 | 5.435 |
| ![#e6ccff](cores/48.png) | **48** | Outras Lavouras Perenes | Other Perennial Crops | `#e6ccff` | 230, 204, 255 | 267.246 |
| ![#02d659](cores/49.png) | **49** | Restinga Arbórea | Wooded Sandbank Vegetation | `#02d659` | 2, 214, 89 | 30 |
| ![#ffaa5f](cores/50.png) | **50** | Restinga Herbácea ou Arbustiva | Herbaceous Sandbank Vegetation | `#ffaa5f` | 255, 170, 95 | 5.300 |
| ![#757272](cores/75.png) | **75** | Usina Fotovoltaica | Photovoltaic Power Plant | `#757272` | 117, 114, 114 | 13.942 |
| ![#86b074](cores/77.png) | **77** | Formação Herbáceo Arbustiva | Herbaceous and Shrub Formation | `#86b074` | 134, 176, 116 | 423.628 |
| ![#403d3e](cores/91.png) | **91** | Parque eólico (beta) | Wind farm (beta) | `#403d3e` | 64, 61, 62 | 4.719 |
| ![#c0392b](cores/97.png) | **97** | **Caju** | Cashew | `#c0392b` | 192, 57, 43 | 248.414 |
| ![#f39c12](cores/98.png) | **98** | **Manga** | Mango | `#f39c12` | 243, 156, 18 | 26.395 |
| ![#16a085](cores/99.png) | **99** | **Coco-da-baía** | Coconut | `#16a085` | 22, 160, 133 | 23.053 |

Em negrito: classes novas (não constam da legenda MapBiomas Coleção 11).
