# Mapa único 2025 — caju, manga e coco sobre MapBiomas Coleção 11

Arquivo: `saidas/mapa/culturas_2025_mapbiomas_c11.tif` — GeoTIFF uint8, EPSG:32724 (UTM 24S), 10 m, paleta embutida.

## Valores das classes

| Valor | Classe | Cor | Origem |
|---|---|---|---|
| **97** | Caju | `#c0392b` | mapa final do projeto (Modelo_Caju), limiar 0.4 (0,30 no sertão CE/RN, fusão U-Net+XGB) |
| **98** | Manga | `#f39c12` | mapa final do projeto (Backup_2026), limiar 0.55 |
| **99** | Coco | `#16a085` | mapa final do projeto (Modelo_Coco), limiar 0.65 |
| 1–91 | Classes da legenda MapBiomas Coleção 11 (ex.: 3 Formação Florestal, 4 Savânica, 15 Pastagem, 21 Mosaico de Usos, 48 Outras Lavouras Perenes) | cores oficiais | MapBiomas 2025, 30 m reamostrado a 10 m (vizinho mais próximo) |
| 0 | Sem dado no MapBiomas | — | — |
| 255 | Fora dos municípios mapeados (NoData) | transparente | — |

Os valores 97, 98 e 99 **não constam** da legenda da Coleção 11 (conferido nas 5 abas da planilha `Legenda Coleção 11 - Descrição Detalhada.xlsx`); o mapa pode ser lido junto com qualquer produto MapBiomas.

## Regras

1. **Fundo**: classe MapBiomas 2025. Onde um modelo marcou caju, manga ou coco, a cultura substitui o fundo.
2. **Sobreposição** (pixel marcado por 2+ culturas): vence a maior margem normalizada `(p − limiar)/(1 − limiar)`, com `p` = probabilidade do modelo de cada cultura. A probabilidade bruta não é usada porque os limiares diferem (caju 0,30/0,40, manga 0,55, coco 0,65). As disputas ficam em `saidas/auxiliar/disputas_2025.tif` (soma de bits: 1 caju, 2 manga, 4 coco; 3 = caju×manga, 5 = caju×coco, 6 = manga×coco).
3. **Cobertura**: cada cultura só foi mapeada nos seus municípios (caju 25, manga 25, coco 15). Num município onde a cultura não foi mapeada, a tabela mostra "—" (não é ausência). Ver `saidas/tabelas/cobertura_municipios.csv`.

## Área por município (ha)

Mapa = área classificada; PAM = área destinada à colheita, média 2023–25 (IBGE 1613); Censo = área existente 2017 (IBGE 6955). Caju: PAM e Censo de castanha de caju. "—" = cultura não mapeada no município ou sem dado.

| UF | Município | Caju mapa | Caju PAM | Caju Censo | Manga mapa | Manga PAM | Manga Censo | Coco mapa | Coco PAM | Coco Censo | Disputa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BA | Abaré | — | — | — | 28 | 351 | 450 | — | 3 | 19 | 0 |
| BA | Barra da Estiva | — | — | — | 0 | 280 | 225 | — | — | 4 | 0 |
| BA | Carinhanha | — | — | — | 600 | 550 | 534 | — | — | — | 0 |
| BA | Casa Nova | — | — | — | 4.368 | 5.790 | 4.277 | 253 | 557 | 98 | 12 |
| BA | Curaçá | — | — | — | 62 | 1.076 | 1.408 | — | 156 | 152 | 0 |
| BA | Dom Basílio | — | — | — | 242 | 1.224 | 1.011 | — | — | — | 0 |
| BA | Ituaçu | — | — | — | 30 | 633 | 442 | — | 40 | 4 | 0 |
| BA | Juazeiro | — | — | — | 2.454 | 13.414 | 12.157 | 997 | 1.626 | 2.020 | 17 |
| BA | Livramento de Nossa Senhora | — | — | — | 3.447 | 6.543 | 5.369 | — | 5 | 2 | 0 |
| BA | Rio de Contas | — | — | — | 257 | 906 | 839 | — | 3 | 5 | 0 |
| BA | Rodelas | — | — | — | — | 75 | 98 | 3.068 | 2.357 | 2.370 | 0 |
| BA | Sento Sé | — | — | — | 862 | 1.082 | 955 | — | 23 | — | 0 |
| BA | Sobradinho | — | — | — | 180 | 409 | 399 | — | 45 | 52 | 0 |
| BA | Tanhaçu | — | — | — | 42 | 467 | 219 | — | 43 | 13 | 0 |
| CE | Acaraú | — | 5.504 | 3.196 | — | 66 | 19 | 624 | 6.806 | 4.499 | 0 |
| CE | Alto Santo | 6.985 | 10.057 | 3.316 | — | 2 | — | — | — | — | 0 |
| CE | Amontada | — | 7.156 | 5.236 | — | 8 | — | 217 | 2.273 | 1.174 | 0 |
| CE | Aracati | 7.779 | 9.192 | 5.387 | — | 45 | — | — | 494 | 36 | 0 |
| CE | Aracoiaba | 4.188 | 4.452 | 1.346 | — | 28 | — | — | 44 | — | 0 |
| CE | Barreira | 9.169 | 10.793 | 5.652 | — | 8 | — | — | 11 | — | 0 |
| CE | Beberibe | 25.688 | 27.948 | 19.697 | — | 287 | 156 | 2 | 2.362 | 685 | 47 |
| CE | Bela Cruz | 32.079 | 31.353 | 18.847 | — | 59 | 32 | — | 255 | 280 | 0 |
| CE | Camocim | — | 3.867 | 2.239 | — | 20 | — | 0 | 1.231 | 493 | 0 |
| CE | Cascavel | 9.155 | 9.499 | 5.346 | — | 243 | 45 | 11 | 1.598 | 215 | 7 |
| CE | Chorozinho | 11.264 | 13.510 | 6.705 | — | — | — | — | 13 | — | 0 |
| CE | Cruz | 13.444 | 12.462 | 7.095 | — | 7 | — | — | 241 | 181 | 0 |
| CE | Fortim | 2.742 | 6.230 | 1.532 | — | — | — | — | 194 | 93 | 0 |
| CE | Granja | 5.359 | 4.841 | 2.979 | — | 22 | — | — | 5 | — | 0 |
| CE | Icapuí | 579 | 7.435 | 2.732 | — | 9 | — | — | 594 | 21 | 0 |
| CE | Itapipoca | 13.390 | 16.195 | 11.774 | — | 114 | 23 | 87 | 3.012 | 1.881 | 61 |
| CE | Itarema | 9.316 | 12.626 | 4.192 | — | 28 | — | 96 | 5.439 | 6.487 | 142 |
| CE | Jijoca de Jericoacoara | 4.844 | 4.198 | 3.538 | — | 14 | — | — | 45 | 63 | 0 |
| CE | Marco | 4.480 | 7.899 | 3.181 | — | 30 | — | — | 61 | 60 | 0 |
| CE | Mauriti | — | 366 | 508 | 5 | 330 | 238 | — | 27 | 21 | 0 |
| CE | Ocara | 14.507 | 14.553 | 3.428 | — | 4 | — | — | 10 | — | 0 |
| CE | Pacajus | 5.796 | 5.550 | 6.807 | — | 5 | — | — | 4 | — | 0 |
| CE | Paraipaba | — | 90 | 279 | — | 19 | 2 | 3.841 | 6.109 | 2.651 | 0 |
| CE | Trairi | — | 11.949 | 8.879 | — | 16 | — | 541 | 8.282 | 4.485 | 0 |
| MG | Janaúba | — | — | — | 3.762 | 433 | 322 | — | 25 | 28 | 0 |
| MG | Jaíba | — | — | — | 2.399 | 1.008 | 1.363 | — | 8 | 3 | 0 |
| MG | Matias Cardoso | — | — | — | 2.410 | 760 | 766 | — | 6 | — | 0 |
| PE | Belém do São Francisco | — | — | — | 537 | 6.100 | 634 | — | 87 | 83 | 0 |
| PE | Buíque | 618 | 1.385 | 658 | — | — | 8 | — | — | — | 0 |
| PE | Cabrobó | — | — | — | 28 | 257 | 674 | — | 80 | 89 | 0 |
| PE | Lagoa Grande | — | — | — | 179 | 1.267 | 435 | — | 3 | 6 | 0 |
| PE | Orocó | — | — | — | 54 | 298 | 172 | — | 8 | 41 | 0 |
| PE | Petrolina | — | — | 5 | 4.303 | 10.617 | 8.266 | 2.246 | 2.133 | 1.801 | 171 |
| PE | Petrolândia | — | — | — | — | 125 | 188 | 4.802 | 3.717 | 1.652 | 0 |
| PE | Santa Maria da Boa Vista | — | — | — | 80 | 1.567 | 806 | — | — | 34 | 0 |
| PI | Monsenhor Hipólito | 4.357 | 2.667 | 3.470 | — | — | — | — | — | 26 | 0 |
| PI | Pio IX | 23.002 | 24.750 | 23.915 | — | — | — | — | — | 3 | 0 |
| RN | Apodi | 14.020 | 11.167 | 2.363 | — | 15 | 8 | — | 21 | 13 | 0 |
| RN | Açu | — | 120 | — | 6 | 260 | 262 | — | 10 | 2 | 0 |
| RN | Cerro Corá | 640 | 3.000 | 255 | — | 20 | 2 | — | 2 | — | 0 |
| RN | Ipanguaçu | — | 20 | — | 59 | 850 | 792 | — | — | — | 0 |
| RN | Serra do Mel | 23.343 | 37.600 | 16.061 | — | — | — | — | — | — | 0 |
| RN | Severiano Melo | 1.670 | 4.000 | 663 | — | — | — | — | — | — | 0 |
| RN | Touros | — | 700 | 1.613 | — | 40 | 10 | 6.268 | 6.633 | 2.814 | 0 |
| | **Total (municípios mapeados da cultura)** | **248.415** | **293.361** | **160.939** | **26.395** | **56.473** | **43.015** | **23.053** | **54.136** | **33.325** | **458** |

Tabela completa (com razões mapa/PAM e área do município): `saidas/tabelas/tabela_final_municipios.csv`.