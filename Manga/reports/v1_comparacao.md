# Manga 2025 — v1 (rede treinada só com amostras da Chapada)

Total mapa 26,551 ha x PAM 60,129 ha (razão 0.44); correlação com a PAM 0.63. Municípios 'extrapolação' estão fora da paisagem das amostras: valores provisórios.

| municipio                        | tipo         |   manga_ha |   pam_2025_ha |   censo_2017_ha |   mapa/PAM |   a5_confiavel_ha |   manga_ha_sem_mascara |
|:---------------------------------|:-------------|-----------:|--------------:|----------------:|-----------:|------------------:|-----------------------:|
| Juazeiro - BA                    | extrapolação |     2462.6 |         14003 |           12157 |       0.18 |              0    |                 5166   |
| Petrolina - PE                   | extrapolação |     4443.7 |         11450 |            8266 |       0.39 |              0    |                 4696.7 |
| Livramento de Nossa Senhora - BA | confiável    |     3446.9 |          6630 |            5369 |       0.52 |           2078.59 |                 3532.8 |
| Casa Nova - BA                   | extrapolação |     4375.5 |          6570 |            4277 |       0.67 |              0    |                 4466.9 |
| Belém do São Francisco - PE      | extrapolação |      537.3 |          6400 |             634 |       0.08 |              0    |                  575.2 |
| Lagoa Grande - PE                | extrapolação |      179.4 |          1900 |             435 |       0.09 |              0    |                  196   |
| Santa Maria da Boa Vista - PE    | extrapolação |       80.3 |          1700 |             806 |       0.05 |              0    |                   86.4 |
| Dom Basílio - BA                 | confiável    |      242   |          1224 |            1011 |       0.2  |             85.61 |                  253.3 |
| Jaíba - MG                       | extrapolação |     2398.8 |          1075 |            1363 |       2.23 |             12.39 |                 5670   |
| Sento Sé - BA                    | extrapolação |      862.1 |           955 |             955 |       0.9  |              0    |                  870.1 |
| Curaçá - BA                      | extrapolação |       62   |           945 |            1408 |       0.07 |              0    |                   73.5 |
| Rio de Contas - BA               | confiável    |      257.3 |           918 |             839 |       0.28 |            nan    |                  283.5 |
| Ipanguaçu - RN                   | extrapolação |       59.2 |           850 |             792 |       0.07 |              0    |                   73   |
| Ituaçu - BA                      | confiável    |       29.6 |           800 |             442 |       0.04 |              1.04 |                   32.1 |
| Matias Cardoso - MG              | extrapolação |     2410.2 |           750 |             766 |       3.21 |              0    |                 2630.4 |
| Carinhanha - BA                  | extrapolação |      600.2 |           600 |             534 |       1    |            nan    |                  663.8 |
| Tanhaçu - BA                     | confiável    |       42.1 |           600 |             219 |       0.07 |              0    |                   42.4 |
| Janaúba - MG                     | extrapolação |     3762.4 |           450 |             322 |       8.36 |            nan    |                 3944.2 |
| Orocó - PE                       | extrapolação |       53.7 |           420 |             172 |       0.13 |            nan    |                   55.3 |
| Cabrobó - PE                     | extrapolação |       27.8 |           370 |             674 |       0.08 |            nan    |                   28.6 |
| Abaré - BA                       | extrapolação |       27.8 |           364 |             450 |       0.08 |            nan    |                   32.8 |
| Barra da Estiva - BA             | confiável    |        0   |           300 |             225 |       0    |            nan    |                    0   |
| Sobradinho - BA                  | extrapolação |      179.5 |           300 |             399 |       0.6  |            nan    |                  192   |
| Mauriti - CE                     | extrapolação |        5.3 |           295 |             238 |       0.02 |            nan    |                   11.6 |
| Assú - RN                        | extrapolação |        5.6 |           260 |             262 |       0.02 |            nan    |                    6.2 |

## Leitura (21/11/2026)
- **Treino**: CV espacial em Livramento (rótulos conferidos), IoU 0,694 · F1 0,82 · P 0,82 · R 0,82 (pesos do caju sem cabeça; ImageNet = 0,302).
- **Chapada**: Livramento 0,52× PAM (0,64× Censo); Dom Basílio 0,20; Rio de Contas 0,28; Ituaçu, Tanhaçu e Barra da Estiva quase nada → a rede pega os pomares parecidos com os de Livramento e perde os menores/diferentes.
- **Vale do S. Francisco**: Casa Nova ≈ Censo (1,02), Petrolina 0,54× Censo, Sento Sé 0,90, Juazeiro 0,18 (a máscara MapBiomas tirou 2.549 ha em cana, classe 20 — correto), Belém/Lagoa Grande perto do Censo, Curaçá/Sta. Maria/Cabrobó/Abaré/Orocó muito abaixo.
- **Norte de MG (não confiável)**: Jaíba 2,2×, Matias Cardoso 3,2×, Janaúba 8,4× a PAM — confusão com banana/limão irrigados. Precisa de amostras próprias.
- **RN/CE**: quase nada (Ipanguaçu, Assú, Mauriti).

Produtos: raster único `dados/predicoes/manga_2025_25mun_{classe,prob}.tif`; por município `dados/predicoes/v1/<rid>_{class,prob}.tif`;
vetores `dados/predicoes/v1/vetores/manga_2025_25mun.{kmz,gpkg}` (2.253 manchas) e `<rid>_manga.{kmz,gpkg}`.
