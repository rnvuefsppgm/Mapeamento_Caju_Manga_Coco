# A5 — Mapa automático de manga por pixel (XGBoost da A4a) — rótulo AUTOMÁTICO

Modelo: `modelos/a4a/xgb_4classes.json` (AUC 0.9565 | Brier 0.0717 | treino 172610 px). Pixel confiável = regra da opção A; fora das classes seguras e dentro das compatíveis MapBiomas; abertura 1 px; manchas ≥ 1.0 ha. Área dentro do limite municipal (sem buffer).

| regiao                        | conhecida   |   ha_p50 |   ha_p80 |   ha_confiavel |   n_manchas |   pam_2025_ha |   censo_2017_ha |   conf/PAM |   p50/PAM |
|:------------------------------|:------------|---------:|---------:|---------------:|------------:|--------------:|----------------:|-----------:|----------:|
| m2918407_Juazeiro             | False       |  7359.32 |   148.38 |           0    |           0 |         14003 |           12157 |       0    |      0.53 |
| m2611101_Petrolina            | False       |  2505.99 |     8.95 |           0    |           0 |         11450 |            8266 |       0    |      0.22 |
| m2907202_CasaNova             | False       |  2858.32 |   115.14 |           0    |           0 |          6570 |            4277 |       0    |      0.44 |
| m2601607_BelemdoSaoFrancisco  | False       |   213.48 |     0.1  |           0    |           0 |          6400 |             634 |       0    |      0.03 |
| m2919504_LivramentodeNossaSen | True        |  8307.59 |  2512.81 |        2078.59 |         138 |          6630 |            5369 |       0.31 |      1.25 |
| m2612604_SantaMariadaBoaVista | False       |   228.4  |     0.03 |           0    |           0 |          1700 |             806 |       0    |      0.13 |
| m2608750_LagoaGrande          | False       |    21.91 |     0    |           0    |           0 |          1900 |             435 |       0    |      0.01 |
| m2909901_Curaca               | False       |   152.15 |     1.16 |           0    |           0 |           945 |            1408 |       0    |      0.16 |
| m2930204_SentoSe              | False       |  1276.91 |    60.28 |           0    |           0 |           955 |             955 |       0    |      1.34 |
| m3135050_Jaiba                | False       | 23717.9  |  2498.9  |          12.39 |           1 |          1075 |            1363 |       0.01 |     22.06 |
| m2404705_Ipanguacu            | False       |    20.76 |     0    |           0    |           0 |           850 |             792 |       0    |      0.02 |
| m2917201_Ituacu               | True        |   421.06 |     6.76 |           1.04 |           1 |           800 |             442 |       0    |      0.53 |
| m3140852_MatiasCardoso        | False       | 13193.5  |  1315.29 |           0    |           0 |           750 |             766 |       0    |     17.59 |
| m2910107_DomBasilio           | True        |   861.34 |   136.61 |          85.61 |          14 |          1224 |            1011 |       0.07 |      0.7  |
| m2931004_Tanhacu              | True        |   126.65 |     0.67 |           0    |           0 |           600 |             219 |       0    |      0.21 |

Total confiável: 2,178 ha · PAM 2025 dos 15: 55,852 ha · p ≥ 0,5: 61,265 ha.

Rasters: `dados/predicoes/xgb_auto/<regiao>_prob.tif` (0–100) e `_conf.tif`; manchas: `manchas_confiaveis.gpkg`.