# A4a — Classificação automática da rodada 1 (rótulo AUTOMÁTICO, não usar para avaliação)

Treino XGBoost (GPU, 4 classes): 172,610 px — manga 60,000, outros 75,090, caju 36,923, coco 597.
CV espacial (5 folds; blocos de 3 km em Livramento): **AUC manga 0.957**, Brier calibrado 0.072.

## Confiabilidade (p calibrada fora do fold × fração real de manga)

| bin           |     n |   frac_manga |
|:--------------|------:|-------------:|
| (-0.001, 0.2] | 94776 |   0.00695324 |
| (0.2, 0.5]    |  7425 |   0.34936    |
| (0.5, 0.8]    | 26557 |   0.686297   |
| (0.8, 0.9]    | 33701 |   0.862942   |
| (0.9, 1.0]    | 10151 |   0.929859   |

## Regras

- Pixel confiável: p ≥ 0.8 (Chapada) ou p ≥ 0.9 e p_manga − max(p_caju, p_coco) ≥ 0.5 (regiões novas); abertura 1 px; manchas ≥ 25 px.
- MANGA_ok (opção A, por pixel): classe MapBiomas compatível [15, 46, 48, 41, 21], parte confiável ≥ 1.0 ha e ≥ 50% do candidato; **o polígono gravado é só o recorte confiável**; área ≤ 0.6 × PAM 2025.
- NAO_MANGA: p ≤ 0.2, ou controle negativo com p < 0.5. DUVIDA: o resto.

## Resultado por município

| regiao                        |   DUVIDA |   MANGA_ok |   NAO_MANGA |   area_ok_ha |   pam_2025_ha |
|:------------------------------|---------:|-----------:|------------:|-------------:|--------------:|
| m2404705_Ipanguacu            |        0 |          0 |           6 |            0 |           850 |
| m2601607_BelemdoSaoFrancisco  |        1 |          0 |           6 |            0 |          6400 |
| m2608750_LagoaGrande          |        5 |          0 |          11 |            0 |          1900 |
| m2611101_Petrolina            |        6 |          0 |          12 |            0 |         11450 |
| m2612604_SantaMariadaBoaVista |        6 |          0 |          12 |            0 |          1700 |
| m2907202_CasaNova             |       12 |          0 |           6 |            0 |          6570 |
| m2909901_Curaca               |        7 |          0 |          11 |            0 |           945 |
| m2910107_DomBasilio           |        2 |          0 |          11 |            0 |          1224 |
| m2917201_Ituacu               |        1 |          0 |          11 |            0 |           800 |
| m2918407_Juazeiro             |        9 |          0 |           9 |            0 |         14003 |
| m2919504_LivramentodeNossaSen |        9 |          3 |           8 |         1128 |          6630 |
| m2930204_SentoSe              |        4 |          0 |           6 |            0 |           955 |
| m2931004_Tanhacu              |        2 |          0 |          11 |            0 |           600 |
| m3135050_Jaiba                |       13 |          0 |           5 |            0 |          1075 |
| m3140852_MatiasCardoso        |       12 |          0 |           6 |            0 |           750 |

Total: {'NAO_MANGA': 131, 'DUVIDA': 89, 'MANGA_ok': 3}

KMZ: `D:\Backup_2026\Geral_desktop\MAPA_MANGA\dados\amostras_usuario\rodada1_auto.kmz` · checagem: `D:\Backup_2026\Geral_desktop\MAPA_MANGA\dados\amostras_usuario\rodada1_checagem.kml` (23 candidatos sorteados, gabarito em `dados/rotulos/candidatos/rodada1_checagem_gabarito.csv`).