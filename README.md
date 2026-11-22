# Mapeamento_Caju_Manga_Coco

Mapeamento de culturas 2025 (10 m) nos municípios produtores da Caatinga, com embeddings AlphaEarth e U-Net ajustada a
partir do modelo de caju. Comparação com PAM/IBGE, Censo Agro 2017 e MapBiomas.

| Pasta | Conteúdo | Repositório de origem (histórico incluído) |
|---|---|---|
| [`Caju/`](Caju/) | scripts v7–v10b, documentação, mapa final dos 25 maiores produtores | [Mapeamento_Caju](https://github.com/rnvuefsppgm/Mapeamento_Caju) |
| [`Manga/`](Manga/) | scripts, tutorial, mapas e vetores dos 25 municípios | [Mapeamento_Manga](https://github.com/rnvuefsppgm/Mapeamento_Manga) |
| [`Coco/`](Coco/) | scripts 00–50, tutorial, mapas e vetores dos 15 maiores produtores da Caatinga | [Mapeamento_coco](https://github.com/rnvuefsppgm/Mapeamento_coco) |
| [`Culturas/`](Culturas/) | mapa único caju + manga + coco sobre MapBiomas Coleção 11, tabela por município × PAM × Censo | novo |

Cada pasta tem seu `README.md` e documentação própria (`docs/`, `TUTORIAL.md`, `DECISOES.md`). Os commits dos três
repositórios de origem foram incorporados com `git subtree` (o histórico de cada um aparece no `git log`).

## Resultado 2025
| Cultura | Municípios | Área mapeada (ha) | PAM 2023–25 (ha) | Censo 2017 (ha) | Código no mapa único |
|---|---|---|---|---|---|
| Caju | 25 | 248.415 | 293.361 | 160.939 | 97 |
| Manga | 25 | 26.395 | 56.473 | 43.015 | 98 |
| Coco | 15 | 23.053 | 54.136 | 33.325 | 99 |

Áreas do mapa único (após resolver 458 ha de sobreposição entre culturas). Detalhe: `Culturas/docs/CLASSES_MAPA_UNICO.md`.

## Dados não incluídos
Embeddings AlphaEarth, caches de treino e pesos dos modelos (dezenas de GB) ficam fora do repositório; cada tutorial
explica como gerá-los.
