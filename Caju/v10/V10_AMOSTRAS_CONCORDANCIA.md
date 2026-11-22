# v10 — Amostras por concordância (agrupamento + XGBoost + similaridade + revisão) e retreino só com embeddings 2025

Data: 20/11/2026 · Pasta: `D:\Modelo_Caju\v10` · Tudo na GPU (RTX 3050, 4 GB), exceto leitura e gravação de arquivos e as manchas ≥ 1 ha.

## Por quê
A predição v8 (U-Net treinada só em Serra do Mel) teve área total = 0,61 × PAM, com o CE e o RN subestimados (0,41 / 0,48) e o PI e PE superestimados (2,2 / 3,6). Causa principal: **polígonos incompletos** (Serra do Mel: ~18 mil ha de polígonos contra 38 mil ha na PAM), que ensinaram caju como "fundo", e paisagens não vistas no treino. A MapBiomas Lavoura Perene quase não vê o caju no RN, PI e PE (Serra do Mel: 879 ha), mas vê muito no litoral do CE, onde também há coqueiro.

## Fontes (todas na grade de 10 m dos embeddings 2025)
| Fonte | O que é | Onde roda |
|---|---|---|
| S1 U-Net | Probabilidade do mapa v8 | GPU (já existia) |
| **S2 XGBoost** | Árvores por pixel com 128 atributos (64 embeddings + média 3×3), `device="cuda"`, `tree_method="hist"`, 400 árvores, profundidade 8. Treino PU em Serra do Mel: positivos = polígonos (erodidos 1 px); negativos confiáveis = > 100 m dos polígonos **e** (U-Net ≤ 0,10 **ou** classe segura MapBiomas) | **GPU** (xgboost 3.2.0; treino e predição na GPU, entrada direta de tensores CUDA) |
| S3 Similaridade | Maior cosseno com 8 assinaturas (k-means nos pixels dos polígonos); limiar T_sim = menor valor com precisão ≥ 0,90 em Serra do Mel | GPU (PyTorch) |
| S4 Grupos | K-means com 40 grupos **por região** (litoral CE / sertão CE-RN / PI-PE) | GPU (PyTorch) |
| MapBiomas 10 m 2025 | Classes seguras de não-caju (3, 5, 23, 24, 25, 30, 31, 33); Lavoura Perene (36) só como atributo | API da plataforma (usada se o download terminar) |

## Regras
Ver o cabeçalho de `02_rotulos.py`. Em resumo: o grupo é classificado como caju, não-caju ou duvidoso. O pixel vira **caju** se está num polígono de campo, ou num grupo caju com ≥ 2 de 3 fontes concordando e fora das classes seguras da MapBiomas, em mancha ≥ 1 ha. Vira **não-caju** se está numa classe segura da MapBiomas, ou num grupo não-caju com U-Net e XGBoost ≤ 0,1. O resto é **ignorado**, e há uma faixa de 2 px em volta do caju. A área de caju rotulada fica limitada a 1,2 × PAM; acima disso, exige 3/3 fontes. Os grupos duvidosos vão para `data/revisao/grupos_duvidosos.gpkg`; a revisão é opcional e pode ser incorporada depois via `grupos_revisados.csv`.

## Amostras e treino
- Chips: 891 de campo (Serra do Mel, com faixa de exclusão de 25 m) + até 40 por município (20 com caju e 20 sem, ≥ 50% rotulados).
- CV de 5 folds avaliado **só nos rótulos de campo**; os chips de concordância entram só no treino, longe dos blocos de validação e teste.
- Comparação com o **C1** (U-Net da v7) nos mesmos folds e rótulos; limiar global fora do fold; modelo final com todos os chips, épocas fixas e 3 sementes.
- Predição final dos 25 municípios na GPU e comparação v8 × v10 × MapBiomas × PAM.

## Execução
`01_fontes.py` → `run_v10.sh` (02 → 03 → 02c → 11 → 08 → 10b → 14 → 15 → 16). Logs em `logs/`, relatórios em `reports/`.

## Não incluído
- Polígonos de manga (`dados_externos/manga`, 448 polígonos, 2.070 ha): todos em Livramento de Nossa Senhora (BA), **fora dos 25 municípios**. Pela regra do usuário, não entram.

## Resultados (21/11/2026, 01:33)

**CV espacial, 5 folds, limiar global fora do fold (rótulos de campo de Serra do Mel):**

| Modelo | Limiar | IoU agregado | F1 | Precisão | Recall |
|---|---|---|---|---|---|
| C1 – U-Net v7 | 0,50 | 0,649 | 0,787 | 0,776 | 0,799 |
| **v10** | 0,55 | **0,661** | **0,796** | **0,785** | **0,808** |

IoU por fold (C1 → v10): 0,710→0,793 · 0,725→0,753 · 0,602→0,643 · 0,468→0,500 · 0,311→0,310.

**Área 2025 × PAM (25 municípios):** v8 180.876 ha → **v10 237.084 ha** (PAM 298.195 ha). **Correlação com a PAM: 0,66 → 0,86.** Municípios com razão mapa/PAM entre 0,5 e 2,5: **6 → 17 de 25.** Tabela completa em `reports/comparacao_final_v10.md`; mapas em `predictions/2025/`.

**Fracos ainda:** Icapuí (0,03), Fortim (0,18), Cerro Corá (0,01): subestimação no litoral leste do CE; Monsenhor Hipólito (3,64) e Buíque (3,09): superestimação no PI/PE.

**Ajustes durante a execução:** XGBoost na GPU via CuPy; regra de rótulo U-Net **e** XGBoost ≥ 0,70 (a similaridade não discriminava); proteção `__main__` no treino final (Windows).
