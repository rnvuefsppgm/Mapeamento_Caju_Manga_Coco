# v10a / v10a-ft — Correção automática dos municípios fracos + ajuste fino

Data: 21/11/2026 · Pasta: `D:\Modelo_Caju\v10a` (a v10 fica intacta) · GPU RTX 3050 4 GB · só embeddings 2025.

## Objetivo
Corrigir, sem revisão manual, os municípios fracos da v10: Icapuí (0,03 × PAM), Fortim (0,18), Cerro Corá (0,01),
Monsenhor Hipólito (3,64) e Buíque (3,09).

## Diagnóstico (v10)
- Icapuí e Cerro Corá: **nenhum** pixel rotulado como caju; só chips sem caju → o treino reforçou "não-caju".
- Monsenhor Hipólito: **nenhum** chip (menos de 50% de área rotulada).
- Buíque: só chips sem caju.
- Causa comum: o XGBoost treinado só em Serra do Mel marcava de 40 mil a 106 mil ha nesses municípios, e a regra "U-Net **e** XGBoost" falhou.

## Mudanças — rótulos e amostras (v10a)
| Etapa | Script | O que faz |
|---|---|---|
| E1 | `21_xgb_regional.py` | XGBoost **por região** (litoral CE, sertão CE/RN, PI/PE) na GPU. Treino nos 17 municípios "bons" da v10 (razão v10/PAM entre 0,5 e 2,0): caju = v10 ≥ 0,80; não-caju = v10 ≤ 0,05; 60 mil + 120 mil pixels por município; 30% das amostras das outras regiões entram em cada região. |
| E2 | `22_rotulos_rank.py` | Rótulos por **ranking com prior de área da PAM**: nota = média (U-Net v10, XGB regional). Caju = topo do ranking até 0,6 × PAM, com nota ≥ 35, em manchas ≥ 1 ha. Não-caju = abaixo da posição de 3 × PAM (nota ≤ 60) ou nota ≤ 10. O resto é ignorado, com anel de 2 px. A PAM só define **quantos** rótulos entram; o mapa final não usa limiar por município. |
| E3 | `03_chips.py`, `02c_build_cache.py` | 30 chips com caju + 30 sem por município; mínimo de 30% de área rotulada (era 50%). |

Rótulos de caju nos fracos: Icapuí 0 → 373 ha; Fortim 127 → 2.411 ha; Cerro Corá 3 → 357 ha; Monsenhor Hipólito 1.629 ha (com não-caju local até a nota 60); Buíque 871 ha.

## Mudanças — treino (v10a-ft, pedido do usuário)
| Item | v10 | v10a-ft |
|---|---|---|
| Pesos iniciais | ImageNet | **U-Net v10**. No CV, cada fold parte do **mesmo fold** da v10, que não viu aquele bloco de teste, para evitar vazamento. O modelo final parte dos 3 modelos finais da v10. |
| Épocas | 28 fixas | CV com até 80 épocas e paciência de 20; o modelo final usa a mediana da melhor época do CV |
| Taxa de aprendizado | 1e-4, cosseno | 2 épocas de aquecimento + cosseno (1e-4 → 1e-6); **encoder a 0,1×** nas 5 primeiras épocas |
| Média de pesos | não | **EMA** (0,998), usada na validação e salva como modelo |
| Perda | focal + dice | focal + dice + **0,5 × Lovász** (otimiza IoU) |
| Amostragem | uniforme | chips dos 5 fracos sorteados **3×** mais |
| Trava | – | IoU agregado do CV de campo ≥ 0,65; senão para e consulta o usuário |

Código: `src/ft.py` (ativado por `FT_INIT`), `08_spatial_cv.py`, `14_final_train.py`.

## Mudanças — predição (v10a-ft)
| Melhoria | Detalhe |
|---|---|
| Ensemble de 8 membros | 3 sementes finais + 5 modelos de fold do CV |
| TTA ×2 | original + rotação de 180°. O ×4 foi descartado porque deixaria a predição ~85 min. |
| Fusão com o XGBoost regional | p = a·U-Net + (1−a)·XGB. O peso `a` e o limiar são escolhidos **sem vazamento** no mapa fora do fold de Serra do Mel (`17_fusao.py`). Só é usada se ganhar ≥ 0,005 de IoU sobre a = 1. |
| Pós-processamento | remove manchas < 1 ha e preenche buracos < 1 ha dentro do caju (`src/pred_a.py`) |

Saídas em `predictions/2025`: `_prob.tif` (U-Net), `_final.tif` (após a fusão), `_class.tif` (classe final).

## Execução
`run_v10a.sh` (E1–E3) → pausa → `run_v10a_ft.sh` (08 → 10b → trava → 17 → 14 → 15 → 16). Log: `logs/run_v10a.log`.

## Riscos
- A PAM entrou como prior de quantidade de rótulos, então a comparação final com a PAM é **otimista**. O CV de campo é a referência independente.
- Litoral do CE: coqueiro pode ser confundido com caju.
- Icapuí e Cerro Corá: o piso de nota 35 limitou os rótulos de caju, então a melhora nesses dois deve ser parcial.

## Resultados
(preenchido ao final)
