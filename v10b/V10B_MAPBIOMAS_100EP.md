# v10b — Máscara MapBiomas (Col. 11, 2025, 30 m) + ajuste fino com 100 épocas e 5 sementes

Data: 21/11/2026 · Pasta: `D:\Modelo_Caju\v10b` (v10 e v10a intactas) · Base: rótulos por ranking da v10a (`v10a/V10A_AJUSTE_FINO.md`).

## Por quê
1. Os polígonos de Serra do Mel cobrem ~18 mil ha, contra 38 mil ha na PAM. Tudo o que ficava fora deles era "não-caju", o que ensinava a rede a ignorar caju real e punia na validação o acerto em caju sem polígono.
2. No CV da v10a-ft, a melhor época foi 0 ou 1 em 4 dos 5 folds pelo motivo do item 1. Seguir essa regra deixaria o modelo final quase igual à v10.

## Fonte MapBiomas
`D:\Eduarda\raster_new\brazil_coverage-col11_2025.tif` (Coleção 11, 2025, 30 m, EPSG:4326), fornecido pelo usuário. O GEE e a API não foram usados: os COGs públicos agora dão 403 e a API estava com o limite de exportações esgotado. O raster é recortado e reamostrado (vizinho mais próximo) para a grade de 10 m de cada município (`30_mapbiomas_recorte.py`).

## Onde o caju cai na MapBiomas (`reports/mb_classes_caju.csv`)
| Área | 48 Outras perenes | 41 Outras temporárias | 15 Pastagem | 21 Mosaico de usos | 4 Formação savânica | 3 Formação florestal |
|---|---|---|---|---|---|---|
| Polígonos de campo de Serra do Mel | 19,0% | **66,3%** | 12,3% | 1,6% | 0,6% | – |
| v10 ≥ 0,8 no litoral CE | 51,1% | 8,9% | 9,0% | 27,1% | 1,3% | 2,1% |
| v10 ≥ 0,8 no sertão CE/RN | 37,4% | 32,7% | 21,6% | 5,8% | 1,1% | 0,3% |
| v10 ≥ 0,8 no PI/PE | – | 23,5% | 52,8% | 4,1% | 18,7% | – |

Decisão: a ideia do usuário ("fora de lavoura perene = não-caju") foi **ampliada pelos dados**. Usar só a perene transformaria 81% do caju de campo em não-caju. Classes compatíveis com caju: C = {48, 41, 15, 21}, que contêm **98,6%** dos polígonos de campo.

## Regras de rótulo (`31_rotulos_mb.py`)
- **Fora de C** (formação florestal, formação savânica, água, área urbana, praia etc.) → **não-caju**, exceto nos polígonos de campo.
- **Dentro de C** → mantém o rótulo do ranking da v10a. A lavoura perene pode ser coqueiro ou manga, então só vira caju pelo ranking, nunca só pela classe.
- **Chips de campo de Serra do Mel** (treino e validação): caju = polígonos; não-caju **só onde é confiável** (fora de C ou nota ≤ 10); o resto fica **ignorado** (`02_chips_v9.py`).
- Anel de 2 px em volta do caju → ignorado.

Efeito nos fracos (caju rotulado v10a → v10b): Icapuí 373 → 290 ha; Fortim 2.411 → 2.219; Cerro Corá 357 → 304; Monsenhor Hipólito 1.629 → 1.557; Buíque 871 → 406 (465 ha estavam em formação savânica ou florestal); Pio IX 14.958 → 11.738. Os rótulos de não-caju confiáveis aumentaram 2× a 3,5×.

## Treino (pedido do usuário)
| Item | v10a-ft | v10b |
|---|---|---|
| CV (trava) | até 80 épocas, paciência 20 | até 60 épocas, paciência 15, com os rótulos de campo corrigidos; trava IoU ≥ 0,65 |
| Modelo final | mediana das épocas do CV | **100 épocas fixas** |
| Sementes | 3 | **5** (42, 7, 123, 2024, 99; as duas extras partem das finais 42 e 7 da v10) |
| Aquecimento | 2 épocas | **5 épocas**; encoder a 0,1× nas **10** primeiras |
| Demais | EMA 0,998, + 0,5 Lovász, sorteio 3× dos fracos, cosseno até 1e-6 | igual |

## Predição e entrega
- 5 membros (as 5 sementes finais) com TTA ×2, fusão com o XGB regional só se ganhar no mapa fora do fold de Serra do Mel, e pós-processamento de 1 ha.
- **Raster único** dos 25 municípios (EPSG:32724, 10 m, COG): `predictions/caju_2025_25mun_classe.tif` e `predictions/caju_2025_25mun_prob.tif` (`32_mosaico.py`).
- Comparação v8 × v10 × v10b × MapBiomas × PAM: `reports/comparacao_final_v10b.md`.

## Avisos
- O IoU do CV da v10b **não é comparável** com o de C1, v10 e v10a, porque os rótulos de avaliação mudaram (menos falso positivo artificial).
- A PAM entra como prior de quantidade de rótulos, então a comparação com a PAM é otimista.
- MapBiomas a 30 m reamostrada para 10 m: há erro de borda de até 1 pixel de 30 m.

## Desempenho (ajustes durante a execução)
| Mudança | Época | GPU |
|---|---|---|
| Original (4 workers, cache float16 de 19 GB) | ~87 s | 2% |
| Cache convertido para 8 bits (9,6 GB; conversão exata, diferença máxima 0,0001) | ~60 s | 6% |
| Conversão, normalização e aumento de dados **na GPU** (`GPU_AUG=1`, `src/ft.gpu_prep`; mesmo resultado do caminho na CPU, diferença 2e-7) | **~25 s** | **95%** |

Causa: pouca RAM livre (~11 GB) para o cache de 19 GB e CPU dividida com outros programas. 8 workers pioraram (cada um ocupa ~1,6 GB).

## Execução
`run_v10b.sh` (03 → 02c → 08 → 10b → trava → 17 → 14 → 15 → 32 → 16). Log: `logs/run_v10b.log`.

## Resultados
Concluída em 21/11/2026, 16:12 (predição).

**CV fora do fold** (Serra do Mel, mesmos chips de teste):

| Versão | IoU, rótulos originais | IoU, rótulos corrigidos | Precisão | Recall |
|---|---|---|---|---|
| v10 | 0,661 | 0,773 | 0,890 | 0,855 |
| v10a-ft | 0,675 | 0,793 | 0,896 | 0,873 |
| **v10b** | 0,667 | **0,804** | **0,904** | **0,879** |

Fusão U-Net + XGB (0,6/0,4) no mapa fora do fold de Serra do Mel: IoU de 0,823, contra 0,808 só U-Net.

**Treino final:** perda final entre 0,334 e 0,342 nas 5 sementes.

**Área 2025 × PAM, 25 municípios** (`reports/comparacao_unet_x_fusao.md`):

| Mapa | Total (ha) | Correlação com a PAM | Razão 0,5–2,5 | Erro típico |
|---|---|---|---|---|
| v10 | 237.084 | 0,86 | 17/25 | 0,33 |
| **v10b só U-Net** (limiar 0,40) | 239.337 | **0,95** | **21/25** | 0,24 |
| v10b fusão em tudo | 332.104 | 0,82 | 19/25 | 0,33 |
| v10b opção (c): fusão só no sertão CE/RN | 248.513 | **0,95** | **21/25** | **0,18** |

Por região, erro típico só U-Net × fusão: litoral CE 0,16 × 0,35; sertão CE/RN 0,31 × 0,20; PI/PE 0,44 × 0,84.

Os fracos (v10 → v10b só U-Net):

| Município | Razão v10 | Razão v10b só U-Net | Situação |
|---|---|---|---|
| Monsenhor Hipólito | 3,64 | 1,56 | corrigido |
| Buíque | 3,09 | 0,45 | corrigido; passou a subestimar |
| Fortim | 0,18 | 0,44 | melhorou |
| Cerro Corá | 0,01 | 0,16 | segue fraco |
| Icapuí | 0,03 | 0,08 | segue fraco |

**Mapa final: opção (c), liberada pelo usuário em 21/11/2026** (`34_mapa_final_c.py`):
- fusão U-Net + XGB regional (0,6/0,4, limiar 0,30) no **sertão CE/RN**;
- só U-Net (limiar 0,40) no **litoral CE** e no **PI/PE**.

Total: 248.513 ha (PAM 298.195 ha); correlação com a PAM 0,95; 21 de 25 municípios com razão entre 0,5 e 2,5; erro típico 0,18.

| Arquivo | Conteúdo |
|---|---|
| `predictions/caju_2025_25mun_FINAL_classe.tif` | **Raster único final**: 0 = não-caju, 1 = caju, 255 = fora dos 25 municípios. EPSG:32724, 10 m, COG com pirâmides |
| `predictions/caju_2025_25mun_FINAL_prob.tif` | Probabilidade 0–100: fusão no sertão, U-Net no resto. **Os limiares diferem por região** (30 no sertão, 40 no resto) |
| `predictions/2025/<município>_class_final.tif` | Mapa final por município |
| `reports/area_caju_2025_final.csv` | Área por município, método usado e razão com a PAM |

Os rasters de fusão em tudo (`caju_2025_25mun_classe.tif`) e só U-Net (`..._classe_unet.tif`) ficam guardados para comparação.
