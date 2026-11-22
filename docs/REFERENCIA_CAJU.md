# Referência ao pipeline do caju (não reler o código; usar este mapa)

Raiz: `D:\Modelo_Caju`. Estratégia final = **v8 (exportação) + v10 (fontes/concordância) + v10b (máscara MapBiomas, treino, mosaico)**.
Entrada do modelo: **Google Satellite Embedding (AlphaEarth) 2025, 64 canais, 10 m**, uint8 = round(x·127)+128 (0 = NoData).

| Etapa coco | Arquivo do caju a adaptar | Linhas | O que faz |
|---|---|---|---|
| E2 exportar embeddings | `v8/01_build_expressions.py` | 117 | expressão EE 2025 + regiões GeoJSON (Polygon puro, buffer 1 km) |
| E2 | `v8/run_v8_export.sh`, `v8/export_one.sh` | – | `datensee export` local, tiles 512 px, EPSG:32724, retomável (`.venv_v8`) |
| E2 | `v8/03_mosaic_vrt.py` | 55 | VRT por região |
| E3 fontes | `v10/01_fontes.py` | 200 | XGBoost PU GPU (64 emb + média 3×3), similaridade cosseno, k-means por região |
| E3 rótulos | `v10/02_rotulos.py` | 175 | grupo → classe; concordância ≥ 2/3; teto 1,2×PAM; anel 2 px ignorado; gpkg de revisão |
| E3 rótulos | `v10a/22_rotulos_rank.py` | – | rótulos por ranking com prior de área PAM |
| E3 máscara | `v10b/30_mapbiomas_recorte.py` | 38 | recorta MapBiomas col. 11 (30 m) para a grade 10 m |
| E3 máscara | `v10b/31_rotulos_mb.py` | 45 | fora das classes compatíveis = negativo |
| E4 chips | `v10b/03_chips.py`, `v10b/02_chips_v9.py` | 61/162 | chips 64 canais campo + pseudo, faixa de exclusão |
| E4 cache | `v10b/02c_build_cache.py` | 45 | cache memmap (uint8) |
| E5 CV | `v10b/08_spatial_cv.py` | 316 | CV espacial leave-one-block-out, 5 folds |
| E5 limiar | `v10b/10b_oof_threshold.py` | 52 | limiar global fora do fold |
| E5 final | `v10b/14_final_train.py` + `src/ft.py` | 58/102 | ajuste fino, 5 sementes, 100 épocas, EMA, Lovász, `GPU_AUG=1` |
| E5 fusão | `v10b/17_fusao.py` | 70 | U-Net × XGB só se ganhar fora do fold |
| E6 predição | `v10b/15_predict_final.py` + `src/pred_a.py` | 52/70 | membros + TTA ×2, pós-proc 1 ha |
| E6 mosaico | `v10b/32_mosaico.py` | 23 | raster único COG |
| E7 comparação | `v10b/16_comparacao.py`, `v8/05_compare_pam.py` | 37/22 | mapa × PAM × MapBiomas |
| Embeddings util | `v10b/src/embeddings.py` | 109 | leitura/alinhamento dos embeddings |

Lições que valem para o coco (de `D:\Modelo_Caju\DECISOES.md`):
- Polígonos incompletos ensinam "alvo = fundo" → fora dos polígonos usar **ignorado**, não negativo, salvo negativo confiável.
- Avaliar o CV **só em rótulos verificados** (manuais), nunca em pseudo-rótulos.
- Cache uint8 + aumento de dados na GPU: época ~25 s (RTX 3050).
- XGBoost: `device="cuda"` + CuPy `inplace_predict`, senão cai na CPU.
- Windows: proteção `if __name__ == "__main__"` no treino com workers.
- MapBiomas: arquivo local `D:/Eduarda/raster_new/brazil_coverage-col11_2025.tif` (API/GEE deram 403/limite).
- Pesos da U-Net do caju (`v10b/models/`) servem de ponto de partida (mesma entrada de 64 canais).
