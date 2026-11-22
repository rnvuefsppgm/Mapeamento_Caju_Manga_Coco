# Mapeamento de manga 2025 — regras do projeto

Réplica da estratégia do caju v10b (`D:\Modelo_Caju\v10b`) para manga. Plano em `docs/PLANO_ETAPAS.md`, decisões em `DECISOES.md`.

## Git (desde 22/11/2026)
- Repositório oficial: https://github.com/rnvuefsppgm/Mapeamento_Caju_Manga_Coco, pasta `Manga/` (cópia local `D:/Mapeamento_Caju_Manga_Coco`). Novas versões vão para lá.
- Publicar: copiar os arquivos alterados do MAPA_MANGA para `D:/Mapeamento_Caju_Manga_Coco/Manga/`, commit e push no repo geral. Não dar push do MAPA_MANGA (o `origin` aponta para o geral só para fetch; push bloqueado porque a estrutura é diferente).
- `rnvuefsppgm/Mapeamento_Manga` (remote `manga_antigo`) fica intacto: não apagar nem arquivar sem pedido do usuário.

## Economia de tokens (obrigatório em toda atividade deste projeto)
1. **Começar por `STATUS.md`** (≤ 40 linhas: etapa atual, próximo passo, arquivos) e `python 00_status.py` (uma linha por etapa). Não reexplorar o projeto se eles responderem.
2. **Nunca listar pastas de dados grandes** (`dados/chips`, `dados/embeddings`, `dados/mapbiomas`, `modelos`). Use contagem: `ls pasta | wc -l` ou o `00_status.py`.
3. **Leitura mínima:** `grep -n`, `head`, `sed -n 'a,bp'`, `wc -l` antes de abrir algo. Nunca `cat` de arquivo > 150 linhas, log inteiro, parquet, raster ou GeoJSON → usar `python tools/resumo.py <arquivo>` (≤ 15 linhas).
4. **Scripts imprimem ≤ 20 linhas** de resumo; o detalhe vai para `logs/` e `reports/*.md|csv`. Checar execução com `tail -n 20 log` ou `grep -c ERROR log`.
5. **Processos longos (> 2 min) em segundo plano** (`run_in_background`), sem polling; checar só quando notificado. Scripts retomáveis (pulam saídas existentes).
6. **Não reler código do caju nem do coco.** Ao portar: copiar o arquivo e adaptar com `grep -n` + `sed`/Edit pontual, lendo só o trecho. Mapas de correspondência nas tabelas abaixo.
7. **Editar com Edit (trecho), não reescrever arquivo inteiro.** Não reler arquivo logo após editar. Agrupar comandos independentes numa chamada.
8. **Parâmetros só em `config/config.yaml`**; nunca caminhos absolutos no código. Exceção: o DatensEE exige caminhos RELATIVOS (o Java lê "D:/" como URL).
9. **Uma linha por decisão** no topo de `DECISOES.md`; **atualizar `STATUS.md` ao fim de cada etapa** (substituir, não acumular).
10. Sem subagentes salvo busca ampla real; sem repetir buscas web (fontes em `docs/REFERENCIAS.md`).
11. Respostas ao usuário: curtas, em português, com tabela; apontar `docs/PLANO_ETAPAS.md` em vez de repetir o plano.

## Ambientes (não criar novos)
- Treino/dados: `D:/Modelo_Caju/.venv/Scripts/python.exe` (Python 3.11, torch CUDA, xgboost, geopandas, rasterio, earthengine-api). Sem openpyxl: xlsx lido por XML.
- Exportação de embeddings: `D:/Modelo_Caju/.venv_v8` (DatensEE, projeto EE `rnv-geodatin`).
- GPU: RTX 3050 4 GB, compartilhada com o caju — não rodar treino enquanto o caju treina.

## Fontes externas
- Limites: `D:/arc_arquivos/BR_Municipios_2024/BR_Municipios_2024.shp`
- Bioma: `D:/Modelo_Caju/dados_externos/Bioma_Predominante_por_Municipio_2024.csv`
- MapBiomas Col. 11 2025 (30 m): `D:/Eduarda/raster_new/brazil_coverage-col11_2025.tif`
- PAM: SIDRA tabela 1613, classificação 82, Manga = 2737 (Coco-da-baía = 2727)

## Mapa caju → manga (scripts)
| Etapa | Caju | Manga |
|---|---|---|
| Amostras / PAM / municípios | v8 `01_build_expressions.regions`, `dados_externos/` | `01_amostras.py`, `02_pam_municipios.py`, `03_regioes.py` |
| Embeddings 2025 | v8 `export_one.sh`, `run_v8_export.sh`, `03_mosaic_vrt.py` | `04_export_embeddings.sh`, `05_vrt.py` |
| MapBiomas recorte / classes | v10b `30_mapbiomas_recorte.py`, `reports/mb_classes_caju.csv` | `06_mapbiomas.py` |
| Rótulos de campo + chips | v9 `02_chips_v9.py`, v10b `03_chips.py`, `02c_build_cache.py` | `07_rotulos_campo.py`, `08_chips.py`, `09_cache.py` |
| Modelo base (U-Net + XGB PU) + CV | v7/v9 `08_spatial_cv.py`, v10 `01_fontes.py` | `10_*` (treino — só após aprovação) |
| Pseudo-rótulos (concordância, ranking PAM, máscara MB) | v10 `02_rotulos.py`, v10a `21/22`, v10b `31` | `20_*` |
| Final / predição / mosaico / comparação | v10b `14,15,17,32,16` | `30_*` |

## Mapa coco → manga (etapa de amostras; `D:\Backup_2026\Geral_desktop\Mapa_COCO\scripts`)
| Coco | Manga | Adaptação |
|---|---|---|
| `00_pam_coco.py` | já feito em `02_pam_municipios.py` | PAM 1613/2737 pronta; não refazer |
| `01_censo_agro_coco.py` | `scripts_amostras/A1_censo_agro_manga.py` | SIDRA 6955, c227 = manga (código a achar nos metadados da 6955) |
| `03_regioes_export.py`, `run_export.sh` | só para município novo | os 25 municípios já foram exportados em EPSG:32724 (38 GB); `regioes_<lista>.txt` = "rid EPSG:32724" |
| `05_candidatos_similaridade.py` | `scripts_amostras/A3_candidatos_similaridade.py` | pasta GE `MANGA`; embeddings em `dados/embeddings/<rid>/*.tif`; modelo extra = `dados/amostras/manga_poligonos.gpkg`; negativos-protótipo coco/caju |
| `12_importar_fotointerpretacao.py` (planejado no coco) | `scripts_amostras/A4_importar_conferencia.py` | pastas `MANGA_ok` / `NAO_MANGA` / `DUVIDA` |
| `tools/resumo.py` | `tools/resumo.py` | cópia |
