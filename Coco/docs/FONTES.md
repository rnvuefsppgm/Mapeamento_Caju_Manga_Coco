# Fontes de dados (consultadas uma vez; não repetir buscas)

| Fonte | Uso | Acesso | Situação |
|---|---|---|---|
| PAM/IBGE tabela 1613, produto 2727 "Coco-da-baía" (var. 2313 área destinada, 216 área colhida, 214 qtd. produzida em mil frutos) | ranking de municípios, prior de área | API SIDRA `apisidra.ibge.gov.br/values/t/1613/n6/all/v/2313,216,214/p/2023,2024,2025/c82/2727` | baixado por `scripts/00_pam_coco.py` |
| Censo Agropecuário 2017, tabela 6955, produto 4945 (var. 10079 área existente, 10078 área colhida, 10080/10082 pés, 9504 estab.) | prior de área, densidade (mediana 167 pés/ha), alocação | API SIDRA `…/t/6955/n6/all/v/…/p/2017/c227/4945` | baixado por `scripts/01_censo_agro_coco.py` |
| Neves et al. 2022, "Análise da comercialização do coco-verde no Ceará 2013–2019", Rev. Econ. Agronegócio 20(1) — repositorio.ufc.br/bitstream/riufc/71734/1/2022_art_rdneves.pdf | Fig. 1 polos produtores do CE; Fig. 2 pontos de produção 2015 (1 ponto = 250 t) | cópia local `dados_externos/artigo_ufc/` (pdf, txt, figuras png) | lido; estudo econômico, sem talhões |
| Municípios IBGE 2024 | limites | `D:\arc_arquivos\BR_Municipios_2024.shp` | local |
| Biomas por município | filtro | `D:\Modelo_Caju\dados_externos\Bioma_Predominante_por_Municipio_2024.csv` | local |
| MapBiomas Coleção 11, 2025 (30 m) | máscara de compatibilidade e negativos seguros | `D:\Eduarda\raster_new\brazil_coverage-col11_2025.tif` | local |
| Descals et al. 2023, mapa global de coqueiro (20 m, ano 2020, U-Net S1+S2) — ESSD 15:3991, doi 10.5194/essd-15-3991-2023; dados doi 10.5281/zenodo.8128183 | semente de positivos (núcleo de coqueirais fechados) | Zenodo (download precisa de autorização do usuário) | pendente |
| Google Satellite Embedding (AlphaEarth) 2025 | entrada do modelo | Earth Engine via DatensEE (`D:\Modelo_Caju\.venv_v8`, projeto EE do usuário) | pendente |
| Mapa de caju v10b (prob.) e polígonos de caju/manga | negativos difíceis (caju ≠ coco) | `D:\Modelo_Caju\v10b\predictions\2025`, `v10b\data\vectors`, `dados_externos\manga` | local |
| Imagens de alta resolução (Google Earth Pro / Esri World Imagery) | fotointerpretação | manual | – |
