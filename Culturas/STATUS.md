# STATUS — Mapeamento Culturas (substituir a cada etapa)

**Etapa atual:** v1 CONCLUÍDA (22/11/2026 07:45). Sem pendências.
**Resultado:** `saidas/mapa/culturas_2025_mapbiomas_c11.tif` (63 MB, paleta embutida) · documento `docs/CLASSES_MAPA_UNICO.md`
· tabela `saidas/tabelas/tabela_final_municipios.csv` · disputas `saidas/auxiliar/disputas_2025.tif`.

| Cultura (código) | Mapa (ha) | PAM 2023–25 (ha) | Censo 2017 (ha) |
|---|---|---|---|
| Caju (97) — 25 mun. | 248.415 | 293.361 | 160.939 |
| Manga (98) — 25 mun. | 26.395 | 56.473 | 43.015 |
| Coco (99) — 15 mun. | 23.053 | 54.136 | 33.325 |

Legendas ArcMap/ArcGIS: `legendas/` (04_legendas.py). Disputas: 458 ha (caju×coco 258; manga×coco 200). Tempo: 02 = 7 min (GPU), 03 = 5 s.
Para refazer: apagar `saidas/tabelas/areas_municipio_classe.csv` e rodar 02 → 03. Decisões: `docs/DECISOES_METODOLOGICAS.md`.
