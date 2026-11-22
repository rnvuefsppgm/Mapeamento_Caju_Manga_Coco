# STATUS — Mapa de manga 2025 (atualizar ao fim de cada etapa; substituir, não acumular)

**Etapa atual:** mapa v1 CONCLUÍDO (`reports/v1_comparacao.md`; raster `dados/predicoes/manga_2025_25mun_classe.tif`; KMZ `dados/predicoes/v1/vetores/`). Coco retomado às 21:38. Próximo (com OK): v2 com amostras do Vale e do Norte de MG.
**Parada combinada:** antes do treino. Treino também espera o fim do treino do caju v10b (GPU).

## Feito
- Dados (A1–A9 de `docs/PLANO_ETAPAS.md`): 258 polígonos de Livramento (1.683 ha), PAM 2023–25, 25 municípios
  (92,5% da manga da Caatinga), embeddings 2025 dos 25 (38 GB, EPSG:32724), MapBiomas 10 m, 110 chips de campo, cache.
- Regras de tokens: `CLAUDE.md`, `STATUS.md`, `tools/resumo.py`, `docs/REFERENCIAS.md`.
- Prompt de amostras adaptado à manga: `docs/PROMPT_AMOSTRAS_MANGA.md`.

## Próximo
1. **Usuário**: mover candidatos para pastas `MANGA_ok` / `NAO_MANGA` / `DUVIDA`; salvar KMZ em `dados/amostras_usuario/rodada1_conferida.kmz`.
   Se possível, digitalizar 5–10 pomares no Vale (Petrolina/Juazeiro/Casa Nova) e 3–5 em Jaíba (pasta `MANGA`).
2. Claude (só com OK): `A4_importar_conferencia.py` → `r_ouro.gpkg`; rodada 2 com modelo por região + `NAO_MANGA` como negativos.
- Rodada 1: T = 0,977; só 9 manchas ≥ T (Livramento 8, Dom Basílio 1); resto na faixa "explorar". Resumo: `reports/candidatos_resumo_top15_caatinga.md`.
- `A3_candidatos_similaridade.py` é retomável: cache por município em `dados/rotulos/candidatos/_cache_*` (apagar para refazer).

## Comandos
`PY=D:/Modelo_Caju/.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8` · estado: `$PY 00_status.py`
