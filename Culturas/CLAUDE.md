# Mapeamento Culturas — instruções para o Claude (lidas em toda sessão)

Mapa único 2025 (10 m, EPSG:32724) de caju + manga + coco sobre o MapBiomas Coleção 11, a partir dos mapas finais
de cada projeto (caju `D:\Modelo_Caju\v10b`, manga `D:\Backup_2026\Geral_desktop\MAPA_MANGA`, coco `D:\Modelo_Coco\v1`).

## Regras de economia de tokens (OBRIGATÓRIAS)
1. **Comece por `STATUS.md`** (≤ 30 linhas): etapa, próximo passo, arquivos.
2. **Não releia o código de caju/manga/coco**: as entradas são só os mosaicos finais listados em `config/config.yaml`.
3. **Leitura mínima**: `grep -n`, `head`, `sed -n 'a,bp'`; nunca `cat` de arquivo > 150 linhas, log inteiro ou raster.
4. **Scripts imprimem ≤ 20 linhas**; detalhe em `saidas/tabelas/` (csv/md) e `logs/`. Checar com `tail -n 20`.
5. **Processos > 2 min em segundo plano**, sem acompanhar; retomáveis (pulam o que existe).
6. **Comandos independentes numa só chamada.** Sem subagentes; sem repetir buscas web.
7. **Decisões**: uma linha no topo de `DECISOES.md`. Atualize `STATUS.md` ao fim de cada etapa (substituir).
8. **Respostas curtas em português, com tabelas.** Configuração só em `config/config.yaml`.

## Ambiente
Python `D:/Modelo_Caju/.venv/Scripts/python.exe` (torch CUDA, RTX 3050 4 GB, rasterio, geopandas).

## Mapa
`scripts/01_pam_censo.py` (PAM 2023–25 + Censo 2017, 3 culturas) → `scripts/02_mapa_unico.py` (GPU, por blocos) →
`scripts/03_documento.py` (tabela + `docs/CLASSES_MAPA_UNICO.md`). Plano: `docs/PLANO.md`. Decisões detalhadas: `docs/DECISOES_METODOLOGICAS.md` (acrescentar novas aqui e uma linha em `DECISOES.md`).
