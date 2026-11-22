# Plano — mapa único caju + manga + coco 2025 (economia de tokens incluída)

## Escopo
Raster único 10 m EPSG:32724 (fundo = MapBiomas col. 11 2025) dos 58 municípios mapeados (caju 25, manga 25, coco 15; sobreposição de municípios:
caju∩coco = Beberibe, Cascavel, Itapipoca, Itarema; manga∩coco = Petrolina, Casa Nova, Juazeiro; caju∩manga = nenhum).

## Códigos (livres na legenda MapBiomas Coleção 11 — conferido na planilha, 5 abas)
| Valor | Classe |
|---|---|
| 97 | Caju |
| 98 | Manga |
| 99 | Coco |
| 1–91 | Classe MapBiomas Coleção 11 (2025), reprojetada para 10 m (vizinho mais próximo) — fundo |
| 255 | Fora dos municípios mapeados |

## Regras
1. Pixel marcado por mais de uma cultura → vence a maior **margem normalizada** `(p − limiar)/(1 − limiar)` de cada
   modelo (caju 0,30 fusão sertão / 0,40 U-Net; manga 0,55; coco 0,65). Raster auxiliar de disputas para conferência.
2. Cultura não mapeada num município não vira "0 = ausente": camada/tabela de cobertura por município.
3. Área por município × classe + PAM (média 2023–25) e Censo 2017 de cada cultura; Censo do caju a baixar do SIDRA 6955.

## Economia de tokens (vale para esta tarefa)
- Um script só: `scripts/02_mapa_unico.py` (por janelas de 4096 linhas, CPU — gargalo é leitura/compressão, GPU não ajuda);
  caminhos em `config/config.yaml`. Saída ≤ 20 linhas; detalhe em `saidas/` (md/csv).
- Não reler código de caju/manga: entradas = mosaicos finais (`*_FINAL_classe/prob`, `manga_2025_25mun_*`, `coco_2025_*`).
- Execução em segundo plano, sem acompanhar; checar só `tail` do log ao fim.
- Documento final: `docs/CLASSES_MAPA_UNICO.md` (valores, cores, regra de disputa, tabela por município).
