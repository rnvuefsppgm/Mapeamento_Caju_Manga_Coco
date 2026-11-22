# Mapa COCO — instruções para o Claude (lidas em toda sessão)

Mapeamento de coqueiro (coco-da-baía) 2025, replicando a estratégia final do caju
(`D:\Modelo_Caju`, v8→v10b). Código e documentos aqui; dados pesados em `D:\Modelo_Coco`.

## Regras de economia de tokens (OBRIGATÓRIAS neste projeto)
1. **Comece por `STATUS.md`** (≤ 40 linhas). Ele diz a etapa atual, o próximo passo e os
   arquivos relevantes. Não reexplore o projeto nem o caju se o STATUS responder.
2. **Não releia o código do caju.** `docs/REFERENCIA_CAJU.md` diz qual arquivo faz o quê.
   Ao adaptar: copie o arquivo e edite com `grep -n` + edição pontual; leia só o trecho.
3. **Leitura mínima:** `grep -n`, `head`, `sed -n 'a,bp'`, `wc -l` antes de abrir algo.
   Nunca `cat` de arquivo > 150 linhas, log inteiro, parquet, raster ou GeoJSON.
4. **Resumir dados com `tools/resumo.py <arquivo>`** (raster/vetor/tabela em ≤ 15 linhas).
5. **Scripts imprimem ≤ 20 linhas** de resumo; detalhe vai para `reports/` (md/csv) e
   `logs/`. Para checar execução: `tail -n 20 log` ou `grep -c ERROR log`.
6. **Processos longos** (> 2 min): `run_in_background`, sem polling; checar só quando
   notificado. Scripts retomáveis (pulam saídas existentes).
7. **Agrupe comandos independentes** numa só chamada; prefira Bash com `;`.
8. **Sem subagentes** salvo busca ampla real; **sem repetir buscas web** — fontes ficam
   em `docs/FONTES.md`.
9. **Decisões**: uma linha no topo de `DECISOES.md` (data, decisão, motivo, quem, detalhe).
   Atualize `STATUS.md` ao fim de cada etapa (substitua, não acumule histórico).
10. **Respostas curtas em português**, tabelas no lugar de prosa; não repetir o plano
    inteiro — apontar `docs/PLANO_ETAPAS.md`.
11. Configuração só em `config/config.yaml`; nada de caminho fixo no código.

## Ambiente
- Python: `D:/Modelo_Caju/.venv/Scripts/python.exe` (3.11, GPU RTX 3050 4 GB, torch CUDA,
  xgboost GPU). Exportação de embeddings: `D:/Modelo_Caju/.venv_v8` (DatensEE).
- Atalho: `PY=D:/Modelo_Caju/.venv/Scripts/python.exe`.

## Mapa do projeto
- `STATUS.md` estado atual · `DECISOES.md` registro · `docs/PLANO_ETAPAS.md` plano completo
- `docs/ESTRATEGIA_ROTULOS.md` como obter rótulos sem dados de campo
- `docs/PLANO_NOTURNO.md` execução autônoma E5–E7 · `docs/DESEMPENHO.md` ajustes de velocidade e impacto na qualidade · `docs/REFERENCIAS.md` referências bibliográficas/dados (acrescentar novas aqui) · `docs/FONTES.md` acesso/caminhos
- `config/config.yaml` · `scripts/NN_*.py` (ordem numérica) · `tools/` utilidades

## Repositório (desde 22/11/2026)
GitHub oficial: `github.com/rnvuefsppgm/Mapeamento_Caju_Manga_Coco`, pasta `Coco/` (cópia local `D:/Mapeamento_Caju_Manga_Coco`).
O antigo `Mapeamento_coco` será desativado (não apagar sem ordem do usuário); novas versões vão para o repo geral.
