# STATUS — Mapa de coco (atualizar ao fim de cada etapa; substituir, não acumular)

**Etapa atual:** v1 CONCLUÍDA (22/11 02:33): mapa `v1/predictions/coco_2025_top15_caatinga_{classe,prob}.tif`; CV IoU 0,789 (aprovada). **Próximo:** decidir rodada v2 de amostras no litoral CE (subestimado, razão mapa/PAM ≤ 0,10).
**Relatórios:** `docs/RELATORIO_FINAL_V1.md` (final), `docs/RELATORIO_PRE_TREINO.md`, `v1/reports/comparacao_final.md`.

## Feito
- E3c–E4 (21/11 18:30): fontes 09, conferência visual 10, r_ouro 12, XGB 13 (v1b), regras 14, QC cego 15 (r2 aprovado: coco 0,93 / não 1,00), chips+cache 20 (1.608 chips, 6,8 GB).
- E0 regras de tokens, docs (`docs/PLANO_ETAPAS.md`, `ESTRATEGIA_ROTULOS.md`, `REFERENCIAS.md`, `FONTES.md`), config.
- E1 PAM 2023–25 + Censo 2017 (`dados_externos/{pam,censo_agro}`), `pam/coco_nordeste_caatinga.csv` (494 munic.).
- E2 embeddings 2025: listas `amostragem_inicial`, `top15_caatinga`, `sao_francisco` (23 GB, `v1/data/embeddings/2025/`).
- E3a modelo litoral = 2 polígonos do usuário (`amostras_usuario/COCO.kmz`).
- E3b litoral: `candidatos_coco_top15_caatinga.kml` (236). Achados só no litoral CE/RN.
- E3b São Francisco SEM amostras do usuário: 06 grupos (K=12) → 07 folhas Esri z17/z18 → grupo 11 = coqueiro
  (copa em estrela, linhas) → 08 modelo (48 polígonos) → `candidatos_coco_sao_francisco_modelo_sao_francisco_g11.kml` (134).
  ≥ T: Rodelas 6.582 ha (PAM 2.357 → superestima ~2,8×), Petrolândia 4.679 (PAM 3.717), Petrolina 971, Casa Nova 523, Juazeiro 31.
  QC visual (6 painéis): 3 coqueiral provável, 1 caatinga (falso), 1 pomar misto, 1 duplicata (corrigido no 05).

## Próximo
- **[TAREFA DO USUÁRIO, 21/11/2026] Ao terminar o mapeamento do coco: subir os scripts e TODOS os resultados no git https://github.com/rnvuefsppgm/Mapeamento_coco** (repo já existe, `main` com commit inicial: fazer o commit em cima dele). Seguir o modelo do Mapeamento_Manga: README + TUTORIAL.md + `resultados/` (mapas de classe, mapas por município, vetores KMZ/GPKG, tabela × PAM/Censo); dados brutos, modelos e rasters grandes (> 50 MB) fora do git; atenção a caminhos locais no config.
1. Usuário decide: iniciar treino (E5) **ou** antes fazer rodada 2 de amostras no litoral (~1 h).
2. E5: adaptar `D:/Modelo_Caju/v10b/08_spatial_cv.py`, `10b_oof_threshold.py`, `14_final_train.py` + `src/ft.py` ao cache
   `v1/data/cache` (uint8; labels.npy treino; labels_ouro.npy avaliação; avaliação = ouro + não-coco automático).

## Comandos
`PY=D:/Modelo_Caju/.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8; cd scripts`
Nova lista: `$PY 03_regioes_export.py <lista>` → `bash run_export.sh <lista>` → `$PY 05_candidatos_similaridade.py <lista> [modelo.kml]`
Região sem modelo: `$PY 06_grupos_perenes.py <lista> 12` → `$PY 07_folhas_visuais.py <lista> [z] [ids]` → `$PY 08_modelo_de_grupo.py <lista> <grupos>`
