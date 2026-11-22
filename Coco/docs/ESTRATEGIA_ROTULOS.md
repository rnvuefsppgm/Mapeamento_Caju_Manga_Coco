# Estratégia de rótulos para coco (sem dados de campo)

Princípio: **duas camadas de rótulo com papéis diferentes** — a lição central do caju.

| Camada | Origem | Papel | Volume alvo |
|---|---|---|---|
| **R-ouro** (verificada) | fotointerpretação em alta resolução | **avaliação** (CV, limiar, trava) + parte do treino | ≥ 400 polígonos de coco em ≥ 4 regiões + ≥ 800 pontos/polígonos de não-coco (≥ 40% difíceis) |
| **R-auto** (pseudo) | concordância de fontes independentes | **só treino**, longe dos blocos de validação | 30+30 chips por município |

## 0. Ponto de partida: amostras-modelo do usuário (few-shot nos embeddings)
1. Usuário digitaliza polígonos de coqueiral certo no Google Earth Pro (pasta `COCO`).
2. `05_candidatos_similaridade.py`: pixels dos polígonos = modelo; cada pixel dos municípios recebe a
   média dos 5 maiores cossenos (kNN) com o modelo; limiar = p10 da similaridade cruzada entre polígonos.
3. Manchas ≥ 1 ha fora das classes MapBiomas seguras de não-coco → KML por faixa (alta/média/limiar) +
   controle negativo (classe 48 com similaridade baixa, para achar caju/manga/dendê parecidos).
4. Usuário confere no Google Earth → R-ouro v0. Cada rodada adiciona os confirmados ao modelo
   (e os `NAO_COCO` como negativos difíceis), cobrindo paisagens novas (sertão irrigado, Amazônia).
Com só 2 polígonos o modelo é estreito (uma paisagem): esperar alta precisão e baixo recall na 1ª rodada.

## Fontes estatísticas
- PAM 2023–25 (área destinada) e **Censo Agro 2017** (área existente, nº de pés, estabelecimentos):
  prior/teto de área por município e alocação de amostras; divergência PAM × Censo vira alerta.
- **Polos do CE** (Neves et al. 2022, Fig. 1): litoral oeste (principal: Trairi/Paraipaba/Paracuru),
  norte, metropolitano, leste + interior irrigado (Ibiapaba, vales do Acaraú, Curu e Jaguaribe);
  coqueiro em tabuleiros até ~20 km da costa. Usados como estratos para a amostragem no CE.

## 1. Fontes prévias (todas reamostradas para a grade de 10 m dos embeddings)
| Fonte | Sinal | Cuidado |
|---|---|---|
| F1 Mapa global de coqueiro (Descals 2023, 20 m, 2020) | positivo forte para coqueiral **fechado** | ano 2020 (supressão/plantio novo até 2025); subdetecta coqueiral esparso/consorciado |
| F2 MapBiomas col. 11 2025 | classes compatíveis C (início: 48 perene, 21 mosaico, 15 pastagem, 41 temporária — **recalibrar com R-ouro**, como no caju); classes seguras de não-coco (3, 4, 5, 23, 24, 25, 30, 31, 33 …) | coco, caju, manga, dendê caem na mesma classe 48 |
| F3 XGBoost PU nos embeddings (GPU) | treino com sementes F1∩C erodidas + R-ouro | modelo regional (litoral N, litoral NE, São Francisco irrigado, Pará/ES se entrarem) |
| F4 Similaridade cosseno às assinaturas de coco (k-means dos pixels-semente) | atributo | no caju não discriminou sozinha → só atributo |
| F5 Mapa de caju v10b (p ≥ 0,8) + polígonos de caju e manga | **negativos difíceis** | só onde F1 = 0 |
| F6 PAM coco (área destinada, média 2023–25) | teto/prior de área por município | área declarada ≠ área plantada |

## 2. Fotointerpretação guiada (R-ouro) — o que o usuário faz
1. `scripts/11_amostra_fotointerpretacao.py` gera **KML/KMZ** com quadrados de 100 × 100 m (alvo para Google Earth Pro), estratificados:
   - S1 F1∩C (coco provável) · S2 F1 só · S3 C=48 sem F1 · S4 caju v10b alto · S5 alta similaridade sem F1 · S6 aleatório.
   - ~60 quadrados por município × 25 municípios; ordem aleatória; campo `classe` vazio.
2. O usuário marca `classe` ∈ {`coco`, `nao_coco`, `duvida`} e, nos coco, digitaliza o polígono do talhão (copa em estrela, espaçamento regular ~7,5–9 m em triângulo/quadrado; caju = copa arredondada densa e irregular; dendê = copa maior e mais escura, plantio em triângulo ~9 m).
3. `scripts/12_importar_fotointerpretacao.py` lê o KMZ devolvido → `rotulos/fotointerpretacao/r_ouro.gpkg`, valida geometria e data da imagem (anotar ano no campo `ano_img`).
4. Tempo estimado: ~20–30 s por quadrado → 1.500 quadrados ≈ 10–12 h de trabalho, divisível em lotes por município.

Rodada 2 (aprendizado ativo, opcional): após o 1º modelo, sortear quadrados nas áreas de maior incerteza (entropia) e de maior discordância modelo × PAM.

## 3. Rótulos automáticos (R-auto) — receita v10/v10b adaptada
- **coco** = (F1 = 1 **ou** R-ouro coco) **e** F3 ≥ 0,70 **e** pixel em C **e** não caju v10b ≥ 0,5 **e** mancha ≥ 1 ha; área limitada a 1,2 × PAM (acima disso, exigir 3/3 fontes).
- **não-coco** = classe segura MapBiomas **ou** (F3 ≤ 0,10 **e** F1 = 0) **ou** F5.
- **ignorado** = resto + anel de 2 px em volta do coco + não-coco a < 50 m de R-ouro coco.
- Fora dos polígonos R-ouro **não** vira negativo automaticamente (erro do caju v8).

## 4. Controle de qualidade (trava antes do treino)
- 200 pontos aleatórios de R-auto (estratificados coco/não-coco) conferidos visualmente → **precisão do rótulo coco ≥ 0,90** e não-coco ≥ 0,95. Se falhar: ajustar limiares/classes C e repetir.
- Relatório `reports/qc_rotulos.md`: área rotulada × PAM por município; matriz F1 × MapBiomas × R-ouro.

## 5. Transferência do caju
Mesma entrada (64 embeddings): a U-Net do coco parte dos pesos finais do caju v10b (encoder), trocando só a tarefa — acelera e estabiliza com poucos rótulos.
