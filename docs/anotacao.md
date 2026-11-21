# Ficha de critérios para o conjunto anotado (WAID, imagens de amplitude)

> **Status:** definida em 15/11/2026. Versão atualizada em 15/11/2026: decisão por patch (sem máscaras).
> **Fonte principal:** FIORITI, L. S. (2026), tese de doutorado, ref. 6 do README. Páginas citadas entre parênteses.
> **Limite:** a ficha usa só essa fonte. Lofts e Bourke (1999) ainda não foi obtido. Nenhum especialista validou os critérios. O artigo declara isso.

## 1. Como ler a imagem

- Eixo vertical: profundidade. Eixo horizontal: azimute (180 colunas; o valor em graus por coluna ainda é plausível, não confirmado).
- Amplitude baixa aparece **escura** (escala `afmhot`). Amplitude alta aparece clara.
- A imagem é um cilindro aberto. O lado esquerdo e o direito se tocam (circular).
- Plano que corta o poço aparece como **senoide**. A altura da senoide mostra o mergulho. O ponto mais baixo mostra a direção do mergulho (p. 52).

## 2. Decisão por patch

Para cada patch, responda uma pergunta só:

| Resposta | Significa |
|---|---|
| Feição escura clara | Há fratura, cavidade ou *breakout* claros, pelos critérios da seção 3. |
| Sem feição | Não há zona escura clara. |
| Dúvida | Qualquer incerteza. O patch inteiro sai do conjunto. |

Não há máscaras nem classes separadas. Separar fratura, cavidade e *breakout* só entra se os critérios permitirem uma decisão repetível.

## 3. Critérios por feição

| Feição | Como aparece na amplitude | Responda "feição clara" quando | Fonte |
|---|---|---|---|
| **Breakout** | Par de zonas de baixa amplitude, a cerca de 180° entre si. Alongadas na vertical. | Há **os dois lados** do par, com continuidade vertical. A curva CAL, se houver, só apoia: ela mostra alargamento do poço, não o par a 180°. | p. 62, 165; tabela 38 (p. 193) |
| **Fratura natural** | Traço **senoidal** contínuo que cruza a imagem. | O traço fecha uma volta (ou quase) e é contínuo. Fratura vertical sem par pode ser fratura induzida: veja a próxima linha. | p. 52 |
| **Cavidade / vug** | Manchas de baixa amplitude, circulares ou irregulares, isoladas. Sem arranjo regular. | A mancha tem borda definida e não faz parte de uma faixa (veja artefatos). | p. 112, 116 |
| **Fratura induzida** | Traços verticais estreitos, quase retos, em **pares a 180°**. | Não é alvo do projeto. Trate como dúvida se aparecer sozinha ou junto de um *breakout*. | p. 62–63, 165 |

## 4. Artefatos: não marque como feição (ou deixe o patch fora)

| Artefato | Aparência na amplitude | Fonte |
|---|---|---|
| *Washout* | Zona extensa de baixa amplitude, com caliper alargado. | p. 61, 164 |
| Espiralamento | Bandas diagonais repetitivas e contínuas. | p. 62, 164 |
| *Reaming* | Marcas finas e diagonais, com espaçamento decimétrico. | p. 63 |
| Marca de cabo | Linha vertical fina e contínua. | p. 63, 165 |
| Marca de broca | Estrias estreitas de alto contraste, descontínuas. | p. 62, 165 |
| Marca de coluna | Trilha linear de baixa amplitude em **um só** azimute. | p. 62, 165 |
| Marca de ferramenta | Faixas verticais ou levemente inclinadas. | p. 64, 165 |
| Marca de amostra lateral | Círculo escuro bem definido. | p. 64 |
| Perda de sinal | Ver D1 do README (zonas já excluídas). | p. 165 |

**Atenção às faixas diagonais.** As faixas diagonais do coala88 (2340–2370 m) e do tatu22 (2382–2424 m) lembram espiralamento ou *reaming*. É uma hipótese, não uma conclusão. Marque os patches dessas zonas como dúvida até confirmar (item 10 da tabela B).

## 5. Regras de decisão

1. Use a ficha, não a memória.
2. Em caso de dúvida entre feição e artefato, marque dúvida. O patch sai.
3. Marque só o que você vê com clareza. Não complete traços que você imagina.
4. O código registra, para cada patch: poço, linhas, decisão, CAL máximo e `cal_z`. Use o campo de observação para anotar o critério usado, se quiser.
5. Decida uma vez. Não reanote o mesmo patch depois de ver o resultado do modelo.

## 6. Procedimento

1. O código sorteia 40 patches válidos (8 por poço, sem zonas da D1 e sem sobreposição).
2. O código mostra o patch sem marcação e a curva CAL, quando existe.
3. Você responde: "Feição escura clara", "Sem feição" ou "Dúvida".
4. O código salva a decisão em `data/annotations/anotacoes.csv`.

## 7. O que o artigo deve dizer

- As anotações seguem critérios da literatura (Fioriti, 2026), feitas sem validação por especialista.
- As métricas medem a concordância com essas anotações, não a verdade geológica.
- A concordância com os pseudo-rótulos de limiar entra como medida secundária.
- Não há referência humana para a qualidade das bordas. IoU e Dice valem só no teste sintético.
