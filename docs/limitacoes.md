# Limitações e ameaças à validade (rascunho)

> **Status:** rascunho de 17/11/2026, para a seção de limitações do artigo. Os números vêm do README (itens 53 a 68) e mudam se as ablações e os testes pendentes mudarem. Cada limite aponta o item do registro de incertezas (U1 a U21) que o sustenta.
> **Regra de linguagem:** verbos calibrados ("sugere", "não foi detectada diferença"), sem "prova", "é o teto" ou "o modelo imita" como fato. Dúvida declarada vale mais que afirmação sem evidência.

## 1. Escopo da tarefa

O trabalho segmenta, de forma binária, feições de baixa amplitude em imagens acústicas de poço. Essas feições incluem fraturas, cavidades (*vugs*) e *breakouts*, e os modelos não os distinguem (U18). Zonas de artefato, como alargamento do poço ou perda de sinal, também têm amplitude baixa. O trabalho excluiu as pontas de três poços por uma regra própria (D1), mas outros intervalos de amplitude baixa não foram analisados (U3, U4). Por isso, "feição escura" não equivale a "feição geológica".

## 2. Dados

A imagem tem 180 colunas por volta. O valor de 2° por coluna é plausível pela literatura consultada, mas não foi confirmado na documentação do WAID (U1). Valores de tamanho de pixel dependem dessa suposição e do diâmetro do poço (U2). A normalização por percentis de cada poço (D3) e a reamostragem do antilope25 (D2) são escolhas do autor; a primeira será testada na ablação H3, e a segunda não será (U6). Os limites da D1 usam parâmetros provisórios (10 dB, janela de 200 linhas, 500 linhas normais). O limite do fim do antilope37 não tem caliper para conferência (U3).

## 3. Conjunto anotado

As 80 anotações foram feitas por uma pessoa, sem especialista em interpretação de imagens de poço, a partir de critérios extraídos de uma fonte (Fioriti, 2026). A fonte de classificação original dos artefatos (Lofts e Bourke, 1999) não foi consultada (U7). Restam 71 patches usáveis (55 com feição clara e 16 sem feição); 9 patches em dúvida ficaram fora. A exclusão dos casos duvidosos tende a deixar só os casos fáceis e pode inflar as métricas (U11). O tatu22 não tem patch sem feição, e o coala88 tem apenas 2; por isso, a AUC por poço não existe ou pouco informa em dois dos cinco poços (U8). A reanotação às cegas cobriu 10 patches, com a mesma pessoa e mais de 10 horas de intervalo; o kappa de Cohen de 1,00 é um indicador grosso (U10).

## 4. Circularidade da avaliação

O critério de anotação ("zona escura clara") e o escore do limiar (fração de pixels escuros) usam a mesma pista. A concordância alta entre eles (AUC de 0,93 para o Otsu) mostra consistência com esse critério. Ela não mostra que os métodos encontram feições geológicas (U9). As métricas do artigo medem concordância com as anotações, não a verdade geológica.

## 5. Pseudo-rótulos e modelos

O pseudo-rótulo de Otsu marca pixels escuros, inclusive sombras e artefatos (U12). No tatu22, o limiar de Otsu separa dois regimes de amplitude: 56,5% dos patches válidos têm mais de 50% de pixels marcados, e alguns têm 100% em imagem cinza uniforme. No coala88 são 8,9% e no botorosa47, 4,2% (U23). Nesses trechos, o pseudo-rótulo marca um nível de amplitude, e não feições; a causa física do regime não foi investigada. As redes foram treinadas com 10 épocas, taxa de 1e-3 e parada antecipada pelo Dice contra o próprio pseudo-rótulo, sem ajuste de hiperparâmetros. A conclusão de que os modelos ficaram abaixo do Otsu vale para essa configuração; outras configurações não foram testadas (U13). A diferença entre a CNN rasa e a U-Net não foi detectada, mas o teste tem pouco poder: 71 patches, 3 sementes, e o desvio entre 3 sementes é uma estimativa instável (U14, U19). O Dice contra o pseudo-rótulo varia de cerca de 0,41 a 0,86 entre poços, sem causa investigada (U15). A explicação de que a soma de probabilidades acumula erro é uma hipótese não testada (U16). O efeito do pré-treino ImageNet não tem fonte consultada e fica para a ablação H7, de prioridade baixa (U20).

## 6. Teste sintético

O teste insere senoides em trechos reais. Espessura, altura, ruído e fatores de contraste são escolhas do autor, sem validação por especialista. O teste cobre só formas senoidais, ou seja, fraturas; não cobre cavidades nem *breakouts*. A região ignorada no cálculo do IoU usa o limiar de Otsu, o que pode favorecer métodos que usam o mesmo limiar (U17). Na ablação da normalização (H3), o teste comum define as feições e a região ignorada pelo limiar da referência. Com um teste diferente por configuração, a normalização por intervalo ficou acima da referência (+0,129); com o teste comum, ficou abaixo (−0,406). O sinal depende do teste, e a H3 fica inconclusiva (U22, U24). Na H3 por intervalo, o IoU quase não muda com o contraste (0,368, 0,369 e 0,360), o que sugere que a saída responde pouco à feição inserida; a hipótese não foi testada. Uma medida sem limiar (AUC por pixel) exige salvar os mapas de probabilidade e retreinar, e não foi feita. A ablação do limiar do pseudo-rótulo (H11, decisão do autor) é lida pela AUC por patch no conjunto anotado; esse conjunto usa um critério ("zona escura clara") próximo do que o pseudo-rótulo marca (U9), e só um limiar alternativo foi testado, sem fonte de especialista.

## 7. Generalização

Cinco poços de uma mesma empresa e região. A validação deixa um poço de fora, mas não testa outras bacias, outras ferramentas nem outras resoluções.

## 8. Reprodutibilidade e uso de IA

As sementes, as configurações e as tabelas de resultados ficam registradas no repositório. O código e a revisão do texto tiveram apoio de um assistente de IA; as decisões, os experimentos e as conclusões são do autor. Por isso, cada afirmação do artigo é conferida contra as tabelas de resultados antes da entrega.
