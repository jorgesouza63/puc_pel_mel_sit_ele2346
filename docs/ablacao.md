# Estudo de ablação — trabalho final WAID (ELE2346)

> **Origem:** chat "Perguntas Gerais", 02/11/2026. Esta explicação foi adotada como válida.
> **Uso:** referência para o chat "Desenvolvimento de projeto". Consulte este documento ao planejar os experimentos, ao configurar os treinos no Colab e ao escrever a seção de resultados do artigo.
> **Status:** hipóteses H1 a H10 e prioridades definidas em 15/11/2026. Regra de leitura definida (3 sementes). As configurações ainda não foram executadas. Nenhum resultado existe ainda.
> **Atualização de 02/11/2026:** ver a seção 6, no fim deste documento.

## 1. Conceito

Ablação é um experimento que **remove ou troca uma única parte** do método e mede o efeito no resultado. O objetivo é responder: **"esta parte do método ajuda de verdade?"**

```
Modelo completo ──mede──► IoU = X            (referência)
  ├─ remove parte 1 ──► IoU = X − Δ1
  ├─ remove parte 2 ──► IoU = X − Δ2
  └─ troca parte 3  ──► IoU = X − Δ3
```

Interpretação:

- A métrica **cai muito** sem a parte → a parte é importante.
- A métrica **não muda** → a parte não contribui; o método pode ser simplificado.
- A métrica **sobe** → a parte atrapalha.

**Regra principal:** mude **só uma coisa por vez**. Mantenha fixos os dados, a divisão treino/teste, as épocas, o otimizador, a taxa de aprendizado e a semente aleatória.

## 2. Configurações candidatas

| # | Configuração | Pergunta que responde | Prioridade |
|---|---|---|---|
| 0 | **Modelo completo** (referência) | — | obrigatória |
| 1 | Normalização **global** em vez de **por poço** | A normalização por poço ajuda a generalizar para um poço novo? | alta |
| 2 | Min-max em vez de **corte p1–p99** | O corte dos valores extremos importa? | média |
| 3 | Sem pseudo-rótulos do caliper (BSC) | Os pseudo-rótulos de *breakout* acrescentam informação útil? | alta |
| 4 | Sem *data augmentation* de deslocamento circular no azimute | A invariância à posição precisa ser ensinada ao modelo? | alta |
| 5 | U-Net sem *skip connections* | As conexões de atalho melhoram as bordas da segmentação? | baixa |
| 6 | Encoder sem pré-treino (ImageNet) | O pré-treino ajuda em imagens acústicas? | baixa |

Justificativas das prioridades:

- **Linha 1:** a mediana de amplitude varia de 28,83 dB (tatu22) a 36,95 dB (coala88). A normalização por poço é uma decisão central do pré-processamento.
- **Linha 2:** os extremos são raros (0,001% a 0,014% fora de −10 a 50 dB), mas chegam a 113,9 dB (antilope25). Com min-max, a faixa p1–p99 ocupa só ~13% da escala nesse poço.
- **Linha 3:** o dataset não tem máscaras rotuladas. Os pseudo-rótulos são a principal fonte de supervisão para *breakout*. O BSC cobre só 35,0% a 98,5% da profundidade, conforme o poço.
- **Linha 4:** liga o trabalho ao tema da disciplina (padrões invariantes à posição). O eixo horizontal da imagem é o azimute (0° a 360°) e é circular. **Hipótese de domínio a confirmar com fonte especializada** (perfilagem de imagem) antes de registrar no artigo: um deslocamento circular no azimute não altera a interpretação geológica.

**Recomendação:** execute 3 a 5 ablações. Priorize as linhas 1, 3 e 4.

## 3. Cuidados específicos deste trabalho

1. **Validação por poço (*leave-one-well-out*).** Treine com 4 poços e teste no 5º. Repita para os 5 poços. Assim, a ablação mede a generalização para um poço novo.
2. **Várias sementes.** Repita cada configuração com 3 ou mais sementes. Informe média ± desvio padrão. Com só 5 poços, a variação entre execuções pode ser maior que o efeito da ablação.
3. **Métrica definida antes dos treinos.** Sem ground truth completo, a avaliação depende dos pseudo-rótulos ou de um pequeno conjunto rotulado à mão. Declare essa escolha no artigo como limitação.
4. **Custo no Colab.** Cada configuração custa: 1 treino × 5 poços × 3 sementes = 15 treinos. Com 4 configurações (referência + 3 ablações), são 60 treinos. Estime o tempo de um treino antes de fechar a lista.

## 4. Checklist de implementação

- [ ] Fixar a semente (`random`, `numpy`, `torch`, `torch.cuda`) num único ponto do código.
- [ ] Criar um arquivo de configuração por experimento (ex.: dicionário ou YAML), com um campo por parte ablada.
- [ ] Registrar cada execução: configuração, poço de teste, semente, métricas, tempo.
- [ ] Salvar os resultados num arquivo tabular (CSV) para gerar a tabela do artigo sem cópia manual.
- [ ] Atualizar o README.md quando os experimentos forem executados.

## 5. Modelo de tabela para o artigo

```
Configuração                   IoU (média ± dp)   Dice (média ± dp)   Δ IoU vs. completo
Modelo completo                0,xx ± 0,xx        0,xx ± 0,xx          —
− normalização por poço        0,xx ± 0,xx        0,xx ± 0,xx         −0,xx
− pseudo-rótulos BSC           0,xx ± 0,xx        0,xx ± 0,xx         −0,xx
− augmentation circular        0,xx ± 0,xx        0,xx ± 0,xx         −0,xx
```

Texto típico da seção: uma frase por linha da tabela, com a interpretação do Δ. Exemplo: "A remoção da normalização por poço reduziu o IoU em X pontos no poço de teste, o que indica que a diferença de nível entre poços prejudica a generalização."

## 6. Atualização de 02/11/2026

O que mudou desde a primeira versão:

1. **Decisões de dado já tomadas** (ver o README do projeto). D1: excluir as zonas de artefato nas pontas de 3 poços. D2: reamostrar o antilope25 na vertical por 2. D3: normalização estática por poço. D4: patches de 256 × 180, passo 128.
2. **Linha 1 (normalização).** A referência passa a ser a D3: por poço, com os percentis 1 e 99 calculados sem as zonas da D1. Variantes a comparar: *global* (um só par de percentis para todos os poços) e *por intervalo* (z-score do intervalo, seguido de tanh, como no artigo arXiv 2603.20729).
3. **Linha 3 (pseudo-rótulos do caliper).** O BSC cobre só 35,0% a 98,5% da faixa de profundidade da imagem. Há 1 ponto de BSC a cada ≈ 30 linhas de imagem. A ablação vale só nos intervalos cobertos. Decida antes se o caliper gera pseudo-rótulos. O caliper alargado também ocorre em cavernas geológicas.
4. **Linha 4 (deslocamento circular no azimute).** A hipótese de domínio, que estava a confirmar, agora tem duas fontes. Anjos et al. (2022) estendem as colunas de forma circular e criam um segundo patch deslocado em 128 colunas, e tratam isso como aumento de dados. Fioriti (2026, p. 134–135) usa rotação de 180° e justifica pela continuidade lateral da imagem; ela aplica o aumento só em treino e validação. Nenhuma das duas testa o deslocamento circular como ablação. Cite as duas fontes no artigo.
5. **Linhas novas (propostas).**
   - 7: *sem a exclusão da D1*. Mede o efeito da D1.
   - 8: *sem peso por confiança na perda*. Ideia do artigo arXiv 2603.20729. Lá, o efeito dessa parte isolada foi pequeno.
   - 9: *pseudo-rótulo da imagem bruta*, em vez da imagem limpa por autoencoder. Também vem desse artigo.
6. **Prioridades (definidas em 15/11/2026).** Alta: linhas 1, 4 e 7. Média: linhas 2, 3 (se o caliper gerar rótulos), 8 e 9. Baixa: linhas 5 e 6.
7. **Custo.** São 3.465 patches de 256 × 180 (teste do Claude). O tempo de um treino ainda não foi medido. Meça no início da Fase 3 e feche a lista depois. Cada configuração custa 5 poços × 3 sementes = 15 treinos.
8. **Cuidado com a métrica.** O artigo arXiv 2603.20729 mede as ablações pela concordância com o pseudo-rótulo usado no treino, o que mede consistência interna. Neste trabalho, o conjunto anotado é por patch (40 patches; ver a seção 8). Reporte a concordância com os pseudo-rótulos como medida secundária.

## 7. Hipóteses, fontes e leitura dos resultados (reavaliação de 02/11/2026)

O trabalho não busca ineditismo. Cada conclusão deve vir de um experimento do projeto ou de uma referência. Por isso, cada ablação testa uma hipótese com fonte.

| Linha | Hipótese | Fonte | Leitura do resultado |
|-------|----------|-------|----------------------|
| 1 | H3: a normalização estática por poço supera a global e a por intervalo | Fioriti (2026, p. 105–108); relatório IODP; Silva (2026) | confirma, refuta ou inconclusivo |
| 2 | H10: os percentis 1 e 99 superam o min-max | Anjos et al. (2022); extremos de −82 a 114 dB no WAID | idem |
| 3 | H8: os pseudo-rótulos do caliper acrescentam informação | Fioriti (2026); cobertura de 35% a 98,5% | idem |
| 4 | H4: o deslocamento circular melhora a generalização | Anjos et al. (2022); Fioriti (2026, p. 134–135) | idem |
| 5 | sem fonte consultada | — | baixa prioridade |
| 6 | H7: o pré-treino ImageNet ajuda | sem fonte consultada | idem |
| 7 | H5: excluir as zonas da D1 muda o resultado | Fioriti (2026, p. 121); caliper | idem |
| 8 | H6: o peso por confiança melhora o resultado | Silva (2026): efeito pequeno quando isolado | idem |
| 9 | H9: o pseudo-rótulo da imagem limpa supera o da imagem bruta | Silva (2026): 0,6138 contra 0,7026 | idem |

**Regra de leitura (definida em 15/11/2026, com 3 sementes).** Uma ablação mostra efeito quando a diferença média passa de 1 desvio-padrão entre sementes e tem o mesmo sinal em pelo menos 4 dos 5 poços. Fora disso, escreva "sem efeito detectável" e traga o valor numérico. Mostre sempre o resultado de cada poço.

**Os desfechos "refuta" e "inconclusivo" também servem ao artigo.** Eles são resultados de experimento.

**Ordem de execução (definida em 15/11/2026).** Primeiro as ablações de prioridade alta (linhas 1, 4 e 7; hipóteses H3, H4 e H5). Depois, se o tempo permitir, as de prioridade média, na ordem H10, H8, H6, H9. H7 e a linha 5 têm prioridade baixa. H1 e H2 vêm do experimento principal (baselines contra a U-Net, com *leave-one-well-out*). A lista final fecha depois de medir o tempo de um treino, no início da Fase 3.

## 8. Atualização de 15/11/2026: métricas com o conjunto anotado por patch

O conjunto anotado passou a ser por patch (feição escura clara, sem feição ou dúvida), com 40 patches (8 por poço). Não há máscaras de referência humana.

- **IoU e Dice** valem só no teste sintético (máscaras exatas), planejado para a Fase 3.
- **F1 e AUC por patch** valem no conjunto anotado. O escore do patch é a fração de pixels preditos como feição.
- **Concordância com os pseudo-rótulos** no poço de teste continua como medida secundária.
- **Limite:** com 8 patches por poço, a métrica por poço é grossa. A regra de leitura (seção 7) precisa de uma métrica por poço estável. Proposta (aguarda confirmação): aplicar a regra à concordância com os pseudo-rótulos e ao IoU do teste sintético, e usar F1 e AUC por patch como verificação externa. As tabelas e os exemplos das seções 1 a 5 usam IoU só como ilustração.
