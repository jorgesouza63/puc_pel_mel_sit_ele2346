# Segmentação de Imagens Acústicas de Poço sem *Ground Truth* (WAID)

> Trabalho final da disciplina **ELE2346 – Deep Learning** · Mestrado em Engenharia Elétrica (Sistemas Inteligentes) · PUC-Rio · 2026.2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jorgesouza63/puc_pel_mel_sit_ele2346/blob/main/notebooks/00_setup_colab.ipynb)
![Licença](https://img.shields.io/badge/licen%C3%A7a-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-artigo%20finalizado-green)

## Resumo

Perfis acústicos de imagem de poço têm altíssima resolução e exigem interpretação manual demorada por petrofísicos. Este projeto aplica **Redes Neurais Convolucionais 2D** (arquiteturas do tipo U-Net) para segmentar **estruturas geológicas** (fraturas/geodos) e **artefatos** (*breakouts*) nas imagens do dataset público [WAID – Wellbore Acoustic Image Database](https://github.com/petrobras/waid), da PETROBRAS.

O trabalho não busca ineditismo. Ele avalia métodos conhecidos em dados públicos e baseia as conclusões em experimentos e referências.

## O problema

O WAID **não possui máscaras rotuladas** (*ground truth*). O desafio central é, portanto: **como segmentar sem rótulos supervisionados?**

## Abordagem

Supervisão fraca combinada com validação independente:

1. **Pseudo-rótulos** a partir de métodos clássicos (limiar de amplitude do notebook oficial, Otsu, k-means, morfologia) e do **sinal do caliper** para breakouts.
2. **Pré-treino auto-supervisionado** (autoencoder com remoção de ruído) em todas as imagens.
3. **U-Net** treinada com perdas robustas a ruído de rótulo (Dice + BCE).
4. **Conjunto pequeno anotado à mão** (apenas validação/teste) para medir desempenho real.
5. **Validação *leave-one-well-out*** (treina em 4 poços, testa em 1).
6. Tratamento do **azimute circular** (padding circular e deslocamento circular como *augmentation*).

## Estrutura do repositório

```
puc_pel_mel_sit_ele2346/
├── configs/          # YAML de configuração por experimento
├── src/
│   ├── config.py     # poços, pastas e constantes compartilhados pelos notebooks
│   ├── data/         # carga, normalização, patches
│   ├── labels/       # anotação assistida (anotacao.py) e pseudo-rótulos
│   ├── eval/         # avaliação por patch (AUC, F1, concordância)
│   └── models/       # U-Net e encoders
├── notebooks/        # notebooks numerados na ordem de criação e execução (01_exploracao, 02_...)
├── results/          # tabelas e figuras
├── paper/            # artigo acadêmico
├── docs/             # ablacao.md, anotacao.md, limitacoes.md e figuras para o README (PDFs ficam só no computador local)
└── scripts/          # utilitários (ex.: download dos dados)
```

## Como executar (Google Colab)

1. Clique no badge **Open in Colab** acima (abre `notebooks/00_setup_colab.ipynb`).
2. Em **Secrets** do Colab (ícone de chave), crie `GITHUB_TOKEN` (somente necessário para `git push`).
3. Execute a célula de setup: ela monta o Google Drive, clona/atualiza este repositório, instala as dependências e verifica a GPU.
4. Os dados e checkpoints ficam no Drive (`MyDrive/waid/`), fora do Git.

### Execução local

```bash
git clone https://github.com/jorgesouza63/puc_pel_mel_sit_ele2346.git
cd puc_pel_mel_sit_ele2346
python -m venv .venv && source .venv/bin/activate
# instale PyTorch conforme https://pytorch.org
pip install -r requirements.txt
```

## Dados

Os dados **não** são versionados aqui. Fonte oficial: <https://github.com/petrobras/waid> (commit usado: `f8926d0`, de 18/06/2026).

- **5 poços:** antilope25, antilope37, botorosa47, coala88 e tatu22.
- **Imagens (AMP):** 62 subarquivos CSV, 1,21 GB no total. Só o subarquivo `00` de cada poço tem cabeçalho.
- **Perfis básicos (BSC):** 5 arquivos CSV, 0,86 MB no total.
- **Formato:** separador `;` e decimal `,`. Linhas = profundidade. Colunas = `IMG_AMP[0]` a `IMG_AMP[179]` (180 colunas; a tese de Fioriti, ref. 6, descreve imagens acústicas com 180 amostras azimutais por volta completa de 360°, o que torna plausível 2° por coluna; falta confirmar na documentação do WAID). Valores = amplitude em dB.

Download para o Google Drive (no Colab, com o Drive montado):

```
!python /content/drive/MyDrive/waid/scripts/download_waid.py --dest /content/drive/MyDrive/waid/data/raw
```

O script está em `scripts/download_waid.py`. Ele pula arquivos já baixados e confere o tamanho de cada arquivo.

### Carga das imagens (função oficial `concat_IMG_data`)

A PETROBRAS fornece a função `concat_IMG_data(well_id, data_path)` para juntar os subarquivos de cada poço em um único `DataFrame`. O projeto usa essa função.

- **Cópia literal:** `src/data/waid_concat.py` (origem: README do WAID, commit `f8926d0`, licença Apache 2.0). Nenhuma alteração no código da função.
- **Invólucro do projeto:** `src/data/loader.py`, função `load_well_image`. Ela chama a função oficial e depois ordena e verifica os dados.

```python
import sys
sys.path.insert(0, "/content/drive/MyDrive/waid")   # pasta que CONTÉM "src"; deve vir ANTES do import

from src.data.loader import load_well_image

# data_path deve apontar para a pasta "img" (onde estão os arquivos *_AMP*.csv)
df = load_well_image("coala88", "/content/drive/MyDrive/waid/data/raw/img")
# índice = profundidade (m); colunas = IMG_AMP[0] ... IMG_AMP[179]; valores em dB
```

No Colab, monte o Drive antes (`drive.mount("/content/drive")`). A pasta `src` precisa existir em `MyDrive/waid/src/`.

**Por que o invólucro.** A função oficial lê os subarquivos na ordem de `os.listdir()`. Essa ordem depende do sistema operacional. No teste com o poço coala88, a profundidade saiu fora de ordem, com passos de −297 m a +396 m entre linhas vizinhas. O conteúdo estava correto: depois de `sort_index()`, o resultado foi idêntico nas duas ordens de leitura testadas (a natural do sistema e a invertida). O invólucro faz essa ordenação (estável) e conta saltos de profundidade. Ele também avisa quando o rótulo de profundidade se repete (ver abaixo). Além disso, converte os valores para `float32` (a função oficial devolve `float64`).

**Remoção de linhas.** A função oficial aplica `dropna()`, que remove qualquer linha com valor ausente (`-9999` ou vazio). No poço coala88 nenhuma linha foi removida (87.796 linhas lidas, 87.796 mantidas). Nos outros poços, 2 linhas foram removidas no antilope37 e nenhuma nos demais. O Colab manteve 89.982 linhas no antilope37; a contagem de linhas brutas (89.984) foi feita em teste fora do Colab.

**Resultado do teste (coala88):** 87.796 linhas × 180 colunas, profundidade de 2212,00 a 2658,00 m, passo médio de 0,00508 m, 0 saltos, 0 valores ausentes.

**Confirmação no Colab (coala88):** mesmo resultado (87.796 linhas × 180 colunas, 2212,00 a 2658,00 m, 0 saltos). Neste ambiente a função oficial leu os subarquivos já na ordem correta (`reordenado: não`). A ordem pode mudar de um ambiente para outro, por isso o invólucro continua obrigatório.

**Rótulos de profundidade repetidos (antilope25 e botorosa47).** Nesses dois poços o rótulo de profundidade tem só 2 casas decimais e o passo real de amostragem é menor que 0,01 m. Por isso o mesmo rótulo se repete em várias linhas vizinhas (cerca de 4 vezes no antilope25 e 2 vezes no botorosa47). As linhas são amostras diferentes, não cópias. A primeira versão do carregador tratava rótulo repetido como erro e parou no antilope25 (item 10 do registro). A versão atual só emite um aviso. Para esses poços, use a **posição da linha** como eixo de profundidade, não o rótulo.

A ordenação do carregador é estável. Se o sistema listar os subarquivos fora de ordem, até 55 linhas do antilope25 (de 219.635) e 6 do botorosa47 (de 81.062) podem trocar de posição nas fronteiras entre subarquivos. Os rótulos não mudam. No Colab, os 5 poços foram lidos em ordem crescente (`reordenado: não`).

**Resultado nos 5 poços.** O teste fora do Colab e a execução no Colab deram os mesmos números. A coluna `dropna` vem da contagem de linhas brutas feita no teste fora do Colab:

| Poço | Linhas | Profundidade (m) | Passo médio (m) | Linhas por rótulo | Linhas removidas (`dropna`) |
|------|-------:|------------------|----------------:|------------------:|----------------------------:|
| antilope25 | 219.635 | 2061,13 a 2619,00 | 0,00254 | 3,94 | 0 |
| antilope37 | 89.982 | 2114,00 a 2571,11 | 0,00508 | 1,00 | 2 |
| botorosa47 | 81.062 | 2178,00 a 2589,79 | 0,00508 | 1,97 | 0 |
| coala88 | 87.796 | 2212,00 a 2658,00 | 0,00508 | 1,00 | 0 |
| tatu22 | 81.983 | 2174,20 a 2590,67 | 0,00508 | 1,00 | 0 |

Nenhum poço tem valores ausentes depois da carga. No antilope25, 6.092 linhas (2,8%) são cópias exatas de outra linha (mesmo rótulo e 180 valores iguais). Esse ponto fica para a exploração.

### Conversão para PNG (inspeção visual)

O módulo `src/data/png_export.py` (função `salvar_png_poco`) salva PNGs para inspeção visual. Cada poço gera 13 arquivos em `data/processed/png/<poço>/`:

- `visao_geral.png`: o poço inteiro em 4 faixas lado a lado (média em blocos de linhas).
- `trecho_01` a `trecho_12`: 12 trechos de 1.000 linhas em resolução total, espaçados de forma uniforme.

A escala de cor é por poço (percentis 1 e 99). Amplitude baixa aparece escura. O eixo horizontal mostra o índice da coluna (0 a 179), não graus, porque o mapeamento de azimute ainda não foi confirmado.

```python
from src.data.png_export import salvar_png_poco

PNG_DIR = "/content/drive/MyDrive/waid/data/processed/png"
for p in pocos:
    df = load_well_image(p, img)
    arquivos = salvar_png_poco(df, p, PNG_DIR)
    print(p, len(arquivos), "PNGs")
    del df
```

Status: executado no Colab em 01/11/2026 (13 PNGs por poço, 65 no total).

### Caracterização do dataset

Resultados das células A, B e C do `01_exploracao.ipynb`, executadas no Colab em 01/11/2026.

**Perfis básicos (BSC).** Cada poço tem 12 ou 13 curvas, com um ponto a cada 0,152 m. A imagem tem um ponto a cada 0,00508 m (0,00254 m no antilope25). Há cerca de 30 linhas de imagem por ponto de BSC (cerca de 60 no antilope25).

| Poço | Linhas BSC | Profundidade BSC (m) | Curvas | Cobertura da faixa da imagem (%) |
|------|-----------:|----------------------|-------:|---------------------------------:|
| antilope25 | 1.422 | 2060,56 a 2277,12 | 13 | 38,7 |
| antilope37 | 1.651 | 2103,99 a 2355,45 | 13 | 52,8 |
| botorosa47 | 2.756 | 2163,96 a 2583,82 | 13 | 98,5 |
| coala88 | 1.052 | 2208,00 a 2368,17 | 12 | 35,0 |
| tatu22 | 1.112 | 2159,91 a 2329,23 | 13 | 37,2 |

**Valores ausentes nas curvas (arredondados para 0,1%).** O botorosa47 tem 2,9% de ausentes em DTC e DTS. O antilope37 tem 4,0% em RES10 e RES90. O coala88 tem 0,1% em CAL, DEN, DTC, NEU, PE, RES10 e RES90, e não tem a curva DTS. As demais curvas e poços aparecem com 0,0%.

**Amplitude da imagem (dB).** Os percentis vêm de todas as linhas. Os valores com asterisco vêm de 1 linha a cada 10, então os extremos reais podem ser maiores.

| Poço | p1 | Mediana | p99 | Mín.* | Máx.* | Fora de −10 a 50 dB (%)* |
|------|---:|--------:|----:|------:|------:|-------------------------:|
| antilope25 | 16,95 | 36,58 | 39,55 | −55,9 | 113,9 | 0,011 |
| antilope37 | 9,21 | 29,47 | 33,40 | −70,2 | 73,2 | 0,014 |
| botorosa47 | 3,86 | 35,48 | 37,86 | −82,0 | 63,3 | 0,012 |
| coala88 | 24,10 | 36,95 | 39,79 | −31,3 | 51,6 | 0,001 |
| tatu22 | 18,03 | 28,83 | 32,46 | −17,3 | 33,2 | 0,010 |

Pontos que afetam as próximas fases:

- **O BSC cobre só parte das imagens** (35,0% a 98,5% da faixa de profundidade). O sinal do caliper para pseudo-rótulos de *breakout* existe apenas nessa parte.
- **A mediana de amplitude varia de 28,8 a 37,0 dB entre os poços.** Isso exige normalização por poço.
- **Valores extremos são raros** (0,001% a 0,014% das amostras fora de −10 a 50 dB). A normalização por percentis (1 e 99) trata esses valores.

### Inspeção visual (visão geral dos 5 poços)

Análise dos arquivos `visao_geral.png` (4 painéis por poço), feita em 01/11/2026 com três critérios: (1) manchas escuras isoladas e bandas horizontais escuras (amplitude baixa); (2) faixas diagonais periódicas; (3) trechos claros, quase brancos, sem estrutura. As profundidades são **estimativas visuais** pelas escalas dos painéis.

| Poço | Padrão geral | Trechos mais escuros (m, estimados) | Faixas diagonais periódicas |
|------|--------------|-------------------------------------|-----------------------------|
| antilope25 | Predominam trechos claros sem estrutura. Traços horizontais escuros finos e descontínuos. | 2061–2065; 2145–2165; ~2290; 2590–2600; 2610–2619 | Não |
| antilope37 | Trechos claros entre bandas horizontais finas e descontínuas. Zona escura e densa no fim do 4º painel. | 2114–2124; 2176–2185; 2260–2280; 2315–2330; 2515–2532; 2548–2571 | Não evidentes |
| botorosa47 | Predominam trechos claros. Manchas isoladas e bandas horizontais finas. | 2180–2187 (faixa intensa); 2348–2355; 2428–2431 | Não |
| coala88 | Manchas e bandas horizontais. Trechos claros extensos no 4º painel. | 2239–2259; 2415 a ~2524; 2645–2658 | **Sim**: ~2340–2370 (2º painel); padrões inclinados no 3º painel |
| tatu22 | Trechos claros dominam o 1º painel. Estruturas tênues e esparsas nos demais. O 4º painel tem bandas e manchas densas em toda a faixa. | Bandas marcadas até ~2372; trechos densos horizontais até a base (início citado: 2338 m, a confirmar, porque o 3º painel começa perto de 2382 m) | **Sim**: padrões inclinados no 3º painel (~2382–2424) |

As faixas diagonais periódicas aparecem só no coala88 e no tatu22. A tese (ref. 6, p. 62 e Tabela 38, p. 192) lista candidatos para faixas diagonais em imagens de amplitude: espiralamento do poço (bandas diagonais repetitivas, com modulação periódica em tempo de trânsito e raio), marcas de repasse da broca (*reaming*, sulcos oblíquos finos, com espaçamento decimétrico) e marcas de cabo (segmentos diagonais empilhados). O WAID não traz tempo de trânsito nem raio, então não dá para distinguir esses casos. No coala88, o caliper no trecho com faixas (2340–2368 m) é quase constante (≈ 12,27; desvio 0,016), sem alargamento. Uma estimativa do período das faixas por autocorrelação não foi conclusiva. A causa continua em aberto.

**Conferência numérica (fora do Colab).** Para cada trecho citado, calculou-se a fração de pixels de amplitude baixa (abaixo do percentil 5 do poço; a média do poço é 5,0% por construção).

- *Acima da média:* antilope25 em 2061–2065 m (91%), ~2290 m (23%), 2590–2600 m (27%) e 2610–2619 m (26%). Antilope37 em 2114–2124 m (20%), 2176–2185 m (17%), 2515–2532 m (14%) e 2548–2571 m (40%). Botorosa47 em 2180–2187 m (85%), 2348–2355 m (20%) e 2428–2431 m (18%). Coala88 em 2239–2259 m (11%), 2415–2524 m (12%) e 2645–2658 m (8%). Tatu22 em 2338–2372 m (12%) e 2382–2424 m (11%).
- *Sem apoio nessa métrica:* antilope25 em 2145–2165 m (5,0%) e antilope37 em 2260–2280 m (3,0%) e 2315–2330 m (6,7%). A métrica mede só pixels muito escuros, então traços finos ou de contraste moderado podem não aparecer.
- *Faixas diagonais do coala88 (2340–2370 m):* só 1,8% de pixels baixos. O padrão tem contraste moderado e não é escuro.

**Pontas dos poços com amplitude muito baixa.** No topo do antilope25 (2061–2066 m), 82% dos pixels são baixos, e esse trecho reúne 86% dos pixels abaixo de 0 dB do poço. No topo do botorosa47 (2178–2188 m), 98% dos pixels abaixo de 0 dB. No fim do antilope37 (2554–2574 m), 63% a 64% dos pixels são baixos. Esses trechos ficam na região do pico secundário do histograma, perto de −4 dB. Foram comparados trechos em resolução total (escala do poço e escala do próprio trecho) com o perfil de amplitude média por linha. Há dois comportamentos:

- *Setor sem retorno, com limite abrupto (topo do antilope25 e do botorosa47).* No `trecho_01` do antilope25 (2061,1–2063,7 m), 55% dos pixels ficam abaixo de 0 dB (mediana −3,0 dB, contra cerca de 37 dB no resto do poço). A região escura é um setor largo do azimute e muda de posição com a profundidade. A amplitude média por linha sobe de cerca de 8 dB para cerca de 37 dB perto da linha 1700 (≈ 2065,4 m). No botorosa47, as primeiras ~1850 linhas (até ≈ 2187 m) se comportam de forma parecida. As primeiras ~350 linhas do botorosa47 são claras e quase uniformes (cerca de 38 dB).
- *Queda gradual com faixas verticais (fim do antilope37).* Nas últimas 6.000 linhas, a amplitude média cai de cerca de 27 dB para cerca de 18 dB. Aparecem faixas verticais escuras em colunas fixas (aproximadamente 50–60, 95–100 e 155–165). Poucos pixels ficam abaixo de 0 dB (0,6% no último trecho).

**Conferência com o caliper (fora do Colab).** O BSC cobre os dois topos. O caliper (CAL; unidade não confirmada, provavelmente polegadas) fica em cerca de 12,3 a 12,5 no resto dos poços.

- *Antilope25:* o CAL está em ≈ 13,3 no início do BSC (2060,6 m), chega a ≈ 16,9 em ≈ 2062,4 m e volta a ≈ 12,6 em ≈ 2065,4 m. Esse é o mesmo intervalo em que a amplitude média da imagem fica em cerca de 8 dB.
- *Botorosa47:* o CAL sobe a ≈ 13,3–13,8 entre ≈ 2180,0 e ≈ 2181,0 m, salta para ≈ 17,0 em ≈ 2181,3 m, chega a ≈ 18,5 e volta a ≈ 12,8 em ≈ 2187,4 m. O início da subida (linha ≈ 390) e o salto (linha ≈ 640) coincidem com mudanças vistas na imagem. A amplitude fica baixa até ≈ 2187,4 m.
- *Antilope37:* o BSC termina em 2355 m e não cobre o fim escuro desse poço.

A figura `caliper_x_amplitude_topos.png` mostra os dois casos. A hipótese principal para os dois topos é poço alargado (arrombamento, *washout*) com perda de sinal. A tese (ref. 6, p. 60, 66 e 192) associa o arrombamento a amplitude baixa, tempo de trânsito e raio maiores, e usa o caliper para identificar zonas alargadas. O WAID não traz tempo de trânsito nem raio, então a hipótese não pode ser confirmada por completo. O caliper alargado também ocorre em cavernas geológicas (tese, p. 114), então ele sozinho não separa arrombamento de caverna. A decisão D1 não depende da causa. Para o fim do antilope37, a causa continua desconhecida.

### Decisões de projeto

**D1 (01/11/2026). Excluir do treino e da avaliação os trechos de amplitude anômala nas pontas dos poços:** topo do antilope25, topo do botorosa47 e fim do antilope37.

- *Justificativa:* esses trechos não são o alvo do projeto (fraturas, geodos e *breakouts*), e um limiar de amplitude os marcaria como feição.
- *Limites:* os limites exatos serão definidos por uma regra documentada no notebook, a partir do perfil de amplitude média por linha. Limites provisórios, conferidos com o caliper: antilope25 até ≈ 2065,4 m (linha ≈ 1700); botorosa47 até ≈ 2187,5 m (linha ≈ 1850); antilope37 a definir (o BSC não cobre o fim do poço).
- *Fontes:* a seção "Referências especializadas" reúne o que foi consultado. A tese de Fioriti (ref. 6) classifica perda de sinal, atenuação da onda acústica, descentralização e arrombamento como **artefatos**: feições não geológicas que atrapalham a interpretação (classificação de Lofts e Bourke, 1999; p. 60–66 da tese). Na limpeza dos dados, a mesma tese removeu os rótulos em que artefatos impedem observar a rocha e removeu dados faltantes no topo e na base das imagens (p. 121). Nenhuma fonte fixa uma regra de exclusão para o WAID. A regra será definida neste projeto.
- *Evidência própria:* nos dois topos, a queda de amplitude coincide com o alargamento do poço medido pelo caliper (figura `caliper_x_amplitude_topos.png`).
- O perfil de amplitude média por linha dos 5 poços mostra outros intervalos de amplitude baixa que **não** fazem parte desta decisão (por exemplo, perto de 2290 m no antilope25 e de 2645 m no coala88). Eles precisam de análise própria.

**Regra provisória para a máscara da D1 (regra provisória; executada no Colab em 02/11/2026, com os mesmos resultados do teste fora do Colab).** A regra usa só o perfil de amplitude média por linha:

1. Suaviza o perfil com média móvel de 200 linhas.
2. Marca como anômalas as linhas abaixo da mediana do perfil menos 10 dB.
3. A partir da ponta indicada na D1, estende a zona até encontrar 500 linhas normais seguidas.

Resultado do teste: antilope25 (topo), linhas 0–1641 (2061,13–2065,29 m); botorosa47 (topo), linhas 0–1895 (2178,00–2187,63 m); antilope37 (fim), linhas 86735–89981 (2554,62–2571,11 m; 3,6% do poço). Nos dois topos, o fim da zona difere do fim do alargamento do caliper em até ≈ 0,25 m (≈ 2065,4 m e ≈ 2187,5 m). No antilope37 não há caliper para comparar, e a queda de amplitude começa antes (por volta de 2520 m), então o limite depende do parâmetro de 10 dB. Os parâmetros ficam no notebook (`JANELA = 200`, `FOLGA_DB = 10`, `TOLERANCIA = 500`). As máscaras (`<poço>_excluir.npy`), a tabela `zonas_excluidas_D1.csv` e a figura ficam em `data/processed/npy/` e `caracterizacao/`. Coala88 e tatu22 não recebem máscara.

**D2 (02/11/2026). Reamostrar o antilope25 na vertical por 2 (média de 2 linhas)**, para igualar o passo de 0,00508 m dos demais poços.

- *Justificativa:* o mesmo passo em todos os poços favorece a generalização entre poços, e o pixel fica quase quadrado (≈ 0,54 cm na horizontal e 0,51 cm na vertical). Sem a reamostragem, 256 linhas valeriam 0,65 m no antilope25 e 1,30 m nos demais, e o poço teria cerca de 1.715 patches. *Custo:* perde-se o detalhe vertical de 2,5 mm do antilope25.
- *Base:* a tese (ref. 6, p. 119–120) padronizou a resolução dos pixels entre poços para evitar distorção.

**D3 (02/11/2026). Normalização estática por poço:** percentis 1 e 99 do poço, calculados sem as zonas da D1, com valores recortados em [0, 1].

- *Justificativa:* as medianas de amplitude variam de 28,8 a 37,0 dB entre os poços. A normalização estática preserva o contraste absoluto dentro de cada poço, que os pseudo-rótulos por limiar da Fase 2 usam. A normalização dinâmica (janela móvel; a tese usou 0,1524 m, p. 108) fica como ablação.

**D4 (02/11/2026). Patches de 256 linhas × 180 colunas, passo de 128 linhas (sobreposição de 50%), sem patches que toquem as zonas da D1.**

- *Justificativa:* 256 linhas valem cerca de 1,30 m. Esse tamanho comporta uma senoide de fratura (por exemplo, 38° em poço de ≈ 12,3 pol dá cerca de 24 cm, ou 48 linhas). Um *breakout* de vários metros não cabe inteiro. A sobreposição de 50% é a da tese (p. 130). Treino, validação e teste são divididos por poço, para evitar vazamento entre patches vizinhos. Para a U-Net, as 180 colunas serão completadas por preenchimento circular até 192 (múltiplo de 32) no treino.

As decisões D2, D3 e D4 foram definidas em 02/11/2026. O código está em uma célula do notebook com os parâmetros no topo (`FATOR_VERTICAL`, `PERCENTIS`, `ALTURA_PATCH`, `PASSO_PATCH`).

**Regra de trabalho.** Quando uma decisão depende de conhecimento de petrofísica ou de perfilagem de imagem, o projeto consulta, sempre que possível, uma fonte especializada e a registra na seção "Referências especializadas".


### Trabalhos relacionados e posicionamento

**Princípio do trabalho (definido em 02/11/2026).** O trabalho não precisa gerar algo inédito. Ele deve basear as conclusões em experimentos e referências. Cada conclusão do artigo cita um experimento do projeto ou uma referência.

Leitura completa, em 02/11/2026, do artigo de Silva (2026, ref. 1) e do trabalho de Anjos et al. (2022, ref. 2).

| Trabalho | Dados | Supervisão | Avaliação | Poço não visto | Artefatos | Azimute circular |
|----------|-------|------------|-----------|----------------|-----------|------------------|
| Silva (2026) | WAID, 5 poços | Pseudo-rótulos de limiar (Multi-Otsu em imagem limpa por autoencoder), refinados por redes; 4 classes de amplitude | Concordância com o próprio pseudo-rótulo (chamado aqui de consistência interna) | Treino descrito por intervalo completo; teste em poço não visto não descrito | Tratamento não descrito | Não descrito |
| Anjos et al. (2022) | Dados privados da Petrobras, 18 poços | Máscaras semiautomáticas ajustadas por intérpretes (*breakouts*) | IoU, F1, precisão e revocação | Sim: 5 dobras separadas por poço | Pré-processamento contra ruído; exclusão não descrita | Sim: extensão circular e patch deslocado |
| Fioriti (2026) | Poços do Pré-Sal (privados) | Rótulos de especialista; classificação por patches de 64×64 e 128×128 | Métricas de classificação | Não verificado | Sim: classes de artefatos e remoção de rótulos com artefatos | Argumento da continuidade lateral (rotação de 180°) |
| **Este trabalho (plano)** | WAID, 5 poços | Pseudo-rótulos; conjunto anotado por patch (40 patches, só para validação e teste) e teste sintético | F1 e AUC por patch no conjunto anotado; IoU e Dice no teste sintético; concordância com os pseudo-rótulos como medida secundária | Sim: *leave-one-well-out* | Exclusão das zonas de artefato (D1), com evidência do caliper | Sim: preenchimento circular; deslocamento como ablação |

**O que este trabalho faz com a literatura:**

1. **Testa afirmações publicadas.** Por exemplo: Silva (2026) relata que o refinamento por rede supera o limiar bruto. O projeto repete essa comparação com avaliação independente (conjunto anotado à mão).
2. **Reproduz *baselines* publicados.** Otsu bruto, autoencoder mais Otsu e o refinador só de imagem, que são reproduzíveis com scikit-image e uma CNN rasa.
3. **Aplica práticas de outros trabalhos e mede o efeito.** Deslocamento circular no azimute (Anjos et al., 2022; Fioriti, 2026) e exclusão de zonas com artefato (Fioriti, 2026) entram como ablações.
4. **Compara com os números publicados**, com as ressalvas da tabela mais abaixo.

**Hipóteses testáveis.** O alvo é confirmar, refutar ou deixar inconclusivo cada ponto. Qualquer desfecho serve ao artigo.

| ID | Hipótese | Fonte | Experimento | Ablação |
|----|----------|-------|-------------|---------|
| H1 | Uma CNN treinada com pseudo-rótulos de limiar fica mais próxima do conjunto anotado do que o próprio limiar | Silva (2026) mede ganho de concordância com o pseudo-rótulo | Baselines contra a U-Net, em F1 e AUC por patch no conjunto anotado | principal |
| H2 | O desempenho se mantém em um poço não visto | Anjos et al. (2022): o IoU variou de 33,3% a 70,0% entre dobras separadas por poço | *Leave-one-well-out*, com resultado por poço | principal |
| H3 | A normalização estática por poço supera a global e a por intervalo | Fioriti (2026, p. 105–108); relatório IODP; Silva (2026) normaliza por intervalo | Trocar a normalização | 1 |
| H4 | O deslocamento circular no azimute melhora a generalização | Anjos et al. (2022); Fioriti (2026, p. 134–135) | Treinar sem o deslocamento | 4 |
| H5 | Excluir as zonas de artefato da D1 muda o resultado | Fioriti (2026, p. 121); evidência do caliper (este projeto) | Treinar e avaliar sem a exclusão | 7 |
| H6 | O peso por confiança na perda melhora o resultado | Silva (2026): efeito pequeno quando isolado | Treinar sem o peso | 8 |
| H7 | O pré-treino ImageNet ajuda em imagens acústicas | Sem fonte consultada | Encoder sem pré-treino | 6 |
| H8 | Pseudo-rótulos do caliper acrescentam informação útil | Fioriti (2026): o caliper identifica poço alargado; cobertura limitada no WAID (35% a 98,5%) | Treinar sem esses rótulos, só nos intervalos com BSC | 3 |
| H9 | O pseudo-rótulo da imagem limpa por autoencoder supera o da imagem bruta | Silva (2026): Otsu bruto 0,6138 contra autoencoder mais Otsu 0,7026 (concordância) | Gerar os pseudo-rótulos da imagem bruta | 9 |
| H10 | Os percentis 1 e 99 superam o min-max | Anjos et al. (2022) removem *outliers* e normalizam para [0, 1]; extremos de −82 a 114 dB no WAID | Trocar a normalização | 2 |

**Regra de leitura dos resultados (definida em 15/11/2026, com 3 sementes).** Uma ablação mostra efeito quando a diferença média passa de 1 desvio-padrão entre sementes e tem o mesmo sinal em pelo menos 4 dos 5 poços. Fora disso, o artigo diz "sem efeito detectável" e traz o valor numérico. Resultados de poços individuais aparecem sempre.

**Valores publicados, só como contexto.** Os dados, os intervalos e as métricas são diferentes dos nossos. A comparação com o nosso resultado é qualitativa.

| Trabalho | Método | Métrica | Valor publicado |
|----------|--------|---------|-----------------|
| Silva (2026) | Otsu bruto | Concordância com o pseudo-rótulo (média entre poços) | 0,6138 |
| Silva (2026) | Autoencoder mais Otsu | idem | 0,7026 |
| Silva (2026) | Autoencoder mais K-means | idem | 0,4713 |
| Silva (2026) | Refinador só de imagem | idem | 0,7339 |
| Silva (2026) | CG-DCA (não será reproduzido) | idem | 0,8571 |
| Anjos et al. (2022) | DC-UNet, *breakouts* | IoU média de 5 dobras (mín. 33,29%; máx. 70,03%) | 59,08% |

**Limites.** O trabalho vale para o WAID (5 poços) e para o conjunto anotado, que é pequeno. O uso de pseudo-rótulos por limiar com mapa de confiança já foi publicado (Silva, 2026), e a separação por poço já foi usada por Anjos et al. (2022). O projeto cita esses trabalhos como base, e não como lacuna.

### Definição dos modelos de segmentação (D5, definida em 14/11/2026)

- **Modelo principal:** U-Net com encoder ResNet18, 1 canal de entrada. Entrada de 256 × 192 (180 colunas mais 6 colunas circulares de cada lado). Saída: classes a definir com o conjunto anotado. O ponto de partida é binário (feição escura contra fundo).
- **Baselines:** limiares (Otsu, Multi-Otsu, K-means) e uma CNN rasa de referência (como os refinadores de imagem do artigo de Silva).
- **Fora por enquanto:** U-Net com ResNet34 e U-Net++. Só voltam se sobrar tempo depois da Fase 3.
- **Perda:** Dice mais entropia cruzada. Variante com peso por confiança, como ablação.
- **Otimização:** Adam, taxa de 1e-3, *warm-up* e decaimento por cosseno, como em Anjos et al. (2022). Parada antecipada por um poço de validação separado do poço de teste.
- **Aumento de dados:** deslocamento circular no azimute, só em treino e validação (fontes: Anjos et al., 2022; Fioriti, 2026, p. 134–135). Ablação prevista.
- **Pré-treino ImageNet:** padrão ligado. Ablação sem pré-treino.
- **Entrada:** só a imagem normalizada (D3). Os perfis básicos não entram como canais, diferente de Silva (2026).
- **Validação:** *leave-one-well-out* com 3 sementes, média ± desvio. F1 e AUC por patch no conjunto anotado (40 patches) e IoU e Dice no teste sintético; concordância com os pseudo-rótulos como medida secundária (ver item 49 do registro).
- **Base da escolha:** a família U-Net já foi aplicada a imagens acústicas de poço por Anjos et al. (2022), no contexto Petrobras e PUC-Rio.
- **Justificativa do ResNet18:** o encoder é leve para o Colab. Com 3.465 patches, o conjunto é pequeno, o que favorece um encoder pré-treinado. Não há fonte consultada sobre o ganho do pré-treino ImageNet em imagens acústicas; o efeito é testado na ablação (H7).
- **Definição:** em 14/11/2026, ficaram definidos os quatro pontos propostos (modelo principal, baselines, itens fora do escopo e justificativa).

### Conjunto anotado (plano de quatro camadas, versão de 15/11/2026)

Não há máscaras rotuladas no WAID e não há especialista no projeto. A anotação por região (máscaras) foi trocada por uma decisão por patch, mais simples e com menos julgamento visual. O plano tem quatro camadas de evidência:

1. **Critérios escritos, com fonte.** A ficha `docs/anotacao.md` reúne os critérios de *breakout*, fratura natural, cavidade e artefatos, com a página da tese de Fioriti (2026) em cada critério.
2. **Decisão por patch.** O código sorteia 40 patches (8 por poço, em faixas de profundidade, sem sobreposição). Para cada patch, uma pergunta: "feição escura clara", "sem feição" ou "dúvida" (o patch sai). O código não pinta regiões nem sugere a resposta.
3. **Curva CAL como evidência de apoio.** O CAL aparece ao lado do patch, onde há BSC, com o desvio robusto do CAL máximo em relação à mediana do poço (`cal_z`). O CAL é uma curva única por profundidade. Ela mostra alargamento do poço (*washout* ou *breakout*), mas não mostra o azimute nem o par a 180°. Por isso, o CAL não confirma *breakout* sozinho. Ele apoia a leitura e entra como informação, não como rótulo.
4. **Teste sintético (Fase 3).** Senoides com mergulho e azimute conhecidos, inseridas em trechos reais, dão máscaras exatas. Medem a capacidade de detecção, com IoU e Dice.

Regras: decidir uma vez; não refazer depois de ver o resultado do modelo; patches anotados não entram no treino. O conjunto serve só para validação e teste.

**Rodada 2 (15/11/2026).** A rodada 1 deixou só 6 patches "sem feição" e nenhum no coala88 e no tatu22. A rodada 2 sorteia 40 patches novos, sem sobreposição com a rodada 1. No coala88 e no tatu22, os 8 patches vêm do quartil mais baixo de área escura (fração de pixels abaixo do limiar de Otsu do poço). Nos outros três poços, 4 vêm desse quartil e 4 são aleatórios. Isso aumenta a chance de "sem feição", mas introduz **viés de seleção** (negativos mais fáceis). O artigo declara o viés e mostra as duas rodadas separadas.

**Saídas** (em `data/annotations/`): `sorteio.csv` e `sorteio_2.csv`, `patches_fracao_escura.csv` e `anotacoes.csv` (1 linha por patch: rodada, poço, linhas, profundidades, decisão, `cal_max`, `cal_z`, observação e data).

**Métricas.** Sem máscaras, a métrica no conjunto anotado é por patch: o modelo dá um escore de feição ao patch (fração de pixels preditos como feição), e o F1 e a AUC medem a concordância com as decisões "feição clara" e "sem feição". IoU e Dice só valem no teste sintético. A concordância com os pseudo-rótulos de limiar entra como medida secundária. Com 8 patches por poço, a métrica por poço é grossa; o artigo mostra os números por poço e declara esse limite.

**Limites declarados no artigo.** As decisões seguem critérios da literatura, sem validação por especialista. As métricas medem a concordância com elas, e não a verdade geológica. Não há avaliação da qualidade das bordas contra referência humana.

### Plano de ablação

O documento "Estudo de ablação — trabalho final WAID (ELE2346)", anexado ao projeto, é a base do plano. A versão atualizada em 02/11/2026 traz as mudanças desta seção. Resumo das configurações, com prioridade sugerida:

| # | Configuração | Hipótese | Prioridade |
|---|--------------|----------|------------|
| 0 | Modelo completo (referência) | — | obrigatória |
| 1 | Normalização global ou por intervalo, em vez de por poço (D3) | H3 | alta |
| 2 | Min-max em vez de percentis 1 e 99 | H10 | média |
| 3 | Sem pseudo-rótulos do caliper (vale só nos intervalos com BSC) | H8 | média |
| 4 | Sem deslocamento circular no azimute | H4 | alta |
| 5 | U-Net sem *skip connections* | sem fonte consultada | baixa |
| 6 | Encoder sem pré-treino | H7 | baixa |
| 7 | Sem a exclusão das zonas da D1 (nova) | H5 | alta |
| 8 | Sem peso por confiança na perda (nova) | H6 | média |
| 9 | Pseudo-rótulo da imagem bruta, sem autoencoder (nova) | H9 | média |

Cada configuração custa 15 treinos (5 poços × 3 sementes). O tempo de um treino será medido no início da Fase 3, e a lista final de configurações sai depois dessa medida.

## Cronograma

Data-alvo de entrega: **02/12/2026**.

- [x] Fase 1 – Exploração e pré-processamento (notebook `01_exploracao.ipynb`)
  - [x] Download do dataset WAID para o Drive (31/10/2026; 62 arquivos de imagem confirmados)
  - [x] Carga e concatenação ordenada dos subarquivos (5 poços confirmados no Colab em 31/10/2026)
  - [x] Conversão para PNG (65 arquivos em `data/processed/png/`, 01/11/2026)
  - [x] Inspeção visual dos PNGs de visão geral (5 poços, 01/11/2026). Trechos em resolução total: topo/fim do antilope25, do botorosa47 e do antilope37 analisados fora do Colab; zonas das faixas diagonais do coala88 e do tatu22 pendentes
  - [x] Conversão das imagens para `.npy` e perfil de amplitude média por linha (confirmados em 01/11/2026; leitura do `.npy` do coala88 em 0,1 s, contra 14,1 s pelo CSV)
  - [x] Máscara de exclusão das zonas anômalas (decisão D1; executada no Colab em 02/11/2026; o limite do fim do antilope37 é provisório)
  - [x] Normalização e geração de patches (decisões D2–D4 definidas em 02/11/2026; executada no Colab em 02/11/2026: 3.465 patches válidos)
  - [x] Entregável: caracterização do dataset (células A a D executadas em 01/11/2026; tabelas e histograma salvos em `data/processed/caracterizacao/`)
  - [x] Entregável: definição dos modelos de segmentação (D5, definida em 14/11/2026)
- [x] Fase 2 – Rótulos e *baselines* (notebooks `02_baselines.ipynb` e `03_cnn_unet.ipynb`; concluída em 16/11/2026)
  - [x] Ficha de critérios de anotação (`docs/anotacao.md`; definida em 15/11/2026)
  - [x] Ferramenta de anotação por patch (`src/labels/anotacao.py`; escrita e testada com dados sintéticos em 15/11/2026; ainda não executada no Colab)
  - [x] Decisão por patch em 40 patches (8 por poço; executada no Colab em 15/11/2026)
  - [x] Segunda rodada: mais 40 patches, dirigida a negativos (executada em 16/11/2026)
  - [x] Revisão da anotação pela ficha (2 patches para "dúvida") e reanotação às cegas de 10 patches (executadas em 16/11/2026; concordância 10 de 10)
  - [x] *Baselines* de limiar: Otsu, Multi-Otsu e K-means, avaliados por patch (seção 2.12 do `02_baselines.ipynb`; executados no Colab em 16/11/2026)
  - [x] *Baseline* CNN rasa (seções 3.1 a 3.6 do `03_cnn_unet.ipynb`; executada no Colab em 16/11/2026, 3 sementes)
  - [x] U-Net de referência (linha 0 da ablação; mesmo notebook; executada em 16/11/2026, 3 sementes)
- [ ] Fase 3 – Experimentação (notebook `04_sintetico.ipynb`; 21 a 27/11/2026, adiantada em 16/11/2026)
  - [x] Tempo de um treino medido (item 59)
  - [x] Teste sintético: código escrito, exemplos e *baseline* de Otsu executados (itens 64 e 65); contrastes redefinidos antes do treino
  - [x] Treino de referência com a métrica sintética (seções 4.6 e 4.7 do `04_sintetico.ipynb`; executado em 16/11/2026; item 67)
  - [x] Ablações H4, H5 e H3 executadas em 17/11/2026 (resultados no item 74; leitura da H3 limitada pela U22) (prioridade alta; código de H4, H5 e H3 escrito (H4 e H5 em 16/11/2026, H3 em 17/11/2026), notebook `05_ablacoes.ipynb`, ainda não executado)
  - [ ] H10, H8, H6 e H9, se houver tempo
- [ ] Fase 4 – Escrita do artigo (28/11 a 01/12/2026; a escrita das seções de dados, método e trabalhos relacionados começa em paralelo em 18/11)
- [ ] Entrega: 02/12/2026

Notebooks por fase (numerados na ordem de criação e execução):

| Notebook | Fase | Conteúdo |
|----------|------|----------|
| `01_exploracao.ipynb` | 1 | carga, PNG, caracterização, normalização e patches |
| `02_baselines.ipynb` | 2 | conjunto anotado, pseudo-rótulos e *baselines* |
| `03_cnn_unet.ipynb` | 2 | CNN rasa e U-Net de referência (*leave-one-well-out*) |
| `04_sintetico.ipynb` | 3 | teste sintético e referência com IoU e Dice sintéticos |
| `05_ablacoes.ipynb` | 3 | ablações H4, H5 e H3 (global e por intervalo), conferência do pseudo-rótulo, H3 com teste sintético comum e regra de leitura |
| `06_investigacao_tatu22.ipynb` | 3 | investigação do pseudo-rótulo do tatu22 (U23), sem GPU e sem treino |

Convenções dos notebooks:

- O nome começa com um prefixo numérico (`01_`, `02_`, ...), na ordem de criação e de execução.
- A primeira célula importa as bibliotecas (`os`, `sys`, `numpy`, `pandas`, `drive`, ...). Cada importação aparece uma só vez.
- Os imports do projeto (`src...`) ficam depois da montagem do Drive e do `sys.path.insert`, em uma célula de configuração que também define as constantes (`pocos`, `img`, `PNG_DIR`).
- Os comentários de código usam o gerúndio e ficam acima do bloco que descrevem.
- Cada notebook começa com uma célula de título (título do trabalho, disciplina, autor, fase). O título usa `#`, a fase usa `##` e as seções numeradas usam `###` (ex.: 1.1, 1.2). Assim, o Colab permite abrir e fechar cada parte no índice e nos cabeçalhos.
- Cada pasta de saída guarda só o que o código daquela etapa gera (por exemplo, `data/processed/png/<poço>/` guarda só os 13 PNGs de `salvar_png_poco`). Arquivos feitos fora do notebook, como as figuras de comparação, ficam em `docs/`.

## Estado do projeto e pendências (para retomar)

Estado em 15/11/2026. Fase 1 concluída. Itens 1 a 5 da tabela B resolvidos. A Fase 2 começa em 16/11/2026 (ver "Cronograma"). Ao retomar, comece pelo item 6 e pela ficha de critérios (`docs/anotacao.md`), definida em 15/11/2026.

### A. Concluído (decisões e execuções)

- **Execuções no Colab (notebook `01_exploracao.ipynb`, 18 células):** download e carga dos 5 poços; PNGs (13 por poço); caracterização (BSC, valores ausentes, histogramas); conversão para `.npy` e perfil de amplitude; máscara da D1; normalização e patches (3.465 patches válidos de 256 × 180). As saídas do notebook de 02/11/2026 são iguais às do teste fora do Colab.
- **Decisões definidas:** D1 (excluir zonas de artefato nas pontas de 3 poços; limite do antilope37 provisório, 10 dB), D2 (reamostrar o antilope25 por 2), D3 (normalização estática por poço), D4 (patches de 256 × 180, passo 128) e D5 (U-Net com ResNet18; baselines com limiares e CNN rasa).
- **Planos definidos em 14 e 15/11/2026:** regra de leitura das ablações (3 sementes), hipóteses H1 a H10 com prioridades (H3, H4 e H5 primeiro), plano de quatro camadas para o conjunto anotado e o cronograma até 02/12.
- **Regras de trabalho:** decisões de domínio apoiadas em fonte especializada; o trabalho não busca ineditismo e baseia as conclusões em experimentos e referências; notebooks numerados, com imports de bibliotecas na primeira célula e comentários em gerúndio.

### B. Ainda não confirmado

| # | Item | O que falta |
|---|------|-------------|
| 1 | ~~Proposta de modelos~~ | **Resolvido em 14/11/2026 (D5).** Fase 1 concluída. |
| 2 | ~~Regra de leitura das ablações~~ | **Resolvido em 15/11/2026:** 3 sementes; diferença maior que 1 desvio-padrão; mesmo sinal em 4 de 5 poços. |
| 3 | ~~Hipóteses H1 a H10 e prioridades da ablação~~ | **Resolvido em 15/11/2026.** Ablações H3, H4 e H5 primeiro. Documento do projeto substituído. |
| 4 | ~~Limite do fim do antilope37 (`FOLGA_DB = 10` dB)~~ | **Resolvido em 15/11/2026:** manter 10 dB, tratar como provisório e testar a sensibilidade (6 e 14 dB) só se sobrar tempo. |
| 5 | ~~Cronograma~~ | **Resolvido em 15/11/2026:** Fase 2 de 16 a 20/11; Fase 3 de 21 a 27/11; Fase 4 de 28/11 a 01/12; entrega em 02/12. |
| 6 | Modelo (template) e limite de páginas do artigo | **Parcialmente resolvido em 17/11/2026.** Modelo recebido: SBC (LaTeX, Word e OpenOffice). O modelo não fixa o limite de páginas ("definido pela conferência") e não trata da declaração de uso de IA. Aguardam resposta da professora: limite de páginas, idioma (português ou inglês) e regra de declaração de IA. **Informação informal em 18/11/2026:** um colega que cursou a disciplina no período anterior informou 8 páginas para o artigo. Não é a regra oficial; a confirmação com a professora continua pendente. A prévia atual tem 6 páginas, com várias seções ainda em `\pend`. **Decisão em 18/11/2026:** finalizar o artigo para 8 páginas (corpo e referências), com a informação informal do colega; se a professora informar outro limite, a questão é reaberta. O campo de e-mail do modelo SBC é genérico: o modelo não exige e-mail institucional. Usado, então, e-mail pessoal do autor. |
| 7 | ~~Conjunto anotado por patch (80 patches)~~ | **Fechado em 16/11/2026.** 55 feição clara, 16 sem feição, 9 dúvida (71 usáveis). Reanotação às cegas: 10 de 10 concordantes (kappa 1,00; n = 10). |
| 17 | Métrica principal das ablações com o conjunto anotado reduzido | **Resolvido em 16/11/2026.** Ficou definido usar a concordância com os pseudo-rótulos no poço de teste (Dice) e o IoU do teste sintético para a regra de leitura das ablações, e o conjunto anotado (F1 e AUC por patch) como verificação externa. Motivo: com 71 patches, o IC 95% da AUC tem cerca de 0,13 de largura (item 57). |
| 18 | Circularidade da avaliação (anotação e limiar usam a pista "escuro") | **Proposta, aguarda decisão.** Avaliar com um critério que não dependa só da amplitude baixa, por exemplo a forma (senoide contínua) ou a concordância com o caliper onde há BSC. Ver U9. |
| 19 | Sensibilidade dos modelos aos hiperparâmetros | **Proposta, aguarda decisão.** Variar épocas e taxa de aprendizado e ver se a conclusão sobre o Otsu se mantém. Ver U13. |
| 8 | Mapeamento do azimute (180 colunas = 360°, 2° por coluna) | Plausível pela tese (ref. 6). Confirmar na documentação do WAID. |
| 9 | Valor "2338 m" do tatu22 na análise visual | O 3º painel começa perto de 2382 m. Confirmar o número. |
| 10 | Faixas diagonais do coala88 (2340–2370 m) e do tatu22 (2382–2424 m) | Não há PNG em resolução total dessas zonas. Há uma célula oferecida e não pedida. A causa é desconhecida. |
| 11 | Causa física das zonas de artefato | Topos do antilope25 e do botorosa47: hipótese de arrombamento, apoiada pelo caliper e sem confirmação plena. Fim do antilope37: causa desconhecida. |
| 12 | Outros intervalos de amplitude baixa no perfil (antilope25 perto de 2290 m e de 2600 a 2619 m; coala88 perto de 2645 m; vários no tatu22) | Sem análise. Não fazem parte da D1. |
| 13 | Referências | Obter e ler Lofts e Bourke (1999). Do artigo de Silva (2026), não foram vistas as figuras e as tabelas suplementares. O notebook oficial de segmentação por limiar do WAID ainda não foi lido. |
| 14 | Unidade do caliper (CAL) | Provavelmente polegadas. Confirmar. |
| 15 | Vídeo explicativo e página HTML interativa | O vídeo foi entregue. As ofertas de registrá-lo no README e de criar a página HTML não tiveram resposta. |
| 16 | GitHub | Concluído: o repositório foi publicado em https://github.com/jorgesouza63/puc_pel_mel_sit_ele2346. O `git clone`, o selo do Colab e o notebook `00_setup_colab` usam esse endereço. |

### C. Ordem sugerida ao retomar

1. Itens 1 a 5 da tabela B resolvidos em 14 e 15/11/2026. Resta o item 6 (modelo e limite de páginas do artigo): perguntar à professora da disciplina até 17/11. São respostas curtas.
2. ~~Fechar a Fase 1~~ (feito em 14/11/2026, D5).
3. Iniciar a Fase 2 em `02_baselines.ipynb`: definir classes e critérios do conjunto anotado (item 7); reproduzir os *baselines* (Otsu bruto, autoencoder mais Otsu, K-means e o limiar do notebook oficial do WAID); treinar a U-Net com pseudo-rótulos; avaliar com *leave-one-well-out*.
4. No início da Fase 3, medir o tempo de um treino no Colab e fechar a lista de ablações.
5. Recalcular o cronograma. A data-alvo continua 02/12/2026.

### D. Arquivos de apoio

`README.md` (registro vivo), `docs/ablacao.md` (plano de ablação; a versão atualizada substituiu a anterior no projeto de apoio), `docs/anotacao.md` (ficha de critérios), `docs/limitacoes.md` (rascunho da seção de limitações), `01_exploracao.ipynb`, `02_baselines.ipynb`, `src/config.py`, `src/labels/anotacao.py`, `src/labels/limiares.py`, `src/eval/patches.py`, `src/models/modelos.py`, `src/train/treino.py`, `src/synth/sintetico.py`, `src/eval/ablacao.py`, `03_cnn_unet.ipynb`, `04_sintetico.ipynb`, `05_ablacoes.ipynb`, `src/data/` (`dataset_patches.py`, `loader.py`, `waid_concat.py`, `png_export.py`), `scripts/download_waid.py`, os arquivos `.npy` em `data/processed/npy/` e as tabelas e figuras em `data/processed/caracterizacao/`.

## Registro de incertezas

Este registro lista o que o trabalho afirma, a evidência que o sustenta e o grau de confiança. Ele existe para manter as afirmações do artigo proporcionais às evidências. Graus: **alta** (medido e conferido), **média** (medido, com limites conhecidos), **baixa** (escolha do autor, hipótese ou amostra pequena) e **não explicada** (resultado sem causa conhecida). O rascunho da seção de limitações está em `docs/limitacoes.md`.

**Convenção de linguagem para o artigo e para o README.** Resultado medido: "foi observado", com o número e o intervalo. Ausência de efeito: "não foi detectada diferença", nunca "não há diferença". Hipótese: "sugere", "pode indicar". Escolha de projeto: "foi definido", com o motivo. Decisões do autor usam "definida", "adotada" ou "escolhida"; a H11 (limiar Multi-Otsu no pseudo-rótulo) e qualquer ablação ou escolha de projeto que o autor decidiu é "decisão do autor". Nenhuma afirmação sobre geologia sem fonte especializada.

| Nº | Afirmação | Evidência | Confiança | O que reduziria a incerteza | Estado |
|---|---|---|---|---|---|
| U1 | 180 colunas valem 2° cada | A tese de Fioriti descreve 180 amostras por volta; a documentação do WAID não confirma | média | Conferir na documentação do WAID | aberto |
| U2 | O pixel tem cerca de 0,54 × 0,51 cm | Calculado a partir de U1 e do diâmetro do poço | média | Resolver U1 | aberto |
| U3 | Os limites das zonas da D1 identificam artefatos | Regra própria (10 dB, 200 e 500 linhas); conferida com o caliper só nos dois topos; o antilope37 não tem caliper | média nos topos, baixa no fim do antilope37 | Teste de sensibilidade ao parâmetro de 10 dB (decidido) | pendente |
| U4 | Os outros intervalos de amplitude baixa não distorcem treino e avaliação | Sem análise (antilope25, coala88 e tatu22) | baixa | Inspecionar esses intervalos (o notebook 06 mostrou um regime de baixa amplitude no tatu22 a partir de cerca de 2.320 m e trechos no botorosa47; ver U23) | parcial |
| U5 | As faixas diagonais do coala88 e do tatu22 são espiralamento ou *reaming* | Hipótese visual | baixa | Fonte especializada ou conferência com perfis | aberto |
| U6 | A normalização por poço é adequada | Escolha (D3). A H3 foi executada em 17/11/2026, mas muda também o teste sintético (U22); por isso o resultado não prova que a normalização por poço é melhor nem pior | baixa | Métrica de detecção que não dependa do limiar (U22, U24) | parcial; H3 inconclusiva |
| U7 | As anotações seguem critérios da literatura | Uma só fonte (Fioriti, 2026); Lofts e Bourke (1999) não obtido; sem especialista | baixa quanto à correção geológica | Obter a segunda fonte; validar com especialista, se houver | aberto |
| U8 | O conjunto anotado mede o desempenho | 71 patches (55 e 16); tatu22 sem negativos; coala88 com 2; IC 95% da AUC com cerca de 0,13 de largura | média | Mais negativos; mais patches | aberto |
| U9 | A AUC de 0,93 do Otsu indica detecção de feições | O critério de anotação e o escore usam a mesma pista ("escuro"); a avaliação é parcialmente circular | baixa | Critério independente (item 18 da tabela B) | pendente |
| U10 | As anotações são repetíveis | 10 de 10 patches iguais, kappa 1,00; mesma pessoa; mais de 10 h de intervalo | baixa | Reanotar mais patches; segunda pessoa | aberto |
| U11 | Excluir os 9 patches em dúvida não distorce as métricas | Sem análise; essa exclusão tende a deixar só os casos fáceis | baixa | Medir o escore dos patches em dúvida | aberto |
| U12 | O pseudo-rótulo de Otsu serve como alvo de treino | Concordância com as anotações; marca também sombras e artefatos | média | H9 e H8 (outros pseudo-rótulos) | pendente |
| U13 | Os modelos ficam abaixo do Otsu | Diferença pareada de AUC de −0,15 e −0,12, com intervalos que não incluem zero; uma só configuração (10 épocas, 1e-3) | média | Item 19 da tabela B | pendente |
| U14 | A CNN rasa e a U-Net não se separam | Diferença de 0,002 no IoU sintético; 71 patches e 3 sementes dão pouco poder | baixa | Mais sementes; mais patches | aberto |
| U15 | O Dice contra o pseudo-rótulo é uma medida confiável | Varia de 0,41 a 0,86 entre poços; separa os modelos, ao contrário do IoU sintético; causa desconhecida | não explicada | Analisar por poço | aberto |
| U16 | A soma das probabilidades acumula erro e explica o desempenho abaixo do Otsu | Hipótese de trabalho, não testada | baixa | Comparar escores por patch dos modelos e do Otsu | aberto |
| U17 | O teste sintético mede a capacidade de detecção | Parâmetros escolhidos pelo autor; só senoides; região ignorada definida pelo limiar de Otsu | baixa | Variar os parâmetros; incluir manchas | aberto |
| U18 | O trabalho segmenta fraturas, cavidades e *breakouts* | Só segmenta "feição escura", sem separar os tipos | alta (como limite) | Trabalho futuro | declarado |
| U19 | A regra de leitura das ablações separa efeito de ruído | Limiar de 1 desvio entre 3 sementes e sinal em 4 de 5 poços: escolha do autor; o desvio de 3 sementes é instável | baixa | Mais sementes nas ablações principais | aberto |
| U20 | O pré-treino ImageNet ajuda | Sem fonte consultada; H7 não foi executada | baixa | Ablação H7 | pendente |
| U22 | O IoU sintético compara a H3 com a referência em condições iguais | Testado em 17/11/2026 (seções 5.10 e 5.11, teste comum, conferido: diferença máxima de `x` 0,00024; máscaras e regiões ignoradas iguais nos 5 poços). O resultado inverteu o sinal da H3 por intervalo (de +0,129 no item 74 para −0,406 no item 77). O teste comum define as feições pelo limiar de Otsu da referência, e por isso favorece a definição de "escuro" da referência (U24) | alta (de que o teste depende da definição) | Métrica que não dependa do limiar (por exemplo, AUC por pixel da probabilidade do modelo contra a máscara; exige salvar mapas de probabilidade e retreinar) | confirmado; sem solução implementada |
| U24 | O IoU sintético mede a detecção de feições, e não a coincidência com a definição de "escuro" de uma configuração | Hipótese, compatível com o item 77: modelos treinados com outro limiar marcam outros pixels como escuros, e os pixels a mais contam como erro no teste comum; o teste com limiar próprio (item 74) favorece cada configuração. Não foi testada diretamente | baixa | Medir o IoU com um corte de contraste que seja o mesmo em dB para todos, e a AUC por pixel | aberto |
| U23 | O pseudo-rótulo de Otsu marca feições escuras e não outra coisa | **Medido em 17/11/2026 (notebook 06): em parte, não.** No tatu22, 56,5% dos patches válidos (361) têm mais de 50% de pixels escuros e vários têm 100%, em imagem cinza uniforme sem feição visível; no coala88 são 8,9% e no botorosa47, 4,2%. O limiar de Otsu do tatu22 (0,760) separa dois regimes de amplitude (histograma em dB com duas massas), e não feições. A causa física do regime (ferramenta, lama, litologia, poço alargado) não foi investigada | baixa | Testar outro limiar de pseudo-rótulo (H11, seção 5.12); fonte especializada sobre o intervalo de baixa amplitude do tatu22 | confirmado em parte; H11 com código pronto |
| U21 | O tempo de treino medido vale para as ablações | 10 épocas, GPU gratuita do Colab, tempo medido em 16/11/2026 | média | Conferir após cada ablação | aberto |

**Correções de análise já feitas** (mostram onde uma leitura inicial estava imprecisa e foi revista): o CAL mostra alargamento do poço, mas não o par de zonas a 180° (item 49); o intervalo entre as duas anotações passou de "7 a 8 h" para "mais de 10 h" por causa do fuso do Colab (item 54); o cálculo da AUC por poço tinha erro de *bootstrap* (itens 55 e 62); a leitura "os intervalos se sobrepõem" foi substituída pela comparação pareada (itens 61 e 63); os contrastes do teste sintético não discriminavam e foram redefinidos (item 65).

## Resultados

### Conjunto anotado por patch (duas rodadas, 15 e 16/11/2026)

| Poço | Feição clara | Sem feição | Dúvida (sai) |
|------|-------------:|-----------:|-------------:|
| antilope25 | 9 | 6 | 1 |
| antilope37 | 9 | 5 | 2 |
| botorosa47 | 12 | 3 | 1 |
| coala88 | 11 | 2 | 3 |
| tatu22 | 16 | 0 | 0 |
| **Total** | **57** | **16** | **7** |

Rodada 1 (aleatória): 27 / 6 / 7. Rodada 2 (dirigida a baixa área escura, com viés de seleção): 30 / 10 / 0. Patches usáveis: 73. O tatu22 não tem nenhum patch "sem feição"; o coala88 e o botorosa47 têm 2 e 3. Por isso, F1 e AUC por poço só são estáveis no antilope25 e no antilope37. Os demais valores vêm dos 5 poços juntos, com intervalo de confiança por *bootstrap*. Curva CAL em 46 dos 80 patches; mediana de `cal_z`: feição clara 0,48; sem feição −0,27; dúvida 0,04 (n = 33, 10 e 3). O CAL não separa as classes e serve só como apoio.

### Resultados dos métodos

*A preencher após a Fase 3.*

| Método | IoU | Dice | Observações |
|--------|-----|------|-------------|
| Limiar fixo | – | – | baseline |
| Otsu | – | – | baseline |
| U-Net (pseudo-rótulos) | – | – | – |
| U-Net + pré-treino | – | – | – |

## Referências especializadas

Consultadas em 01/11/2026 e 02/11/2026. Nas referências 3 a 5, só foi lido o trecho recuperado na busca. As referências 1 e 2 foram lidas por completo em 02/11/2026. A referência 6 foi lida nas seções indicadas.

1. SILVA, J. L. L. J. *Weakly supervised multimodal segmentation of acoustic borehole images with depth-aware cross-attention*. arXiv:2603.20729, 21 mar. 2026. <https://arxiv.org/abs/2603.20729>. Trabalho relacionado: refina pseudo-rótulos obtidos por limiar, com perfis de poço alinhados em profundidade, e faz análise entre poços. Lido por completo em 02/11/2026 (texto da versão 1, em revisão; figuras e tabelas suplementares não foram vistas). Ver "Trabalhos relacionados e posicionamento".
2. ANJOS, G. B. et al. *Automatic segmentation of breakouts in acoustic borehole image logs using convolutional neural networks*. CILAMCE 2022. <https://publicacoes.softaliza.com.br/cilamce2022/article/download/5555/4621/1539>. Descreve *breakouts* como cavidades verticais irregulares, aproximadamente 180° entre si, e rótulos obtidos por segmentação do histograma de amplitude com ajuste manual. Usa uma variante da U-Net (DC-UNet), patches de 256 × 256 com extensão circular das colunas e um segundo patch deslocado em 128 colunas, e validação cruzada em 5 dobras separadas por poço (18 poços; IoU média de 59,1%).
3. IODP/LDEO. *UBI Image Data Processing* (relatórios de poços das expedições 327 e 352, entre outros). <https://brg.ldeo.columbia.edu/data/iodp-usio/exp327/U1362A/documents/327-U1362A_info-ubi.html> e <https://brg.ldeo.columbia.edu/data/iodp-usio/exp352/U1440B/documents/352-U1440B_info-ubi.html>. Princípio da ferramenta (transdutor rotativo que mede amplitude e tempo de trânsito), efeito da excentricidade na amplitude e causas de imagem de baixa qualidade (poço de diâmetro grande, movimento da ferramenta, pontos de aderência).
4. Patente US 10.551.518 B2. *Method to correct eccentricity in ultrasonic image profiles*. <https://patents.google.com/patent/US10551518>. A atenuação da amplitude por excentricidade da ferramenta é um efeito geométrico, e não uma propriedade acústica da rocha. As correções clássicas podem gerar artefatos em zonas de variância alta (fraturas, geodos e cavernas).
5. EarthDoc/EAGE (2026), DOI 10.3997/2214-4609.2026101491 (título não confirmado). <https://www.earthdoc.org/content/papers/10.3997/2214-4609.2026101491>. Resumo: artefatos de aquisição afetam tempo de trânsito e amplitude em imagens acústicas, e a excentricidade introduz um viés azimutal de baixa frequência. Cita LOFTS, J. C.; BOURKE, L. B. *The recognition of artefacts from acoustic borehole imaging devices*. Geological Society, London, Special Publications, v. 159, p. 59–76, 1999. **Esta referência ainda não foi obtida nem lida.** A tese (ref. 6) resume a classificação de artefatos que ela propõe.

6. FIORITI, L. S. *Classificação digital de image facies controladas por variações de texturas, estruturas e artefatos em perfis de imagem analisados por aprendizado profundo*. Tese (Doutorado em Ciências) – Escola Politécnica, Universidade de São Paulo, São Paulo, 2026. Versão corrigida; orientador: C. C. Carneiro. Lidos: resumo (p. 8–11), seções 3.1.3 e 3.2 (p. 57–70), limpeza e padronização dos dados (p. 105–130), rotulação de artefatos (p. 164–165 e 192) e conclusões (p. 296–300). Usos: classificação de artefatos (Lofts e Bourke, 1999) com a assinatura de cada um em imagens de amplitude; precedente de limpeza de dados com remoção de trechos com artefatos; amostragem azimutal de 180 pontos por volta de 360°; trabalho relacionado (classificação de *image facies* com aprendizado profundo em poços do Pré-Sal, com rótulos de especialista). Limites: o modelo dedicado a detectar artefatos teve desempenho limitado (acurácia contínua de cerca de 35%, resumo), e a tese trata de classificação por patches com rótulos, e não de segmentação sem rótulos.

## Uso de ferramentas de IA

Assistente de IA usado para apoio ao código e à revisão do texto. Decisões, experimentos e conclusões são do autor.

## Licenças e atribuição

- Código deste repositório: [Apache License 2.0](LICENSE).
- Dados: WAID/PETROBRAS, licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Os dados originais pertencem à PETROBRAS e devem ser citados conforme o repositório oficial.

## Registro de execução

Um item por passo executado, em ordem de data.

### 31/10/2026

1. **Montagem do Google Drive no Colab.**
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
   Resultado: `Mounted at /content/drive`.

2. **Download completo do WAID para o Drive.** O teste com `--dry-run` não foi executado (a linha estava comentada). O teste rápido com o poço coala88 também não foi executado.
   ```
   !python /content/drive/MyDrive/waid/scripts/download_waid.py --dest /content/drive/MyDrive/waid/data/raw
   ```
   Resultado: commit `f8926d0`, 67 arquivos baixados (5 BSC + 62 AMP), 1,21 GB, sem erros na saída. Destino: `MyDrive/waid/data/raw/` (subpastas `bsc/` e `img/`).

3. **Apresentação em PDF adicionada ao projeto.** Arquivo `ELE2346 - Deep Learning - Trabalho - Apresentação.pdf` colocado na pasta `docs/` do computador local. É a versão em PDF da apresentação usada como base do trabalho.

4. **Renomeação do PDF.** O arquivo passou a se chamar `docs/ELE2346_apresentacao.pdf`.

5. **PDFs de `docs/` fora do Git.** Regra `docs/*.pdf` adicionada ao `.gitignore`, porque a capa da apresentação tem dados pessoais. O PDF do artigo, em `paper/`, não é afetado.

6. **Conferência dos arquivos de imagem no Drive.**
   ```
   !ls /content/drive/MyDrive/waid/data/raw/img | wc -l
   ```
   Resultado: `62` (igual ao total esperado de subarquivos AMP).

7. **Decisão: usar a função oficial `concat_IMG_data` do WAID.** Código copiado em `src/data/waid_concat.py` e invólucro em `src/data/loader.py`. O teste com o poço coala88 foi feito em ambiente próprio, não no Colab do projeto. O resultado está na seção "Carga das imagens". O teste nos 5 poços, no Colab, ainda não foi executado.

8. **Primeira tentativa de importar o carregador no Colab.**
   ```python
   import sys
   from src.data.loader import load_well_image
   ```
   Resultado: `ModuleNotFoundError: No module named 'src'`. Causa: a linha `sys.path.insert(...)` estava comentada e ficava depois do `import`. Correção: inserir `/content/drive/MyDrive/waid` no `sys.path` antes do `import` (exemplo atualizado na seção "Carga das imagens").

9. **Carga do poço coala88 no Colab, com o invólucro do projeto.**
   ```python
   import sys
   sys.path.insert(0, "/content/drive/MyDrive/waid")   # pasta que contém "src"

   from src.data.loader import load_well_image

   df = load_well_image("coala88", "/content/drive/MyDrive/waid/data/raw/img")
   ```
   Resultado: `coala88: 87796 linhas x 180 colunas | profundidade 2212.00 a 2658.00 m | passo mediano 0.00488 m | saltos: 0 | reordenado: não`. Os números batem com o teste feito antes fora do Colab. Os outros 4 poços ainda não foram carregados no Colab.

10. **Carga dos 5 poços no Colab (primeira versão do carregador).**
    ```python
    pocos = ["antilope25", "antilope37", "botorosa47", "coala88", "tatu22"]
    for p in pocos:
        df = load_well_image(p, img)
        ...
    ```
    Resultado: `ValueError: antilope25: 163846 profundidades duplicadas. Verifique os subarquivos.` O laço parou no primeiro poço. Causa: o rótulo de profundidade do antilope25 tem 2 casas decimais e se repete cerca de 4 vezes. Os dados não estão corrompidos. Correção: `src/data/loader.py` agora trata rótulo repetido como aviso, ordena de forma estável (`mergesort`) e informa o passo médio e as linhas por rótulo. A nova versão ainda não foi executada no Colab.

11. **Recarga do carregador corrigido e carga dos 5 poços no Colab.**
    ```python
    import importlib, src.data.loader
    importlib.reload(src.data.loader)
    from src.data.loader import load_well_image
    ```
    Em seguida, o laço dos 5 poços (o mesmo do item 10). Resultado: os 5 poços carregaram sem erro, com `saltos: 0` e `reordenado: não` em todos. Os números são iguais aos do teste fora do Colab. O aviso de rótulo repetido apareceu só no antilope25 e no botorosa47. Resumo (profundidade em m, amplitude em dB):
    ```
          poco  linhas  colunas  prof_ini_m  prof_fim_m  nan  amp_p1  amp_mediana  amp_p99
    antilope25  219635      180     2061.13     2619.00    0   16.95        36.58    39.55
    antilope37   89982      180     2114.00     2571.11    0    9.21        29.47    33.40
    botorosa47   81062      180     2178.00     2589.79    0    3.86        35.48    37.86
       coala88   87796      180     2212.00     2658.00    0   24.10        36.95    39.79
        tatu22   81983      180     2174.20     2590.67    0   18.03        28.83    32.46
    ```
    A Carga está concluída. A mediana de amplitude varia de 28,8 a 37,0 dB entre os poços, o que confirma a necessidade de normalizar por poço.

12. **Execução completa do notebook `01_exploracao.ipynb` e salvamento das saídas.** As 8 células rodaram em ordem crescente, sem erro, em sessão nova do Colab (o Drive foi montado de novo). O download (célula 2) encontrou os 67 arquivos já no Drive e não baixou nada. Os resultados repetiram os do item 11: mesma tabela-resumo dos 5 poços. O laço dos 5 poços levou 139 s. O notebook ainda não inclui a conversão para PNG.

13. **Ajuste de comentários no notebook `01_exploracao.ipynb`.** A definição de `img` passou da célula 8 para a célula 5, com comentário próprio. O comentário da célula 8 passou a dizer apenas "Definindo a lista de poços". O código e as saídas não mudaram.

14. **Padrão de nomes definido.** O prefixo numérico (`01_`, `02_`, ...) vale para os notebooks do projeto, na ordem de criação e de execução. O próximo notebook começa com `02_`. Os módulos `.py` ficam sem número, porque nome iniciado por número não funciona no `import` comum. O `png_export.py` ficou com o nome original: a renomeação para `02_png_export.py` foi entregue e desfeita antes de ser usada.

15. **Mapa de notebooks por fase definido.** O `01_exploracao.ipynb` cobre toda a Fase 1 (carga, PNG, caracterização, normalização e patches, definição dos modelos). O `02_baselines.ipynb` só começa na Fase 2. O checklist da Fase 1 passou a incluir os dois entregáveis do cronograma.

16. **Padrão da primeira célula dos notebooks.** Em todo notebook, a primeira célula define as importações de bibliotecas, sempre que possível. Os imports do projeto (`src...`) não entram nela, porque dependem da montagem do Drive e do caminho de busca. O padrão e as demais convenções (prefixo numérico, comentários em gerúndio) ficaram descritos na seção "Cronograma".

17. **Célula de configuração do `01_exploracao.ipynb` (sys.path, imports do projeto e constantes).**
    Resultado: o `import` de `src.data.loader` funcionou e o de `src.data.png_export` falhou com `ModuleNotFoundError: No module named 'src.data.png_export'`. O Python não achou o arquivo `png_export.py` nesta sessão, embora a listagem de uma sessão anterior mostrasse o arquivo. Depois de desconectar e excluir o ambiente de execução e reconectar, tudo funcionou. A causa exata não foi confirmada (provável lista de arquivos desatualizada na sessão).

### 01/11/2026

18. **Geração dos PNGs de inspeção visual no Colab.**
    ```python
    for p in pocos:
        df = load_well_image(p, img)
        arquivos = salvar_png_poco(df, p, PNG_DIR)
        print(p, len(arquivos), "PNGs")
        del df
    ```
    Resultado: 13 PNGs em cada um dos 5 poços (65 no total), salvos em `MyDrive/waid/data/processed/png/<poço>/`. As linhas de resumo do carregador repetiram os números dos itens 11 e 12. A inspeção visual ainda não foi feita.

19. **Caracterização do dataset: perfis básicos e histogramas de amplitude.** Depois de acrescentar `import matplotlib.pyplot as plt` na célula 0 e as constantes `BSC_DIR` e `CARAC_DIR` na célula de configuração, foram executadas as células A (resumo do BSC), B (valores ausentes) e C (histogramas de amplitude). As saídas de texto repetiram os números do teste fora do Colab e estão na seção "Caracterização do dataset". A figura do histograma e a célula D (salvar as tabelas em `data/processed/caracterizacao/`) não foram reportadas.

20. **Salvamento das tabelas e do histograma (célula D) e figura do histograma.** Na pasta `MyDrive/waid/data/processed/caracterizacao/` apareceram 4 arquivos, todos de 01/11/2026 (16:00 e 16:01): `ausentes_bsc.csv`, `histograma_amplitude.png`, `resumo_bsc.csv` e `resumo_imagens.csv`. A figura do histograma enviada é igual à do teste fora do Colab: curvas com formas diferentes por poço, com pico secundário perto de −4 dB no antilope25 e no botorosa47.

21. **Inspeção visual dos 5 arquivos `visao_geral.png`.** Análise por poço com três critérios (manchas e bandas escuras, faixas diagonais periódicas, trechos claros sem estrutura). O resultado e a conferência numérica estão na seção "Inspeção visual".

22. **Inspeção do `trecho_01_2061.13m.png` (antilope25).** A leitura do arquivo isolado não permitiu concluir. O trecho foi regenerado com a escala do poço e com a escala do próprio trecho, e comparou com o início do botorosa47 e o fim do antilope37. O resultado está na seção "Inspeção visual".

23. **Decisão D1: excluir do treino e da avaliação os trechos de amplitude anômala nas pontas dos poços** (topo do antilope25, topo do botorosa47, fim do antilope37). Justificativa: não são o alvo do projeto, e um limiar de amplitude os marcaria como feição. Os limites exatos serão definidos por regra, a partir do perfil de amplitude média por linha. Ver "Decisões de projeto".

24. **Regra de trabalho definida e primeira consulta a fontes.** Decisões que dependem de conhecimento de domínio passam a ser apoiadas, sempre que possível, em fonte especializada citada na seção "Referências especializadas". Para a decisão D1 foram consultadas 5 fontes. Elas sustentam, em geral, que imagens acústicas têm artefatos de aquisição sem relação com a rocha. Nenhuma trata do comportamento observado nem fixa uma regra de exclusão.

25. **Tese anexada ao projeto (Fioriti, 2026) lida e avaliada.** O conteúdo ajuda e foi incorporado à decisão D1, às decisões pendentes e às referências (ref. 6). Foram conferidas duas coisas com os dados: o caliper nas zonas anômalas dos topos (coincide com o alargamento do poço) e o período das faixas diagonais (estimativa inconclusiva). Ver as seções "Inspeção visual" e "Decisões de projeto".

26. **Célula de conversão para `.npy` e perfil de amplitude média por linha.** Foi confirmada a geração dos arquivos em `MyDrive/waid/data/processed/npy/`. A figura do perfil e a linha com o tempo de leitura do `.npy` não foram reportadas.

### 02/11/2026

27. **Notebook `01_exploracao.ipynb` enviado com a execução mais recente (16 células).** Conferência das saídas:
    - A célula de conversão para `.npy` foi executada: os 5 poços carregaram, a figura do perfil apareceu e a leitura do `.npy` do coala88 levou 0,1 s (`(87796, 180) float32`). A mesma leitura pelo CSV levou 14,1 s na célula 6.
    - A célula de salvamento listou os 4 arquivos de `caracterizacao/` (a figura do perfil foi salva depois dessa listagem).
    - A contagem de PNGs mostrou **14 arquivos** em antilope25, antilope37 e botorosa47, e 13 no coala88 e no tatu22. O esperado era 13 em cada poço. O arquivo a mais em cada um dos 3 poços foi identificado (item 29).
    - A célula da máscara de exclusão (D1) ainda não está nessa versão do notebook.

28. **Modelo (template) e limite de páginas do artigo: não informados.** Essas informações não estão disponíveis no momento. Ação: perguntar à professora da disciplina. Enquanto isso, escrever o conteúdo de forma independente do modelo e gerar as figuras em PDF vetorial e PNG de 300 dpi.

29. **Explicação dos 14 arquivos nas pastas de PNG.** Foram colocadas em `png/antilope25/`, `png/antilope37/` e `png/botorosa47/` as figuras `antilope25_topo_comparacao.png`, `antilope37_fim_comparacao.png` e `botorosa47_topo_comparacao.png`, geradas fora do notebook. Orientação: mover essas figuras (e `caliper_x_amplitude_topos.png`) para `MyDrive/waid/docs/` e repetir a contagem, que deve dar 13 arquivos por poço.

30. **Notebook `01_exploracao.ipynb` enviado de novo (16 células) e contagem de PNGs repetida.** A contagem mostrou 13 arquivos em cada um dos 5 poços, o que confirma a retirada das figuras de comparação das pastas de PNG (item 29). A célula da máscara de exclusão (D1) ainda não está no notebook; houve dúvida sobre qual é essa célula, e ela foi enviada de novo.

31. **Execução da célula da máscara de exclusão (D1) no Colab.** O notebook enviado (17 células) tem a célula, com os parâmetros e o código iguais aos entregues. Resultado, igual ao teste fora do Colab: antilope25 (topo), linhas 0–1641 (2061,13–2065,29 m); antilope37 (fim), linhas 86735–89981 (2554,62–2571,11 m; 3,61% do poço); botorosa47 (topo), linhas 0–1895 (2178,00–2187,63 m; 2,34% do poço). A figura confirma as zonas em destaque.

32. **Célula de normalização e patches (D2, D3 e D4) escrita fora do Colab; não executada no Colab.** Teste fora do Colab (34 s): 3.465 patches válidos de 256 × 180 (antilope25: 849 de 856; antilope37: 676 de 701; botorosa47: 617 de 632; coala88: 684 de 684; tatu22: 639 de 639). Percentis 1 e 99 por poço, sem as zonas da D1, em dB: antilope25 21,75 e 39,54; antilope37 11,27 e 33,42; botorosa47 23,44 e 37,80; coala88 24,07 e 39,79; tatu22 17,98 e 32,46. A célula salva imagens normalizadas em `float16` (162 MB no total) e o índice de patches.

33. **Definições D2, D3 e D4 registradas.** As três decisões foram definidas e registradas como definições do projeto. Ver "Decisões de projeto". Foi iniciada a execução da célula de normalização e patches; o resultado ainda não foi reportado.

34. **Leitura completa do artigo arXiv 2603.20729 (Silva, 2026) e do trabalho de Anjos et al. (2022).** Resultados: o artigo de Silva usa o WAID, pseudo-rótulos de limiar e mapa de confiança, e mede a concordância com o pseudo-rótulo de treino; o treino é descrito por intervalo. O trabalho de Anjos et al. usa U-Net (DC-UNet), validação cruzada por poço e extensão circular das colunas. O posicionamento foi revisado no item 36. A proposta de modelos está na seção "Definição dos modelos".

35. **Documento "Estudo de ablação" anexado ao projeto.** Passou a ser a base do plano de ablação. O README resume o plano, e a versão atualizada do documento ficou em `docs/ablacao.md`. A hipótese de domínio do deslocamento circular ganhou duas fontes (Anjos et al., 2022; Fioriti, 2026).

36. **Princípio do trabalho e reavaliação do posicionamento.** Ficou estabelecido que o trabalho não precisa gerar algo inédito, mas deve basear as conclusões em experimentos e referências. A avaliação foi refeita: saíram a linguagem de "diferencial" e de "reivindicar"; entraram hipóteses testáveis com fonte, experimento e ablação (H1 a H10), uma regra de leitura dos resultados (proposta), os valores publicados como contexto e a prioridade menor das variantes de arquitetura. Ver "Trabalhos relacionados e posicionamento".

37. **Notebook `01_exploracao.ipynb` enviado com a execução mais recente (18 células).** A célula de normalização e patches foi executada no Colab, com o código igual ao entregue. Resultado, igual ao teste fora do Colab: 3.465 patches válidos (antilope25: 849 de 856; antilope37: 676 de 701; botorosa47: 617 de 632; coala88: 684 de 684; tatu22: 639 de 639) e a figura com um patch por poço, em escala visual parecida. A máscara da D1 e a conversão `.npy` seguem como nos itens 31 e 26. O trabalho foi pausado por outra tarefa; a retomada parte das conclusões deste notebook e dos itens ainda não confirmados. Ver "Estado do projeto e pendências".

### 14/11/2026

38. **Definição dos modelos registrada (D5).** Ficaram definidos os quatro pontos propostos: U-Net com encoder ResNet18 como modelo principal; baselines com limiares e CNN rasa; ResNet34 e U-Net++ fora por enquanto; justificativa do ResNet18 (leve para o Colab, conjunto pequeno, pré-treino a testar na ablação H7). A Fase 1 passou a concluída. Item 1 da tabela B resolvido. Próximo: item 2 (regra de leitura das ablações).

### 15/11/2026

39. **Estrutura hierárquica no `01_exploracao.ipynb`.** Foram criadas células-modelo (título e seção) e o notebook foi executado. O modelo foi aplicado: título real do trabalho na primeira célula, cabeçalho da Fase 1 e 9 seções numeradas (1.1 ambiente; 1.2 download; 1.3 configuração; 1.4 carga; 1.5 PNG; 1.6 caracterização; 1.7 `.npy`; 1.8 D1; 1.9 D2–D4). O notebook passou a ter 29 células (11 de texto e 18 de código); o código e as saídas não mudaram. Convenção registrada na seção "Cronograma".

40. **Regra de leitura das ablações definida (item 2).** Ficou definido o uso de 3 sementes por configuração e o uso da regra proposta: efeito detectado quando a diferença média passa de 1 desvio-padrão entre sementes e tem o mesmo sinal em pelo menos 4 dos 5 poços. Fora disso, o artigo escreve "sem efeito detectável" e traz o valor numérico. O artigo declara as limitações (desvio-padrão fraco com 3 sementes; critério prático, não teste de significância; poços não independentes). Próximo: item 3 (hipóteses H1 a H10).

41. **Hipóteses e prioridades definidas (item 3).** Foram definidas as hipóteses H1 a H10 e as prioridades. Ordem de execução: ablações H3, H4 e H5 primeiro; depois, se houver tempo, H10, H8, H6 e H9; H7 e a linha 5 têm prioridade baixa. H1 e H2 vêm do experimento principal. A lista final fecha depois de medir o tempo de um treino. O documento foi salvo em `docs/ablacao.md`. A versão atualizada substituiu a anterior no projeto de apoio. Próximo: item 4 (limite do fim do antilope37).

42. **Limite do fim do antilope37 definido (item 4).** Ficou definida a opção C: manter `FOLGA_DB = 10` dB, que é o valor já executado; registrar no artigo que o limite é provisório e que a causa física é desconhecida; fazer um teste de sensibilidade (6 e 14 dB) só se sobrar tempo, junto com a ablação H5 (com e sem D1).

43. **Plano do conjunto anotado definido (item 7).** Ficou definido o plano de quatro camadas de evidência: (1) ficha de critérios escritos, extraídos das referências especializadas, com a fonte de cada critério; (2) anotação assistida (substituída no item 49 pela decisão por patch), só casos claros, 30 a 50 patches; (3) evidência independente de *breakout* pelo caliper (texto corrigido no item 49: o CAL é curva única e não mostra o par a 180°); (4) teste sintético com senoides de mergulho e azimute conhecidos inseridas em trechos reais. O artigo declara que as anotações seguem critérios da literatura, sem validação por especialista, e que as métricas medem a concordância com elas; a concordância com os pseudo-rótulos de limiar entra como medida secundária. Próximo: item 5 (cronograma).

44. **Cronograma recalculado e definido (item 5).** Fase 1 concluída em 14/11. Fase 2: 16 a 20/11. Fase 3: 21 a 27/11 (tempo de um treino medido no primeiro dia). Fase 4: 28/11 a 01/12. Entrega: 02/12. A escrita das seções de dados, método e trabalhos relacionados começa em paralelo em 18/11. Se o tempo de treino no Colab não permitir 15 treinos por configuração, o número de configurações diminui; o número de sementes (3) não muda. Itens 1 a 5 da tabela B resolvidos. Próximo: ficha de critérios do conjunto anotado (início da Fase 2).

45. **Ficha de critérios de anotação escrita (rascunho).** Foram extraídos da tese de Fioriti (2026) os critérios de *breakout* (pares a 180°, alongados na vertical; p. 62 e 165), fratura natural (senoide; p. 52), cavidades (p. 112 e 116) e artefatos (tabela 38, p. 193). Arquivo `docs/anotacao.md` (vai para `docs/`). Classes iniciais: fundo, feição escura e dúvida (o patch sai). Aguardava leitura e definição (ver item 47). A ficha liga as faixas diagonais do coala88 e do tatu22 ao espiralamento ou ao *reaming*, como hipótese a confirmar (item 10 da tabela B). Lofts e Bourke (1999) segue não obtido.

46. **Declaração de uso de IA definida.** Para a seção de método do artigo, ficou definida a frase: "Assistente de IA usado para apoio ao código e à revisão do texto. Decisões, experimentos e conclusões são do autor." O texto também consta na seção "Uso de ferramentas de IA". Não houve definição de regra da disciplina sobre declaração de IA.

47. **Ficha de critérios definida.** Os critérios e as classes da ficha foram adotados como estão. O arquivo foi salvo como `docs/anotacao.md`, ao lado de `docs/ablacao.md`. Risco registrado: a anotação, feita sem especialista, tem limites de confiabilidade. O plano de quatro camadas e a declaração desses limites no artigo tratam desse risco.

48. **Ferramenta de anotação assistida e `02_baselines.ipynb` escritos.** Pacote com `src/config.py` (poços e pastas compartilhados, para evitar repetição de definições entre notebooks), `src/labels/anotacao.py` (limiar de Otsu por poço, sorteio, regiões escuras com junção circular nas bordas, caixas, interface com `ipywidgets`, tabela `anotacoes.csv`) e `notebooks/02_baselines.ipynb` (seções 2.1 a 2.5). Teste com dados sintéticos fora do Colab: sorteio sem sobreposição, junção de regiões na borda, caixas que cruzam a borda, validação de entradas, gravação e substituição de decisões. **Não executado no Colab com dados reais.** O `01_exploracao.ipynb` mantém suas definições próprias de pastas; pode adotar `src/config.py` depois.

49. **Plano de anotação simplificado (combinação A + B).** Diante do desconforto com a anotação sem especialista, ficou definida a combinação: decisão por patch (feição escura clara, sem feição ou dúvida), curva CAL como apoio e teste sintético na Fase 3. A ferramenta foi reescrita: saíram as regiões do limiar, as caixas e as máscaras; entrou o resumo do CAL (`cal_max`, `cal_z`). O `02_baselines.ipynb` perdeu o cálculo do limiar de Otsu, que volta na seção dos *baselines*. Duas correções: (1) o CAL do WAID é uma curva única por profundidade; ele mostra alargamento do poço, mas não o azimute nem o par de zonas a 180°, então não confirma *breakout* sozinho (o texto do item 43 estava impreciso); (2) sem máscaras no conjunto anotado, IoU e Dice só valem no teste sintético. A métrica no conjunto anotado passa a ser F1 e AUC por patch. Nova pendência: item 17 da tabela B (métrica principal das ablações). A reavaliação do plano fica aberta se o desconforto continuar.

### 15/11/2026 (noite)

50. **Decisão por patch executada no Colab.** O erro anterior (versão antiga do módulo `anotacao.py` no Colab; o quinto argumento era tratado como `limiares`) não se repetiu na execução seguinte. Resultado dos 40 patches: 27 feição escura clara, 6 sem feição e 7 dúvida (antilope25: 5/2/1; antilope37: 4/2/2; botorosa47: 5/2/1; coala88: 5/0/3; tatu22: 8/0/0). Observações: (1) 33 patches usáveis, com só 6 negativos; a AUC por patch será grossa e precisa de intervalo de confiança; (2) coala88 e tatu22 não têm negativos, então as métricas por poço não existem nesses dois; (3) 27 de 40 patches com feição (67%) pede cuidado: a classe agrupa fratura, cavidade e *breakout*, e a ficha não separa feição de artefato em todos os casos; (4) o `cal_z` não separa as classes (medianas 0,86; −0,46; 0,04; n pequeno). Próximo: conferir as profundidades dos patches "feição clara" contra as zonas de faixas diagonais (item 10 da tabela B) e decidir o item 17.

51. **Segunda rodada de anotação preparada.** Para reduzir a incerteza da AUC (só 6 negativos) e dar métrica por poço ao coala88 e ao tatu22, foi escrito o sorteio dirigido: `limiar_otsu_poco`, `fracao_escura` e `sortear_segunda_rodada` em `src/labels/anotacao.py`; seções 2.6 a 2.8 no `02_baselines.ipynb`. A tabela `anotacoes.csv` ganhou a coluna `rodada` (linhas antigas valem 1). Teste com dados sintéticos: sem sobreposição entre os patches novos e os da rodada 1, e sem sobreposição entre os novos. Não executado no Colab. Ficou combinado manter a verificação das profundidades dos patches "feição clara" contra as zonas de faixas diagonais (item 10), usando o `anotacoes.csv`, em paralelo à rodada 2.

### 16/11/2026

52. **Segunda rodada executada e `anotacoes.csv` analisado.** Total de 80 patches: 57 feição clara, 16 sem feição e 7 dúvida (rodada 2: 30 feição e 10 sem feição). Achados da análise do arquivo: (1) o tatu22 teve 16 de 16 "feição clara", inclusive os 8 da rodada 2, tirados do quartil de menor área escura; o coala88 teve 6 de 8; isso pode ser real, mas também pode indicar viés de resposta ("feição" como padrão) ou que a fração escura de Otsu não representa o que a imagem mostra; (2) o tempo mediano entre decisões foi de cerca de 8 s e 40 de 79 intervalos ficaram abaixo de 10 s (mediana no tatu22: 5,5 a 6,5 s), o que torna a reanotação às cegas de uma amostra mais importante; (3) três patches de "feição clara" foram sinalizados pela ficha: tatu22 2386,18–2387,47 m (dentro da zona de faixas diagonais, 2382–2424 m, ainda sem causa confirmada), antilope37 2123,11–2124,40 m (`cal_z` = 12,8: poço muito alargado; a ficha trata zona escura extensa com caliper alargado como *washout*, artefato) e coala88 2338,80–2340,09 m (borda da zona de faixas diagonais, 2340–2370 m); (4) o patch antilope25 2291,96–2293,26 m cai em intervalo de amplitude baixa ainda não analisado (item 12), sem regra na ficha. Os 3 patches sinalizados aguardam decisão: manter ou passar para "dúvida" (a mudança fica registrada na coluna de observação). Os valores de `cal_z` altos em "feição clara" (antilope25 3,2 e botorosa47 3,0) ficam como estão, pois *breakout* é alvo do projeto.

53. **Decisões sobre o item 52.** (1) Os patches tatu22 2386,18–2387,47 m e antilope37 2123,11–2124,40 m passam de "feição clara" para "dúvida", pelos critérios da ficha; o patch coala88 2338,80–2340,09 m fica como está (borda da zona de faixas diagonais). Totais após a revisão: 55 feição clara, 16 sem feição e 9 dúvida (71 usáveis). A função `revisar_decisao` guarda a decisão original (coluna `decisao_original`) e o motivo (coluna `observacao`). (2) Foi definida a reanotação às cegas de 10 patches (4 do tatu22; 6 de outros poços, metade "sem feição"), com o menor intervalo de tempo possível. A interface não mostra a decisão anterior. O resultado vai para `reanotacao.csv`. A concordância e o kappa de Cohen saem da função `concordancia`. Com 10 patches, o kappa é só um indicador grosso, e o artigo declara o intervalo de tempo curto entre as duas rodadas. Pacote 08: `src/labels/anotacao.py` e `02_baselines.ipynb` (seções 2.9 a 2.11). Teste com o `anotacoes.csv` real em cópia fora do Drive: revisão, repetição sem sobrescrever o motivo, sorteio da reanotação (4 do tatu22, 3 "sem feição" e 3 "feição clara" nos outros) e concordância. Não executado no Colab.

54. **Reanotação às cegas executada (seções 2.9 a 2.11).** Resultado: 10 patches comparados, concordância 1,00 e kappa de Cohen 1,00 (7 "feição clara" e 3 "sem feição", todos iguais à decisão original). Foi informado que nenhuma decisão anterior foi lembrada durante a reanotação. O intervalo entre as rodadas foi de mais de 10 horas (decisões originais entre 22:28 e 00:10, horário local; reanotação depois das 10:25 do dia seguinte; o Colab registra a hora em UTC, 3 horas à frente). Leitura: indica consistência da leitura em casos que o autor decide com clareza. Limites: n = 10 (o intervalo de confiança do kappa é largo); os 4 patches do tatu22 eram todos "feição clara", então não testam a decisão "sem feição" nesse poço; consistência do mesmo anotador não prova validade (um erro sistemático se repetiria); a amostra não inclui patches de "dúvida". A anotação está fechada. A revisão pela ficha (item 53) foi aplicada: totais 55 / 16 / 9. Próximo: *baselines* de limiar.

55. **Código dos *baselines* de limiar escrito.** Módulos novos: `src/labels/limiares.py` (limiar por poço com Otsu, Multi-Otsu de 3 classes e K-means de 2 grupos, sobre 1 a cada 10 linhas válidas; `limiar_otsu_poco` e `fracao_escura` passaram para cá, e `anotacao.py` os reexporta) e `src/eval/patches.py` (escore por patch = fração de pixels abaixo do limiar; AUC com intervalo de confiança por *bootstrap*; F1, precisão e revocação com o corte escolhido nos outros poços; AUC por poço onde há as duas classes). Constante nova `RES_DIR` em `src/config.py` (`results/`). Seção 2.12 do `02_baselines.ipynb`. Saídas previstas: `limiares_baselines.csv`, `baselines_limiar_escores.csv` e `baselines_limiar_patches.csv`. Teste com dados sintéticos fora do Colab (3 poços, patches com e sem faixa escura): o pipeline roda e dá AUC 1,00, como esperado em dados triviais. Primeira execução no Colab (16/11/2026) falhou em `avaliar`: o cálculo da AUC por poço usava `n_boot=1`, e o percentil de uma lista vazia gerou `IndexError`. Corrigido (`n_boot=0` dá só a AUC; lista vazia dá NaN). Reexecutar a seção 2.12. Os limiares são calculados sem treino; só o corte do F1 usa as anotações, e em *leave-one-well-out*.

56. **Código da CNN rasa e da U-Net escrito.** Módulos novos: `src/models/modelos.py` (CNN rasa com 4 camadas convolucionais dilatadas, campo receptivo de 17 × 17 pixels, e U-Net com encoder ResNet18 pela biblioteca `segmentation-models-pytorch`), `src/data/dataset_patches.py` (pseudo-rótulo = pixel abaixo do limiar de Otsu do poço; deslocamento circular em azimute como aumento de dados, H4; 6 colunas circulares de cada lado, de 180 para 192 colunas; exclusão dos patches que se sobrepõem aos anotados) e `src/train/treino.py` (perda Dice mais BCE; Adam com 1 época de aquecimento e decaimento por cosseno; parada antecipada pelo Dice no poço de validação; *leave-one-well-out* com poço de validação = próximo da lista; cada *fold* é salvo e o laço retoma de onde parou). `avaliar_modelos` em `src/eval/patches.py` calcula AUC e F1 por patch, por semente e em média. Notebook `03_cnn_unet.ipynb` (seções 3.1 a 3.6), com um teste rápido de tempo (3.4) para fechar a lista de ablações. Teste fora do Colab, em CPU e com dados sintéticos: o laço roda para os dois modelos e a CNN rasa aprende o limiar (Dice de validação de cerca de 0,90 em 15 épocas). **Não executado no Colab com dados reais.** Pendência: com o resultado da seção 2.12, decidir o item 17 (métrica das ablações).

57. **Resultado dos *baselines* de limiar (seção 2.12, executada em 16/11/2026).** Avaliação em 71 patches anotados (55 feição clara, 16 sem feição; 9 em dúvida ficaram fora). Escore do patch = fração de pixels escuros; F1 com corte escolhido nos outros poços.

| Método | AUC | IC 95% | F1 (LOWO) | Precisão | Revocação |
|---|---|---|---|---|---|
| Otsu | 0,932 | 0,856–0,988 | 0,920 | 0,897 | 0,945 |
| Multi-Otsu (3 classes) | 0,911 | 0,818–0,982 | 0,939 | 0,900 | 0,982 |
| K-means (2 grupos) | 0,932 | 0,857–0,987 | 0,919 | 0,911 | 0,927 |

AUC por poço: antilope25 0,926; antilope37 0,975; botorosa47 0,972; coala88 1,000; tatu22 não definida (só há a classe "feição clara"). Leituras: (1) os três métodos são estatisticamente indistinguíveis, pois os intervalos se sobrepõem quase por inteiro; (2) o teto para os modelos é alto, e com 71 patches o intervalo tem cerca de 0,13 de largura, o que limita a detecção de diferenças pequenas entre modelos e entre ablações (relevante para o item 17); (3) a anotação por patch e a fração escura concordam muito, o que é esperado pela ficha, mas não prova que o escore mede feição geológica; (4) o AUC do tatu22 e o do coala88 (só 2 negativos) não são conclusivos. Saídas: `limiares_baselines.csv`, `baselines_limiar_escores.csv`, `baselines_limiar_patches.csv` em `results/`.

58. **Item 17 resolvido (métrica das ablações).** Foi adotada a proposta: a regra de leitura das ablações (item 2 da tabela B) usa o Dice contra os pseudo-rótulos no poço de teste e o IoU do teste sintético; o conjunto anotado (F1 e AUC por patch) serve de verificação externa. O pacote 10 (CNN rasa e U-Net) ainda não foi atualizado nem executado no Colab até este registro.

59. **Teste de tempo executado (seção 3.4 do `03_cnn_unet.ipynb`, Colab com GPU, 16/11/2026).** Um *fold* (teste no coala88, validação no tatu22), 2 épocas, avaliação a cada época. CNN rasa: 7,0 s por época e 20,1 s no *fold* completo. U-Net com ResNet18 e pré-treino ImageNet: 11,6 s por época e 30,3 s no *fold* completo. Estimativa para 10 épocas, 5 *folds* e 1 semente: 5,8 min (CNN rasa) e 9,6 min (U-Net). Com 3 sementes: cerca de 17 min e 29 min. Uma configuração de ablação com a U-Net (15 treinos) custa cerca de 29 min, sem contar a parada antecipada, que pode reduzir o tempo. As ablações H3, H4, H5, H10, H8, H6 e H9 (7 configurações) cabem em cerca de 3,5 h de GPU, dentro da Fase 3 (21 a 27/11). Os avisos sobre `HF_TOKEN` no *download* dos pesos ImageNet são inofensivos.

60. **Primeira execução do treino LOWO (seções 3.5 e 3.6; Colab com GPU; 16/11/2026; semente 0 apenas).** Resultado preliminar no conjunto anotado (71 patches; escore = probabilidade média do modelo no patch; todos os *folds* juntos). CNN rasa: AUC 0,807 (IC 95% 0,671–0,916), F1 0,898, precisão 0,841, revocação 0,964. U-Net com ResNet18: AUC 0,800 (IC 95% 0,677–0,903), F1 0,814, precisão 0,793, revocação 0,836. Para comparação, o limiar de Otsu deu AUC 0,932 (IC 95% 0,856–0,988) e F1 0,920 (item 57). Leituras: (1) com 1 semente não há desvio-padrão (NaN na tabela); (2) os intervalos dos modelos e dos limiares se sobrepõem em parte, então a diferença de AUC ainda não é conclusiva; (3) a CNN rasa e a U-Net empatam em AUC; (4) os modelos aprendem a imitar o limiar de Otsu, e por isso não se espera que o superem no conjunto anotado; o artigo relata o resultado como ele sair, inclusive se os modelos não superarem o limiar. A execução com sementes 0, 1 e 2 foi iniciada em 16/11/2026; os resultados entram em item próprio.

61. **Resultado do treino LOWO com 3 sementes (seções 3.5 e 3.6; Colab com GPU; 16/11/2026).** Conjunto anotado (71 patches; escore = probabilidade média; média e desvio-padrão entre as sementes 0, 1 e 2).

| Modelo | AUC | F1 (LOWO) | Precisão | Revocação |
|---|---|---|---|---|
| CNN rasa | 0,794 ± 0,014 | 0,839 ± 0,052 | 0,808 ± 0,029 | 0,873 ± 0,080 |
| U-Net com ResNet18 | 0,811 ± 0,010 | 0,845 ± 0,041 | 0,814 ± 0,019 | 0,879 ± 0,074 |
| Otsu (item 57) | 0,932 | 0,920 | 0,897 | 0,945 |

Leituras: (1) a diferença de AUC entre a U-Net e a CNN rasa (0,017) tem a ordem de 1 a 1,5 desvio-padrão; pela regra de leitura, fica como "sem efeito detectável"; (2) os dois modelos ficaram cerca de 0,12 a 0,14 de AUC abaixo do Otsu; essa diferença é muito maior que o desvio entre sementes, mas os intervalos de confiança de 95% do conjunto anotado ainda se sobrepõem em parte (cerca de 0,68–0,92 contra 0,86–0,99), por isso foi escrita a comparação pareada (`comparar_pareado`, seção 3.7); (3) o F1 tem desvio de 0,04 a 0,05 entre sementes, então não separa os modelos; (4) os modelos treinam para imitar o limiar de Otsu, e o resultado não mostra ganho sobre ele no conjunto anotado. O artigo relata isso como resultado. Pendências para a seção 3.7: diferença pareada de AUC contra o Otsu e comparação entre `score_soft` e `score`. Pacote 12: `src/eval/patches.py` (função `comparar_pareado`) e `03_cnn_unet.ipynb` (seção 3.7).

62. **Comparação pareada com o Otsu (seção 3.7, 16/11/2026) — somente semente 0.** Após reiniciar a sessão, a célula 3.2 voltou a `SEMENTES = [0]` e a 3.5 carregou só os *folds* da semente 0; por isso a tabela saiu sem desvio-padrão. Resultado (71 patches, semente 0): CNN rasa, diferença de AUC contra o Otsu de −0,125 (IC 95% pareado −0,247 a −0,030); U-Net, −0,132 (−0,219 a −0,062). Os dois intervalos não incluem zero: nos mesmos patches, os modelos ficam abaixo do Otsu de forma detectável. Escore por fração de pixels preditos (`score`) contra probabilidade média (`score_soft`): AUC de 0,809 e 0,807 (CNN rasa) e de 0,806 e 0,800 (U-Net), ou seja, sem diferença relevante; o F1 variou mais (CNN rasa: 0,839 contra 0,898), o que confirma que o F1 com 71 patches é ruidoso. O notebook passou a definir `SEMENTES = [0, 1, 2]` na célula 3.2. Falta repetir a 3.7 com as 3 sementes.

63. **Resultado final da Fase 2 (seção 3.7, 3 sementes, 16/11/2026).** Comparação pareada de AUC contra o Otsu nos mesmos 71 patches (IC 95% por *bootstrap* pareado): CNN rasa, −0,152 (−0,264 a −0,069); U-Net, −0,120 (−0,210 a −0,048). Os dois intervalos não incluem zero: os modelos ficam abaixo do Otsu de forma detectável. Escore por fração de pixels preditos (`score`) contra probabilidade média (`score_soft`): AUC de 0,816 ± 0,008 e 0,794 ± 0,014 (CNN rasa) e de 0,819 ± 0,011 e 0,811 ± 0,010 (U-Net); o F1 de `score` foi de 0,827 ± 0,010 (CNN rasa) e 0,850 ± 0,051 (U-Net). A escolha do escore não muda a conclusão. A U-Net e a CNN rasa não se separam. Ficou definido manter `score_soft` como escore principal e registrar `score` como alternativa. A Fase 2 foi dada como concluída. A consulta à professora sobre o *template* e o limite de páginas (item 6 da tabela B) foi enviada por e-mail em 16/11/2026; falta a resposta.

64. **Código do teste sintético escrito (Fase 3 adiantada).** O Dice contra o pseudo-rótulo muda de significado quando uma ablação altera a normalização ou o próprio pseudo-rótulo (H3, H9); por isso o IoU sintético, com máscara exata, passa a ser a medida comparável entre configurações (relacionado ao item 17). Módulo novo `src/synth/sintetico.py`: para cada poço, 24 fundos reais do quartil de menor fração escura recebem 1 ou 2 senoides (1 volta completa, altura de 10 a 60 linhas, espessura de 4 a 8 linhas) em 2 contrastes (valor normalizado 0,05 e 0,20 dentro da senoide); sorteio com semente fixa (123), igual para todos os modelos. Os parâmetros são escolhas deste trabalho, em faixas plausíveis, sem validação por especialista, e o artigo declara isso (forma de senoide e altura ligada ao mergulho: Fioriti, 2026, p. 52). `executar_fold` passou a medir IoU e Dice sintéticos no poço de teste (colunas `iou_sint`, `dice_sint` e por contraste). `preparar_dados` (em `src/data/dataset_patches.py`) reúne o preparo dos dados e evita repetir o código do notebook 03. Notebook `04_sintetico.ipynb` (seções 4.1 a 4.7): exemplos, *baseline* de Otsu no sintético, repetição do treino de referência (cerca de 46 min, em `results/modelos_sint/`, sem sobrescrever o notebook 03) e resumo. Teste fora do Colab, em CPU e com dados sintéticos: o código roda de ponta a ponta. Limite conhecido: o IoU do Otsu no sintético é o teto de referência, pois os modelos imitam o limiar. **Não executado no Colab com dados reais.** O `03_cnn_unet.ipynb` mantém seu código de preparo próprio (versão já executada).

65. **Teste sintético: exemplos e *baseline* de Otsu executados (seções 4.4 e 4.5; Colab; 16/11/2026) e ajuste do desenho.** A figura mostrou senoides coerentes (1 volta completa, 1 ou 2 por patch; 48 patches por poço; 3,3% dos pixels). Resultado do Otsu (primeira versão, contrastes 0,05 e 0,20): IoU de 0,957 (antilope25), 0,919 (antilope37), 0,718 (botorosa47), 0,869 (coala88) e 0,818 (tatu22). Duas falhas de desenho apareceram: (1) o IoU dos dois contrastes saiu idêntico em todos os poços, pois os dois valores ficam abaixo do limiar de Otsu e o contraste não discriminava nada; (2) o teto variou de 0,72 a 0,96 entre poços porque os fundos ainda têm pixels escuros fora da senoide, que contavam como erro. Ajustes antes de rodar o treino da seção 4.6: o valor dentro da senoide passou a ser `fator × limiar de Otsu do poço`, com fatores 0,25, 0,75 e 0,95 (0,95 fica quase no limiar); e os pixels que já eram escuros no fundo, fora da senoide, são ignorados na previsão (`ignorar`; marcar uma feição que já existia não é erro). Colunas novas: `iou_sint_25`, `iou_sint_75`, `iou_sint_95` (e `dice_...`). Teste fora do Colab, em CPU: com o limiar a 0,35, o Otsu dá IoU 1,00 nos fatores 0,25 e 0,75 e 0,81 em 0,95. Pacote 15. Falta repetir as seções 4.3 a 4.5 e executar a 4.6.

66. **Teste sintético corrigido e validado (seções 4.4 e 4.5; Colab; 16/11/2026).** Com os contrastes relativos ao limiar e a região ignorada, o Otsu deu IoU de 1,000 nos fatores 0,25 e 0,75 em todos os poços e, no fator 0,95, de 0,913 (antilope25), 0,940 (antilope37), 0,952 (botorosa47), 0,955 (coala88) e 0,971 (tatu22). IoU total: 0,971 a 0,990; Dice total: 0,985 a 0,995. O teto agora é estável entre poços (0,97 a 0,99) e o fator 0,95 discrimina. O conjunto tem 72 patches sintéticos por poço (24 fundos × 3 contrastes); 3,9% dos pixels são senoide. Limite registrado: o ruído do valor dentro da senoide (desvio de 0,02) e os fatores foram escolhidos neste trabalho, sem validação por especialista. Próximo passo: executar a seção 4.6 (treino de referência com a métrica sintética, cerca de 46 min).

67. **Treino de referência com métrica sintética (seções 4.6 e 4.7; Colab com GPU; 16/11/2026).** CNN rasa e U-Net com 3 sementes e 5 poços de teste (30 folds; 43 min no total: cerca de 58 s por fold da CNN rasa e 115 s por fold da U-Net). Média entre os poços de teste, depois média ± desvio-padrão entre as sementes:

| Métrica | CNN rasa | U-Net | Otsu |
|---|---|---|---|
| IoU sintético (total) | 0,773 ± 0,021 | 0,771 ± 0,021 | 0,982 |
| Dice sintético (total) | 0,825 ± 0,021 | 0,824 ± 0,023 | 0,991 |
| IoU, fator 0,25 (contraste forte) | 0,911 ± 0,013 | 0,909 ± 0,030 | 1,000 |
| IoU, fator 0,75 | 0,905 ± 0,029 | 0,887 ± 0,031 | 1,000 |
| IoU, fator 0,95 (quase no limiar) | 0,503 ± 0,061 | 0,518 ± 0,060 | 0,946 |
| Dice contra o pseudo-rótulo (poço de teste) | 0,680 ± 0,025 | 0,625 ± 0,011 | – |

Leituras: (1) a CNN rasa e a U-Net empatam no IoU sintético (diferença de 0,002, muito menor que o desvio de 0,021); (2) os dois modelos ficam abaixo do Otsu em todos os contrastes: cerca de 0,09 a 0,11 de IoU nos fatores 0,25 e 0,75 e cerca de 0,43 a 0,44 no fator 0,95; o Otsu é o teto, pois os modelos imitam o limiar; (3) no fator 0,95, o valor da senoide fica quase no limiar, e um modelo que aprende uma versão suave do limiar acerta cerca de metade dos pixels; esse nível é o mais sensível, mas também o mais ruidoso (desvio de 0,06); (4) o Dice contra o pseudo-rótulo separa os modelos (0,680 contra 0,625, diferença maior que 1 desvio), ao contrário do IoU sintético; a regra completa (sinal em 4 dos 5 poços) ainda não foi checada; (5) o Dice contra o pseudo-rótulo varia muito entre poços (cerca de 0,41 a 0,55 no tatu22 e no antilope25 contra 0,69 a 0,86 no botorosa47 e no antilope37), sem causa investigada; (6) para a leitura das ablações, o desvio entre sementes do IoU sintético total (0,021) é o tamanho do efeito mínimo detectável. O aviso `lr_scheduler.step() before optimizer.step()` apareceu uma vez, na primeira época da U-Net com precisão mista (o `GradScaler` pode pular o primeiro passo); é inofensivo. Saídas: `results/modelos_sint/` e `results/modelos_sint_resumos.csv`.

68. **Código das ablações H4 e H5 escrito (notebook `05_ablacoes.ipynb`).** Módulo novo `src/eval/ablacao.py`: `ler_ablacao` e `tabela_ablacao` aplicam a regra de leitura (|média da diferença| > 1 desvio-padrão entre sementes e mesmo sinal em pelo menos 4 dos 5 poços), pareando por modelo, semente e poço de teste; a métrica principal é `iou_sint` (item 17), e as de apoio são `dice_sint`, os IoU por fator e `dice_pseudo_teste`. `executar_fold` e `rodar_lowo` ganharam o parâmetro `indice_avaliacao`, para que ablações que mudam só o treino mantenham fixos os patches de avaliação. H4: `deslocar=False`. H5: `indice_sem_exclusao` inclui no treino e na validação os patches das zonas da D1; o limiar de Otsu de cada poço e a avaliação não mudam. As ablações usam só a U-Net (modelo principal) e a referência é a U-Net do item 67 (15 folds; cerca de 30 min por configuração). Teste fora do Colab, em CPU e com dados sintéticos: a regra aponta "efeito" para um deslocamento forte e "sem efeito detectável" para nenhum. H3 (normalização global ou por intervalo) ainda não tem código. **Não executado no Colab.**

69. **Registro de incertezas e rascunho da seção de limitações criados (17/11/2026).** Motivo: um professor do programa observou que trabalhos feitos com apoio de IA tendem a ter afirmações demais e dúvidas de menos. Foram revisadas as afirmações do trabalho e listadas 21 incertezas (U1 a U21), cada uma com evidência, confiança e o que a reduziria (seção "Registro de incertezas"). O rascunho da seção de limitações está em `docs/limitacoes.md`. Ficou definida uma convenção de linguagem: "foi observado" para o medido, "não foi detectada diferença" para ausência de efeito, "sugere" para hipótese e nenhuma afirmação geológica sem fonte. Duas propostas novas aguardam decisão na tabela B: item 18 (avaliação com critério independente de "escuro") e item 19 (sensibilidade aos hiperparâmetros). Principais pontos de atenção: a avaliação é parcialmente circular (U9), a exclusão dos patches em dúvida pode inflar as métricas (U11) e a ausência de diferença entre os modelos tem pouco poder estatístico (U14).

70. **Vídeos das Fases 1 e 2 revisados (17/11/2026).** Após o registro de incertezas, seis cenas foram corrigidas para não afirmar mais que a evidência: o tamanho do pixel passou a "estimativa" (U1), a causa das zonas da D1 passou a "provavelmente artefato" e a regra a "provisória", e os resumos listam D1 e D3 como escolhas pendentes de teste. Na Fase 2, as cenas informam a exclusão dos 9 patches em dúvida, o pouco poder do teste, a circularidade parcial e que os modelos ficam abaixo do Otsu apenas "nesta configuração (10 épocas)". Arquivos: `video_fase1_v2.mp4`, `video_fase2_v2.mp4` e os roteiros `_v2`.

71. **Código da ablação H3 escrito (17/11/2026).** Módulo novo `src/data/normalizacao.py`: `carregar_brutas` (amplitude em dB com a D2 aplicada e máscara da D1), `normalizar` (modos `poco`, `global` e `intervalo`) e `limiares_otsu`. Na normalização global, os percentis 1 e 99 vêm dos 5 poços juntos. Na normalização por intervalo, vêm de blocos de 2.000 linhas (cerca de 10 m); o tamanho do bloco é escolha do autor, sem apoio de especialista, e blocos com menos de 200 linhas válidas usam os percentis do poço. O limiar de Otsu de cada poço é recalculado sobre a nova imagem, e o pseudo-rótulo muda junto; por isso a métrica principal é o IoU sintético. Notebook `05_ablacoes.ipynb` ganhou as seções 5.6 (global) e 5.7 (por intervalo); a leitura passou a 5.9, depois da seção 5.8 (conferência da fração de pixels escuros em cada normalização, adicionada em 17/11/2026). A célula 5.6 confere que o modo `poco` reproduz as imagens da referência. Teste fora do Colab, só com dados sintéticos: os três modos geram imagens em [0, 1] e limiares coerentes. **Não executado no Colab.** Limites: o salto de escala na borda dos blocos não foi estudado; os fundos do teste sintético podem mudar de patch quando a normalização muda, porque a escolha usa a fração escura.

72. **Modelo de artigo recebido (17/11/2026): SBC.** Arquivo `modelosparapublicaodeartigos.zip`, com `sbc_template.doc/.odt/.pdf` e `template-latex/` (`sbc-template.tex`, `sbc-template.sty`, `sbc.bst`). Regras do modelo: A4, coluna única, margens de 3,5 cm (topo), 2,5 cm (base) e 3,0 cm (laterais); fonte Times 12 pt, 6 pt antes de cada parágrafo; sem cabeçalho, rodapé nem numeração de páginas; título em 16 pt negrito centralizado; títulos de seção em 13 pt negrito, alinhados à esquerda; legendas em Helvetica 10 pt negrito; tabelas sem fundo colorido, legenda antes da tabela; figuras em preto e branco ou tons de cinza, exceto em versão eletrônica; referências no formato autor-ano (`sbc.bst`). Artigo em português exige *resumo* e *abstract* (até 10 linhas cada, na primeira página); em inglês, só *abstract*. O modelo LaTeX compila no ambiente de trabalho (4 páginas). Consequência: figuras coloridas (mapas de pseudo-rótulo, curvas) precisam de versão em cinza ou de escolha de cores legíveis em cinza, se o artigo for impresso. O limite de páginas continua desconhecido (item 6).

73. **Revisão de linguagem (17/11/2026): voz ativa do autor nas decisões.** Para deixar claro, as decisões do autor foram garantidas em toda a documentação expressões como "definida", "definido" ou "adotada": README (12 ocorrências, entre elas os itens 33, 41, 43, 44, 45, 47, 53 e 58), `src/eval/ablacao.py` (docstring), `docs/ablacao.md`, `docs/anotacao.md`. A convenção de linguagem passou a incluir essa regra. Também foi gerada nesta data: esqueleto do artigo em LaTeX (modelo SBC, português; pasta `artigo/`).

74. **Ablações H4, H5 e H3 executadas (notebook `05_ablacoes.ipynb`, Colab Pro, 17/11/2026).** Conferência da H3: a normalização por poço do código novo reproduz a referência (diferença máxima 0,0 nos 5 poços). Métrica principal: `iou_sint` da U-Net, 3 sementes, 15 *folds* por configuração, pela regra de leitura (|média| > 1 desvio entre sementes e mesmo sinal em ≥ 4 de 5 poços).

    | Ablação | IoU sintético (ref → cand.) | Diferença | Desvio entre sementes | Poços com mesmo sinal | Leitura |
    |---|---|---|---|---|---|
    | H4 sem deslocamento circular | 0,771 → 0,763 | −0,008 | 0,043 | 3/5 | sem efeito detectável |
    | H5 sem exclusão da D1 | 0,771 → 0,763 | −0,008 | 0,042 | 3/5 | sem efeito detectável |
    | H3 normalização global | 0,771 → 0,604 | −0,168 | 0,051 | 4/5 | efeito (pior) |
    | H3 normalização por intervalo | 0,771 → 0,900 | +0,129 | 0,017 | 5/5 | efeito (melhor) |

    Nas métricas de apoio, H4 e H5 também ficaram sem efeito detectável em todas. Nas duas H3, `iou_sint_25` ficou sem efeito detectável (3/5) e as demais mostraram efeito. O Dice contra o pseudo-rótulo foi 0,654 (H4), 0,606 (H5), 0,450 (H3 global) e 0,936 (H3 por intervalo); ele não é comparável entre normalizações, pois o pseudo-rótulo muda junto.

    **Fração de pixels abaixo do limiar de Otsu (seção 5.8)**, na ordem referência / global / intervalo: antilope25 0,035 / 0,030 / 0,187; antilope37 0,106 / 0,157 / 0,150; botorosa47 0,155 / 0,129 / 0,164; coala88 0,182 / 0,075 / 0,251; tatu22 0,521 / 0,519 / 0,239. Limiares (antilope25, antilope37, botorosa47, coala88, tatu22): referência 0,541, 0,623, 0,670, 0,682, 0,760; global 0,580, 0,393, 0,682, 0,701, 0,511; intervalo 0,670, 0,662, 0,666, 0,670, 0,666. Tabela salva em `results/ablacoes/h3_fracao_escura.csv`.

    **Leitura calibrada.** (1) H4 e H5: não foi detectada diferença no IoU sintético com 3 sementes; isso não prova ausência de efeito (U14). Nelas, o teste sintético é idêntico ao da referência, então a comparação é direta. (2) H3: a regra aponta efeito nas duas normalizações, em sentidos opostos. Porém o limiar de Otsu muda, e com ele o contraste do teste sintético (fator × limiar), os fundos escolhidos e a região ignorada. A referência e a H3 foram avaliadas em testes diferentes (U22). Por isso o resultado não permite dizer que a normalização por intervalo é melhor nem que a global é pior. (3) Os limiares quase iguais na normalização por intervalo (0,662 a 0,670) e a fração escura de 52% do tatu22 na referência são fatos medidos; a causa de cada um não foi investigada (U23). **Próximo passo possível:** gerar o teste sintético uma vez, em dB, e aplicar cada normalização sobre as mesmas imagens e máscaras; exige novo treino das duas H3 (cerca de 1 h de GPU).

75. **Código da H3 com teste sintético comum e do notebook de investigação do tatu22 (17/11/2026).** Foi definido tornar a H3 comparável (opção 1 do item 74). Módulo novo `src/synth/comum.py`: `gerar_comum` gera o teste sintético uma vez, na normalização da referência (mesma semente, mesmos fundos, máscaras e regiões ignoradas), e converte para dB (fundos pela amplitude bruta; valor da senoide pelo valor em dB que tinha na referência); `projetar` aplica cada normalização às mesmas imagens. Em `src/data/normalizacao.py`: `parametros` (percentis por linha de cada poço, em cada modo) e `aplicar_parametros`. Em `src/train/treino.py`: o argumento `sint` de `executar_fold` e `rodar_lowo` recebe o teste pronto; sem ele, o comportamento é o anterior. `05_ablacoes.ipynb` ganhou a seção 5.10 (gera o teste comum, confere que a normalização por poço reproduz o teste da referência com diferença máxima abaixo de 1e-3, e treina de novo as duas H3 em `results/ablacoes/h3c_global` e `h3c_intervalo`) e a seção 5.11 (leitura só das métricas sintéticas; o Dice contra o pseudo-rótulo fica de fora, por não ser comparável). A referência não é retreinada: o teste comum na normalização por poço é igual ao dela, e a seção 5.10 confere isso. Teste fora do Colab, com dados sintéticos: a projeção reproduz `normalizar` sem diferença, máscaras e regiões ignoradas iguais, diferença de `x` em cerca de 2,4e-4 (arredondamento float16), e um *fold* completo roda em CPU com o teste comum. Novo notebook `06_investigacao_tatu22.ipynb` (CPU, sem treino): tabela de limiares e frações escuras por poço (Otsu, Multi-Otsu e K-means), histogramas em valor normalizado e em dB, fração escura por patch ao longo da profundidade e por poço, e patches do tatu22 e do botorosa47 com o pseudo-rótulo, em tons de cinza; saídas em `results/investigacao_tatu22/`. **Nada disso foi executado no Colab.**

76. **Notebook 06 executado (investigação do tatu22; Colab, 17/11/2026).** Tabela 6.3 (limiar Otsu / Multi-Otsu / K-means; fração escura): antilope25 0,541 / 0,475 / 0,545 (0,035 / 0,029 / 0,036); antilope37 0,623 / 0,455 / 0,622 (0,106 / 0,047 / 0,106); botorosa47 0,670 / 0,490 / 0,666 (0,155 / 0,057 / 0,152); coala88 0,682 / 0,447 / 0,682 (0,182 / 0,040 / 0,183); tatu22 0,760 / 0,439 / 0,758 (0,521 / 0,035 / 0,519). Mediana em dB do tatu22: 28,8 (os outros: 29,6 a 36,9). **Fatos observados:** (1) a fração escura por patch do tatu22 é próxima de 0 de cerca de 2.170 a 2.290 m e passa a 0,6–1,0, com oscilações, de cerca de 2.320 a 2.580 m; (2) 361 dos patches válidos do tatu22 (56,5%) têm mais de 50% de pixels escuros; coala88, 61 (8,9%); botorosa47, 26 (4,2%); (3) nos patches do tatu22 com 100% escuro (2.581 m), a imagem é cinza uniforme, com poucos pontos escuros isolados, e o pseudo-rótulo é preto por inteiro; (4) o histograma em dB do tatu22 tem duas massas; (5) o botorosa47 tem patches de 74% a 75% escuros por volta de 2.344 m, em imagem cinza texturizada; (6) o limiar de K-means coincide com o de Otsu em todos os poços; o da classe mais escura do Multi-Otsu dá frações de 3% a 6% nos 5 poços. **Leitura calibrada:** nesses trechos, o pseudo-rótulo de Otsu marca um nível de amplitude, e não feições; isso é compatível com o tatu22 não ter nenhum patch anotado "sem feição" (item 53) e com o Dice baixo desse poço como poço de teste. **Hipótese, não testada:** a normalização por intervalo (H3) reduz o efeito, o que poderia explicar parte do ganho de IoU observado (item 74). A causa física do regime de baixa amplitude não foi investigada. **Consequência para o artigo:** a comparação com o Otsu (Fase 2) e o treino em pseudo-rótulos de Otsu têm esse limite (U23, U12). **Decisão do autor (H11), código escrito; executada em 18/11/2026 (item 79):** treinar com o pseudo-rótulo da classe mais escura do Multi-Otsu (seções 5.12 e 5.13 do `05_ablacoes.ipynb`); **Correção feita em 18/11/2026 (item 77):** a frase "a comparação com a referência é direta" estava errada para a H11. O teste sintético comum define as feições pelo limiar da referência, e a H11 muda essa definição; o IoU ficaria enviesado contra a H11. Ela só deve ser lida de forma qualitativa.

77. **H3 com teste sintético comum executada (notebook 05, seções 5.10 e 5.11; Colab Pro, 18/11/2026). Decisão do autor (18/11/2026): a H3 é relatada como inconclusiva.** Conferência: a normalização por poço reproduz o teste da referência (diferença máxima de `x` 0,00024; máscaras e regiões ignoradas iguais nos 5 poços). Os números abaixo foram recalculados em 18/11/2026 a partir de `h3c_global_resumos.csv`, `h3c_intervalo_resumos.csv` e `modelos_sint_resumos.csv` (5 poços × 3 sementes, U-Net), com `tabela_ablacao`:

    | Configuração | Métrica | Ref → cand. | Diferença | Desvio entre sementes | Poços com mesmo sinal | Leitura |
    |---|---|---|---|---|---|---|
    | H3 global | iou_sint | 0,771 → 0,521 | −0,251 | 0,057 | 4/5 | efeito |
    | H3 global | dice_sint | 0,824 → 0,575 | −0,249 | 0,063 | 4/5 | efeito |
    | H3 global | iou_sint_25 | 0,909 → 0,826 | −0,082 | 0,039 | 3/5 | sem efeito detectável |
    | H3 global | iou_sint_75 | 0,887 → 0,497 | −0,390 | 0,113 | 4/5 | efeito |
    | H3 global | iou_sint_95 | 0,518 → 0,239 | −0,279 | 0,067 | 4/5 | efeito |
    | H3 intervalo | iou_sint | 0,771 → 0,366 | −0,406 | 0,020 | 5/5 | efeito |
    | H3 intervalo | dice_sint | 0,824 → 0,467 | −0,357 | 0,019 | 5/5 | efeito |
    | H3 intervalo | iou_sint_25 | 0,909 → 0,368 | −0,541 | 0,033 | 5/5 | efeito |
    | H3 intervalo | iou_sint_75 | 0,887 → 0,369 | −0,518 | 0,034 | 5/5 | efeito |
    | H3 intervalo | iou_sint_95 | 0,518 → 0,360 | −0,158 | 0,055 | 4/5 | efeito |

    A referência é a U-Net de `modelos_sint_resumos.csv`. A coluna `dice_pseudo_teste` não entra na leitura, porque o pseudo-rótulo muda entre as configurações (média: referência 0,625; global 0,451; intervalo 0,937).

    IoU sintético por poço (média das 3 sementes): global — antilope25 0,926, antilope37 0,306, botorosa47 0,458, coala88 0,446, tatu22 0,468; intervalo — antilope25 0,372, antilope37 0,521, botorosa47 0,436, coala88 0,323, tatu22 0,177.

    **Fato observado:** no teste comum, as duas normalizações ficaram abaixo da referência em `iou_sint`; a H3 por intervalo, que no item 74 tinha IoU de 0,900 (+0,129) em um teste com limiar próprio, passou a 0,366 (−0,406). **Fato observado adicional:** na H3 por intervalo o IoU é quase o mesmo nos três contrastes (0,368; 0,369; 0,360) e quase igual entre sementes em cada poço (antilope25: 0,372; 0,373; 0,371), enquanto o Dice contra o pseudo-rótulo é alto (0,907 a 0,955). **Hipótese, não testada:** a saída do modelo quase não depende do contraste da feição sintética, o que sugere que ele reproduz o pseudo-rótulo e não responde ao que o teste insere. Essa hipótese e a U24 (o teste usa o limiar de Otsu da referência para definir as feições e a região ignorada) não são separáveis com os dados atuais. **Leitura calibrada:** o sinal da H3 depende do teste usado; por isso nenhum dos dois testes sustenta uma ordem entre as normalizações, e a leitura do item 74 ("por intervalo é melhor") fica retirada. **Estado da H3:** inconclusiva. **H4 e H5 não são afetadas**, porque mantêm o limiar do pseudo-rótulo e o teste da referência. **H11 (item 76):** a H11 também muda a definição de "escuro"; ela fica fora da leitura por IoU e só pode ser lida de forma qualitativa. **Proposta não executada:** medir a detecção sem depender do limiar (AUC por pixel da probabilidade contra a máscara sintética); exige salvar os mapas de probabilidade e retreinar a referência e as H3 (cerca de 2 h de GPU, mais cerca de 1 h de código). Fica como trabalho futuro, citado nas limitações do artigo.

78. **Artigo e limitações atualizados (18/11/2026).** Ajustes: a nota da Tabela 3 passou a citar a fonte (itens 75 e 77). Tabela 3 do artigo preenchida com H4 e H5 (sem efeito detectável) e H3 (inconclusiva), com os números do item 77 e do item 74. Texto de resultados das ablações escrito com verbos calibrados. Resumo e *abstract* atualizados. Lista de limitações do artigo e `docs/limitacoes.md` com U22, U23 e U24. Prévia compilada com 6 páginas (limite informal de 8 páginas: item 6) (o `babel` com `brazil` continua só no arquivo para o Overleaf). **Fase 4 (escrita), pendências no `artigo.tex` marcadas com `\pend`:** Discussão; Conclusão (no máximo 3 afirmações, cada uma ligada a uma tabela); lista de contribuições; comparação com Silva (2026); ablações H10, H8, H6, H9 se forem executadas; e-mail institucional; repositório e licença; forma exigida da declaração de IA (item 6).

79. **H7 e H11 definidas para execução; H9 em avaliação (18/11/2026).** Decisão do autor: executar a H7 (encoder sem pré-treino) e a H11 (limiar Multi-Otsu no pseudo-rótulo); tentar a H9 e confirmar mais adiante se ela será executada. **Código:** o notebook 05 ganhou as seções 5.14 (H7, `dict(CFG, pretreino=False)`, sem mudança em `src`) e 5.15 (leitura da H7 pelo IoU sintético e AUC por patch no conjunto anotado para a referência, a H7 e a H11). A seção 5.12 (H11) foi reescrita: o IoU sintético da H11 não é neutro (U22, U24), e por isso a leitura principal da H11 é a AUC por patch no conjunto anotado, de 71 patches (leitura qualitativa, com pouco poder). A H7 usa o mesmo teste sintético da referência, e o IoU vale. Cada ablação leva cerca de 30 min de GPU. **H9, requisitos antes de executar:** (a) definir o autoencoder (arquitetura e treino) a partir de Silva (2026); a fonte ainda precisa ser lida em detalhe; (b) decidir em quais imagens o autoencoder treina (treinar com o poço de teste, sem rótulos, é um ponto a declarar); (c) gerar o pseudo-rótulo por Otsu na imagem reconstruída; (d) como a H9 muda o pseudo-rótulo, o IoU sintético seria enviesado como na H11 (U24), e a leitura teria de usar a AUC por patch. Estado: não iniciada.

80. **Regra de leitura por AUC definida (18/11/2026).** Para as ablações que mudam o pseudo-rótulo (H11, H9), e como leitura de apoio na H7: há efeito se o intervalo de 95% da diferença de AUC por patch (candidata menos referência, *bootstrap* pareado nos mesmos 71 patches anotados) exclui zero e a diferença média entre sementes passa de 1 desvio-padrão; caso contrário, "não foi detectada diferença". O critério de 4 de 5 poços não vale aqui: o tatu22 não tem negativos e o coala88 tem 2, e a AUC por poço fica indefinida ou instável. Código: `src/eval/leitura_auc.py` (`ler_auc`), testado com dados sintéticos; seção 5.16 do notebook 05. Limites: 71 patches (pouco poder); a anotação usa o critério "zona escura clara", próximo do que o pseudo-rótulo marca (U9). **H9, decisões já definidas (18/11/2026):** fonte do autoencoder: Silva (2026), seção "Interval-wise denoising autoencoder" (patches 32×32, passo 8; canais 1→16→32→64; latente 64; ruído 0,05; Adam 1e-3; lote 128; 60 ou 120 épocas; janela de Hann); um autoencoder por poço, treinado só na imagem do próprio poço (simplificação adotada: Silva treina por intervalo); pseudo-rótulo por Otsu binário na imagem reconstruída (Silva usa Multi-Otsu com 4 classes); a entrada da U-Net continua a imagem normalizada; leitura por AUC (item 80). Falta: plano de execução e tempo.

81. **H9: código da etapa 1 escrito e testado (18/11/2026; ainda não executado no Colab).** Arquivos novos: `src/models/autoencoder.py` (autoencoder de Silva (2026): 32×32, canais 1→16→32→64, latente 64, saída sigmoide; a ativação intermediária não é dita na fonte, e foi adotada a ReLU, escolha do autor); `src/labels/pseudo_ae.py` (treino por poço com ponto de controle por época, reconstrução com janela de Hann de piso 0,05, Otsu binário da imagem reconstruída, tabela de diagnóstico com limiar, fração escura e erro de reconstrução); `notebooks/07_h9_autoencoder.ipynb` (7.1 a 7.8). Arquivos alterados, com comportamento antigo preservado: `src/data/dataset_patches.py` (parâmetro `imagens_rotulo`: o rótulo vem de outra imagem, com o mesmo deslocamento circular) e `src/train/treino.py` (`imagens_rotulo` em `executar_fold` e `rodar_lowo`). **Testes (dados sintéticos, CPU):** o `DatasetPatches` antigo e o novo dão saídas idênticas, com e sem deslocamento, e com `imagens_rotulo` igual à imagem; o rótulo de outra imagem sai com a fração escura esperada; o treino do autoencoder roda, salva e retoma de onde parou; a reconstrução tem a mesma forma da imagem e valores em [0, 1]; `rodar_lowo` com `imagens_rotulo` roda de ponta a ponta com a CNN rasa. **Não testado:** tempo real na GPU e qualidade da reconstrução nos dados do WAID. **Detalhes de implementação, a declarar no artigo:** os patches de treino não contêm linhas excluídas pela D1; a perda é a soma dos erros quadrados por patch, média no lote (equação 5 de Silva); a reconstrução fica no domínio normalizado pela D3 (sem a transformação por tanh de Silva); a última posição de patch em cada eixo é incluída para cobrir a borda; o teste sintético da H9 é o da referência. **Execução:** o notebook 07 tem uma parada na seção 7.4 (teste de 2 épocas e estimativa de tempo) e outra na 7.6 (fração de pixels escuros por poço).

82. **H7 executada; H11 executada, leitura pareada pendente (notebook 05, seções 5.12 a 5.15; Colab; lido em 18/11/2026).** **H7 (encoder sem pré-treino):** IoU sintético da referência 0,771 e da H7 0,758 (diferença −0,013; desvio entre sementes 0,030; 2 de 5 poços com o mesmo sinal); as outras quatro métricas sintéticas também ficaram sem efeito detectável (`dice_sint` −0,003, `iou_sint_25` −0,034, `iou_sint_75` −0,035, `iou_sint_95` +0,030). AUC por semente: referência 0,783, 0,818 e 0,818; H7 0,820, 0,803 e 0,805. Média da diferença por semente +0,003, abaixo do desvio (0,030). Foi observado que o encoder de pesos aleatórios não mudou o resultado nesta configuração; não foi detectada diferença (U20 passa de "sem teste" para "testada nesta configuração; sem fonte de especialista"). **H11 (decisão do autor; limiar Multi-Otsu no pseudo-rótulo):** AUC por semente 0,922, 0,924 e 0,920 (média 0,922 ± 0,002), contra 0,806 ± 0,020 da referência; F1 0,934, precisão 0,928, revocação 0,939 (referência: 0,845, 0,814 e 0,879; Otsu: 0,920, 0,897 e 0,945). As diferenças por semente são +0,139, +0,106 e +0,102 (média +0,116; mais de 1 desvio entre sementes). O intervalo pareado de 95% ainda não foi calculado (seção 5.16); por isso o veredito da H11 fica **provisório**. **Cuidados:** a anotação usa o critério "zona escura clara", próximo do que o pseudo-rótulo marca (U9); a AUC mede a concordância com esse critério, e não a qualidade geológica. Foi testada uma só alternativa de limiar, sem ajuste. **U25 (aberta):** a AUC da referência no item 61 é 0,811 ± 0,010, e na seção 5.15 saiu 0,806 ± 0,020 (por semente: 0,783, 0,818 e 0,818); o F1, a precisão e a revocação são iguais nos dois casos. A causa não foi apurada (coluna de escore ou arquivo lido). Não afeta as comparações da H7 e da H11, que usam a mesma referência; o artigo precisa de um só valor.

83. **Leitura pareada por AUC (seção 5.16, 18/11/2026; Colab, CPU).** AUC do escore médio das 3 sementes, mesmos 71 patches. **H7:** referência 0,809, H7 0,810, diferença +0,001 (IC 95% −0,072 a 0,073); diferença média entre sementes +0,003, desvio 0,029; leitura: não foi detectada diferença. **H11 (decisão do autor):** referência 0,809, H11 0,923, diferença +0,114 (IC 95% 0,046 a 0,200); diferença média entre sementes +0,116, desvio 0,020; leitura: **efeito** (AUC maior). O veredito da H11 do item 82 deixa de ser provisório. **U25 (restrita):** os valores de AUC da referência dependem da definição: 0,806 é a média das AUC por semente; 0,809 é a AUC do escore médio das sementes (usada na leitura pareada); 0,811 vem do item 61 (notebook 03) e pode vir de outra execução de treino; a causa da diferença entre 0,806 e 0,811 e a igualdade do F1 não foram apuradas. O artigo usa 0,809 para a leitura pareada e declara a definição. **Código:** seção 5.17 do notebook 05 (H7 e H11 contra o Otsu, mesma técnica da Fase 2, escores em `baselines_limiar_escores.csv`), ainda não executada; serve para dizer se a H11 fecha a distância de 0,12 para o Otsu. **Artigo:** Tabela 3 com a linha da H7; nova Tabela 4 (AUC por patch, H7 e H11); texto da H11 com a ressalva da U9; resumo, *abstract*, métodos e limitações atualizados.

84. **H11 e H7 contra o Otsu (seção 5.17, 18/11/2026; Colab, CPU).** AUC do escore médio das 3 sementes, mesmos 71 patches; Otsu 0,932. Diferença pareada (modelo menos Otsu), IC 95% por *bootstrap*: referência −0,123 (−0,213 a −0,050); H7 −0,122 (−0,203 a −0,046); H11 −0,009 (−0,052 a 0,031). Foi observado que a referência e a H7 ficam abaixo do Otsu, com intervalos que excluem zero, e que para a H11 não foi detectada diferença para o Otsu. O intervalo da H11 é largo (de −0,052 a 0,031) e não prova equivalência; a H11 também não superou o Otsu. A referência concorda com a Fase 2 (item 63: −0,120, de −0,210 a −0,048) dentro da diferença de definição da AUC (U25). **Leitura calibrada para a Discussão:** (1) com o pseudo-rótulo do Otsu binário, os modelos ficaram abaixo do limiar nesta configuração; (2) com o limiar Multi-Otsu (decisão do autor), não foi detectada diferença para o Otsu; isso sugere que a definição de "escuro" do pseudo-rótulo limita os modelos da referência, hipótese não testada com outros limiares; (3) nenhum modelo superou o Otsu, e por isso H1 (modelo mais próximo do conjunto anotado que o limiar) não é confirmada nesta configuração; (4) a AUC concorda com um critério de anotação próximo da pista do pseudo-rótulo (U9), e o Otsu usa essa pista direto. **Artigo:** Tabela 4 reescrita (AUC, diferença para a referência e para o Otsu), frase sobre o Otsu e resumo/*abstract* atualizados.

85. **Discussão e Conclusão escritas em rascunho (18/11/2026).** Discussão em 4 parágrafos: (1) modelos abaixo do Otsu com o pseudo-rótulo binário, três hipóteses abertas (pseudo-rótulo ruidoso, 10 épocas, sem ajuste), indício da H11 e do regime do tatu22 (U23), sem causa física; (2) H4, H5 e H7 sem diferença detectada, poder baixo, H3 inconclusiva; (3) comparação qualitativa com Silva (2026) e Anjos et al. (2022), sem comparar números; (4) limite da AUC contra anotação de critério próximo (U9). Conclusão com 3 afirmações, cada uma ligada a uma tabela. **Estado da prévia:** 8 páginas, sendo o corpo até a página 7 e as referências na 8; 9 marcadores `\pend`. Marcadores novos ou mantidos: comparação com Silva (2026) após ler as figuras suplementares; incluir a H9 se for executada; unificar o valor da AUC da U-Net (0,811 na Tabela 1 e 0,809 na Tabela 4; U25). Limite de páginas: 8 é a informação informal do item 6; os marcadores vermelhos ocupam espaço e somem com `\rascunhofalse`. **H9:** a GPU não estava disponível em 18/11/2026 às 11:31 (mensagem do Colab: nenhum *back-end* com GPU); a seção 7.4 do notebook 07 espera a GPU. O treino de um autoencoder na CPU com os dados completos levaria muitas horas por poço, e por isso a H9 não pode ser executada em CPU.

86. **Teste de tempo da H9 e figuras do artigo (18/11/2026).** Colab Pro com GPU L4 (22,5 GB; 1,94 unidade por hora; 90,09 unidades). Seção 7.3 do notebook 07: 213 escores da referência (esperado: 213). Seção 7.4: 2 épocas no antilope25; época 1: 9,0 s, época 2: 7,4 s; perda por patch (soma dos erros quadrados) 4,975 e 2,268; patches por poço: antilope25 272.440, antilope37 216.760, botorosa47 197.860, coala88 219.440, tatu22 204.900 (total 1.111.400). **Estimativa para 5 poços e 60 épocas: 30 min.** Decisão do autor: seguir para a seção 7.5 (autoencoders), parar na 7.6 se a fração de pixels escuros da imagem reconstruída ficar fora de 1% a 60% em algum poço (critério pendente de definição pelo autor), e seguir para 7.7 e 7.8. **Artigo:** duas figuras novas em `artigo/figuras/` (`fig_metodo.pdf`, diagrama do método; `fig_auc.pdf`, diferenças de AUC com IC de 95%, valores dos itens 83 e 84). A prévia com marcadores tem 9 páginas; sem marcadores (`\rascunhofalse`) tem 8 páginas, e o corpo termina na página 8 junto das referências. A H9 acrescentará cerca de 0,15 página; os cortes previstos são nas limitações do artigo (o texto completo fica em `docs/limitacoes.md`).

87. **H9: autoencoders e conferência (notebook 07, seções 7.5 e 7.6; Colab, GPU L4; 18/11/2026).** Treino dos 5 autoencoders: 60 épocas por poço, cerca de 5,3 s por época no tatu22, cerca de 28 min no total (a estimativa da seção 7.4 era 30 min). Perda por patch no tatu22: 1,260 na época 9 e 1,131 na época 60. Tabela de diagnóstico (limiar da imagem bruta / da reconstruída; fração escura bruta / reconstruída; erro quadrático médio da reconstrução): antilope25 0,541 / 0,552; 0,035 / 0,036; 0,00117. antilope37 0,623 / 0,634; 0,106 / 0,111; 0,00246. botorosa47 0,670 / 0,690; 0,154 / 0,172; 0,00297. coala88 0,682 / 0,697; 0,182 / 0,205; 0,00164. tatu22 0,760 / 0,758; 0,522 / 0,533; 0,00094. **Conferência:** todas as frações da imagem reconstruída ficam entre 1% e 60% (critério agora adotado pelo autor ao seguir para a 7.7). **Foi observado:** os limiares e as frações escuras do pseudo-rótulo da H9 ficam perto dos da referência (diferença de fração entre +0,001 e +0,023); na figura, o pseudo-rótulo da H9 tem formas mais suaves e menos pontos isolados que o da referência, nas mesmas regiões. **Hipótese, não testada:** por isso a H9 deve mudar pouco o treino; o resultado da seção 7.8 confirma ou não. O regime do tatu22 (fração de 0,533 com o autoencoder; U23) não muda com o autoencoder, pois o limiar continua o Otsu binário. A H11, que muda o limiar, mudou o resultado (item 83); a H9 muda a imagem e não a definição de "escuro". Execução das seções 7.7 e 7.8 em andamento.

88. **H9 executada e lida (notebook 07, seções 7.7 e 7.8; Colab, GPU L4; 18/11/2026).** AUC do escore médio das 3 sementes, mesmos 71 patches. Referência 0,809; H9 0,849; diferença pareada +0,040 (IC 95% −0,007 a 0,097); diferença média entre sementes +0,037, desvio 0,027. AUC por semente: referência 0,783, 0,818 e 0,818; H9 0,837, 0,869 e 0,824 (diferenças +0,054, +0,051 e +0,006). **Leitura pela regra do item 80:** a diferença média entre sementes passa de 1 desvio, mas o intervalo de 95% inclui zero; por isso, não foi detectada diferença. **Calibração:** a estimativa é positiva nas 3 sementes; com 71 patches, o poder é baixo, e um efeito pequeno não é excluído. A hipótese do item 87 ("a H9 deve mudar pouco o treino") subestimou a diferença: o ponto estimado é +0,040, maior que o esperado, embora sem efeito detectado pela regra. **Contraste com a H11:** a H11 muda a definição de "escuro" (+0,114; efeito); a H9 muda só a limpeza da imagem (+0,040; sem efeito detectado). Isso sugere que o limiar pesa mais que a limpeza do rótulo; não foi testado com outros limiares. **Código:** seção 7.9 do notebook 07 (referência e H9 contra o Otsu, mesma técnica da seção 5.17), ainda não executada; a célula da Tabela 4 do artigo fica com `\pend{seção 7.9}` até lá. **Artigo (rascunho de 18/11/2026):** H9 incluída nos métodos (autoencoder por poço, 60 épocas, Otsu binário da reconstrução), na Tabela 4 e na figura de diferenças de AUC (linha "H9 − referência"), na Discussão (contraste com a H11), na Conclusão (afirmação 3), no resumo e no *abstract*. Cortes para o limite de páginas: limitações condensadas em 5 itens (o texto completo fica em `docs/limitacoes.md`); parágrafo do IoU sintético encurtado; duas frases da Discussão retiradas. **Tamanho:** 8 páginas sem marcadores (corpo e referências até a página 8), 9 com marcadores; qualquer acréscimo precisa de um corte equivalente.

89. **H9 contra o Otsu (seção 7.9, 18/11/2026; Colab, GPU L4).** AUC do escore médio das 3 sementes, mesmos 71 patches; Otsu 0,932. Diferença pareada (modelo menos Otsu), IC 95% por *bootstrap*: referência −0,123 (−0,213 a −0,050); H9 −0,083 (−0,170 a −0,016). Foi observado que a H9 continua abaixo do Otsu, com intervalo que exclui zero; a H11 (item 84) é a única configuração sem diferença detectada para o Otsu. **Leitura calibrada:** com 71 patches, o poder é baixo; a distância da H9 para o Otsu (−0,083) é menor que a da referência (−0,123), mas a comparação direta entre as duas distâncias não foi feita (a diferença H9 menos referência é +0,040, com intervalo que inclui zero, item 88). **Artigo:** Tabela 4 completa (linha da H9), gráfico com a linha "H9 − Otsu", texto dos resultados e da Discussão atualizados. Execução da seção 7.9 após reinício do ambiente: foi preciso executar as seções 7.1, 7.2, 7.3, 7.5, 7.7 e 7.9, nesta ordem (as variáveis `esc_ref` e `esc_h9` nascem na 7.3 e na 7.7).

90. **Definições sobre os dados do artigo e licença (18/11/2026).** (1) E-mail do artigo: o e-mail pessoal do autor (campo `\email` do modelo SBC; o modelo não exige e-mail institucional). (2) Repositório: definido na sequência; o artigo mantém `\pend{endereço do GitHub}`. (3) Licença do código: Apache 2.0, padrão; permite usar, modificar e redistribuir com atribuição e aviso de licença, com concessão de patente e sem garantia; o aviso original do arquivo copiado do WAID (`src/data/waid_concat.py`, Apache 2.0, commit `f8926d0`) é mantido; os dados do WAID continuam da PETROBRAS, sob CC BY 4.0, e não fazem parte do repositório. Arquivos criados: `LICENSE` (texto padrão da Apache 2.0, sem alteração; conferido com o texto oficial em apache.org antes de publicar) e `NOTICE` (direitos autorais de 2026 do autor, a origem do arquivo copiado do WAID e a nota sobre os dados). Ambos ficam na raiz da pasta `waid`. (4) Declaração de IA: a disciplina não informou regra; fica a frase "Assistente de IA usado para apoio ao código e à revisão do texto. Decisões, experimentos e conclusões são do autor." na seção "Uso de ferramentas de IA". (5) Limite de páginas: a disciplina não informou regra; o artigo fecha com até 8 páginas (corpo e referências). Se a professora informar outra regra, as duas decisões (4 e 5) serão reabertas. **Pendências do repositório:** o que entra e o que não entra. Endereço e conferências: item 92.

91. **Comparação com Silva (2026) fechada; U25 em execução (18/11/2026).** (1) *Silva (2026), arXiv:2603.20729:* usa o WAID sem anotação de especialista; gera pseudo-rótulos com Multi-Otsu de quatro classes sobre a imagem limpa por autoencoder, refina com redes e mede a concordância com o próprio pseudo-rótulo, que o texto descreve como consistência interna, não validação geológica. Valores relatados, média entre poços: Otsu na imagem bruta 0,6138; AE→Otsu 0,7026; AE+K-means 0,4713; refinador só com imagem 0,7339. Métricas, classes (quatro contra duas) e dados diferem; a comparação no artigo é só qualitativa, e as duas marcações `\pend` sobre Silva foram removidas do artigo. (2) *U25:* a Tabela 1 vem do treino do notebook 03 (U-Net 0,811 ± 0,010), e as ablações vêm do treino do notebook 04 (0,806 ± 0,020 por semente; 0,809 no escore médio). Foi criada a seção 5.18 do notebook 05, que recalcula AUC, F1 e a diferença pareada contra o Otsu para a CNN rasa e a U-Net só com o treino do notebook 04. Pendente: executar a seção 5.18, trocar a Tabela 1 e os números do resumo, do texto abaixo da tabela e da Conclusão (item 1, hoje com redação provisória), e declarar as duas definições de AUC. (3) *Páginas:* o texto ficou em 8 páginas sem marcadores (9 com marcadores), após cortes pequenos de redação e a remoção do campo de versão da referência de Silva no `.bib`.

92. **Repositório: endereço e conferências (18/11/2026).** (1) *Endereço:* `https://github.com/jorgesouza63` (perfil pessoal; o nome do repositório ainda não foi definido). (2) *Conferência do aviso original:* feita pelo autor e confirmada nas imagens enviadas em 18/11/2026: o cabeçalho de `src/data/waid_concat.py` (arquivo de 31/10/2026) registra a origem (`https://github.com/petrobras/waid`, seção "Acoustic image logs dataset"), o *commit* `f8926d060cac24cdee37563546db85daf06a2859` (18/06/2026), a licença Apache 2.0 do código do WAID e "ALTERAÇÕES: nenhuma", e a função `concat_IMG_data` aparece como cópia literal. (3) *Texto da LICENSE:* substituído o arquivo da raiz pelo texto oficial copiado de `apache.org/licenses/LICENSE-2.0.txt`. (4) *Pendente:* o que entra e o que não entra no repositório (decisão a ser tomada, na sequência); o nome do repositório; o campo `\pend` do endereço do artigo trocou-se pelo endereço acima.

93. **U25 resolvida: AUC dos modelos unificada com o treino de referência (seção 5.18; Colab; 18/11/2026).** A Tabela 1 do artigo passou a usar o treino do notebook 04 (`results/modelos_sint/`), o mesmo das ablações. **AUC por semente** (média ± desvio entre 3 sementes): CNN rasa 0,800 ± 0,024 (0,826, 0,780, 0,794); U-Net 0,806 ± 0,020 (0,783, 0,818, 0,818). F1: 0,839 ± 0,052 e 0,845 ± 0,041; precisão: 0,808 ± 0,029 e 0,814 ± 0,019; revocação: 0,873 ± 0,080 e 0,879 ± 0,074 (iguais aos do notebook 03). **AUC do escore médio das 3 sementes** e diferença pareada contra o Otsu (0,932; 71 patches): CNN rasa 0,791, diferença −0,141 (IC 95% −0,242 a −0,062); U-Net 0,809, diferença −0,123 (IC 95% −0,213 a −0,050). Os valores anteriores (CNN rasa 0,794 ± 0,014, U-Net 0,811 ± 0,010; diferenças −0,152 e −0,120; itens 61 a 63) vêm do treino do notebook 03 e ficam como registro histórico. As duas execuções dão F1, precisão e revocação iguais e AUC diferente; a causa não foi investigada (hipótese, não testada: treinos repetidos com não determinismo da GPU, que não muda as decisões em 0,5 e muda a ordem dos escores). No artigo, as duas definições de AUC estão declaradas (por semente na Tabela 1; escore médio nas diferenças pareadas e na Tabela 4). A diferença entre 0,806 e 0,809 para a U-Net vem da definição, não de outro treino. Conclusão (item 1) e texto abaixo da Tabela 1 foram atualizados. O artigo tem 8 páginas sem marcadores.

94. **Marcações do artigo fechadas (18/11/2026).** Artigo: (1) lista de contribuições ajustada (seis ablações, lidas por IoU sintético e AUC pareada); (2) Métodos: as ablações H10, H8 e H6 do plano não foram executadas; (3) Limitações: mantidas as cinco frases condensadas (limite de 8 páginas); o texto completo fica em `docs/limitacoes.md`, atualizado com a H7 (não foi detectada diferença; AUC +0,001), a H9 e as ablações não executadas; (4) item 3 da Conclusão reescrito para citar H4, H5, H7 e H9 (não foi detectada diferença) e H3 (inconclusiva); a H7 estava ausente da frase anterior; (5) cortes de repetição para manter 8 páginas (frase sobre o limiar único removida dos Resultados, que a repete nas Limitações; nota "não consultada" da referência de Lofts e Bourke removida do `.bib`, porque a Limitação 1 já declara que a fonte não foi consultada). Resultado: 0 marcações `\pend` no artigo; 8 páginas.

95. **Revisão final dos números e decisões do repositório (19/11/2026).** (1) *Conferência dos números:* 96 valores decimais distintos do artigo comparados com o README; as Tabelas 1 a 4 e as frases de texto conferem. Três valores não aparecem em forma literal: 0,614 e 0,703 (arredondamentos de 0,6138 e 0,7026 de Silva, item 91) e 52,1% (fração 0,521 do tatu22, item 76). Duas correções feitas no artigo: (a) resumo, em inglês e português: "0,81 e 0,79" misturava a AUC por semente (U-Net 0,806) com a do escore médio (CNN rasa 0,791); passou a "CNN rasa 0,80 e U-Net 0,81" (Tabela 1, AUC por semente); (b) "sem efeito detectável" (Tabela 3 e Métodos) passou a "sem diferença detectada", conforme a convenção de linguagem. (2) *Repositório:* nome definido `puc_pel_mel_sit_ele2346`. Montado `.gitignore` com base em arquivo pré-existente de outro projeto.

## Autor

**Jorge Alberto Avelino de Souza** – Mestrado em Engenharia Elétrica, PUC-Rio
Orientador: Prof. Dr. Marco Aurélio Cavalcanti Pacheco
Disciplina: ELE2346 - Deep Learning
Professora: Dra. Karla Figueiredo
