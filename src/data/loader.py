"""Carga das imagens acústicas do WAID (um DataFrame por poço)."""
import os

import numpy as np

from .waid_concat import concat_IMG_data  # função oficial da PETROBRAS


def load_well_image(well_id, data_path, dtype="float32", verbose=True):
    """Carrega a imagem de amplitude (dB) de um poço.

    well_id   : "antilope25", "antilope37", "botorosa47", "coala88" ou "tatu22"
    data_path : pasta com os subarquivos AMP (ex.: .../waid/data/raw/img)
    Retorna   : DataFrame. Índice = rótulo de profundidade (m). Colunas = IMG_AMP[0..179].

    Passos:
      1. Chama a função oficial concat_IMG_data().
      2. Ordena por profundidade com ordenação ESTÁVEL (a função oficial não
         garante a ordem dos arquivos; a ordenação estável mantém a ordem original
         das linhas que têm o mesmo rótulo).
      3. Mede rótulos repetidos. Em alguns poços (antilope25, botorosa47) o
         rótulo de profundidade tem só 2 casas decimais e se repete em várias
         linhas vizinhas. Isso NÃO é erro: são amostras diferentes. A ordem das
         linhas é a ordem real de profundidade.
      4. Conta saltos de profundidade entre rótulos únicos (indício de linhas
         removidas pelo dropna).
    """
    df = concat_IMG_data(well_id, data_path)

    reordenado = not df.index.is_monotonic_increasing
    if reordenado:
        df = df.sort_index(kind="mergesort")  # mergesort = estável

    rotulos = df.index.values.astype("float64")
    unicos = np.unique(rotulos)
    linhas_por_rotulo = len(rotulos) / len(unicos)
    passo_medio = float((rotulos[-1] - rotulos[0]) / (len(rotulos) - 1))

    dif = np.diff(unicos)
    passo_rotulo = float(np.median(dif))
    n_saltos = int((dif > 1.5 * passo_rotulo).sum())

    df = df.astype(dtype)

    if verbose:
        print(
            f"{well_id}: {df.shape[0]} linhas x {df.shape[1]} colunas | "
            f"profundidade {rotulos[0]:.2f} a {rotulos[-1]:.2f} m | "
            f"passo médio {passo_medio:.5f} m | "
            f"linhas por rótulo {linhas_por_rotulo:.2f} | saltos: {n_saltos} | "
            f"reordenado: {'sim' if reordenado else 'não'}"
        )
        if linhas_por_rotulo > 1.5:
            print(
                f"  aviso: o rótulo de profundidade de {well_id} se repete em várias linhas "
                f"(resolução do rótulo menor que o passo de amostragem). Use a posição da "
                f"linha como eixo de profundidade, não o rótulo."
            )
            nomes = [f for f in os.listdir(data_path) if f.startswith(well_id)]
            if nomes != sorted(nomes):
                print(
                    "  aviso: o sistema listou os subarquivos fora de ordem. Nas fronteiras "
                    "entre subarquivos, poucas linhas com o mesmo rótulo podem ficar trocadas."
                )
    return df
