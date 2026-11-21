"""Teste sintético comum às normalizações (ablação H3).

Na H3 o limiar de Otsu muda com a normalização. Se cada configuração gerasse o próprio teste
sintético, o contraste (fator × limiar), os fundos e a região ignorada mudariam junto, e o IoU
não seria comparável. Aqui o teste é gerado uma vez, na normalização da referência, e passado
para dB: os fundos vêm da amplitude bruta e as senoides têm o valor em dB que tinham na
referência. Cada normalização é aplicada depois sobre as mesmas imagens, máscaras e regiões
ignoradas.
"""
import numpy as np

from src.data.normalizacao import aplicar_parametros
from src.synth.sintetico import gerar_sintetico


def gerar_comum(imagens_ref, brutas, par_ref, indice, limiares_ref, pocos, altura=256, **kw):
    """Gera o teste da referência e o converte para dB. Devolve dict poço -> dicionário."""
    comum = {}
    for p in pocos:
        x, m, ig, meta = gerar_sintetico(imagens_ref, indice, limiares_ref[p], p, altura=altura, **kw)
        pb, pa = par_ref[p][0]
        x_db = []
        for xi, mi, r in zip(x, m, meta.linha_ini):
            fundo = brutas[p][int(r):int(r) + altura].copy()
            fundo[mi] = xi[mi] * (pa - pb) + pb          # valor da senoide, de volta em dB
            x_db.append(fundo)
        comum[p] = {"x_db": np.stack(x_db), "mascara": m, "ignorar": ig, "meta": meta}
    return comum


def projetar(comum, par):
    """Aplica uma normalização (saída de `parametros`) ao teste comum.

    Devolve dict poço -> (x, mascara, ignorar, meta), o formato que `executar_fold` usa.
    """
    return {p: (aplicar_parametros(c["x_db"], c["meta"].linha_ini.values, par[p]),
                c["mascara"], c["ignorar"], c["meta"]) for p, c in comum.items()}
