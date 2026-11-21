"""Leitura das ablações pela AUC por patch no conjunto anotado (regra definida em 18/11/2026).

Há efeito se o intervalo de 95% da diferença de AUC (candidata menos referência, bootstrap pareado
nos mesmos patches) exclui zero E |média da diferença entre sementes| > 1 desvio-padrão entre sementes.
Caso contrário: "não foi detectada diferença". Vale para ablações que mudam o pseudo-rótulo (H11, H9)
e como leitura de apoio nas demais (H7).
"""
import pandas as pd

from src.eval.patches import avaliar_modelos, comparar_pareado


def ler_auc(esc_ref, esc_cand, coluna="score_soft"):
    """Compara os escores por patch da candidata com os da referência. Devolve uma tabela de uma linha."""
    base = (esc_ref.groupby(["poco", "linha_ini"], as_index=False)[coluna].mean()
            .rename(columns={coluna: "ref"}))
    par = comparar_pareado(esc_cand, base, col_modelo=coluna, col_base="ref").iloc[0]
    a_ref = avaliar_modelos(esc_ref, coluna)[0].set_index("semente").auc
    a_cand = avaliar_modelos(esc_cand, coluna)[0].set_index("semente").auc
    d = (a_cand - a_ref).dropna()
    media, dp = float(d.mean()), float(d.std())
    exclui_zero = bool(par.ic95_baixo > 0 or par.ic95_alto < 0)
    efeito = exclui_zero and abs(media) > dp
    return pd.DataFrame([{"n_patches": int(par.n), "auc_ref": par.auc_base, "auc_cand": par.auc_modelo,
                          "dif_pareada": par.diferenca, "ic95_baixo": par.ic95_baixo, "ic95_alto": par.ic95_alto,
                          "dif_media_sementes": round(media, 3), "dp_sementes": round(dp, 3),
                          "veredito": "efeito" if efeito else "não foi detectada diferença"}])
