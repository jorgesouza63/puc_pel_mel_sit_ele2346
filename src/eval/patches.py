"""Avaliação por patch contra o conjunto anotado (feição clara × sem feição).

O escore do patch é a fração de pixels escuros (ou a fração de pixels preditos
como feição, no caso de um modelo). Patches em "dúvida" não entram.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.labels.limiares import fracao_escura


def tabela_escores(npy_dir, anot, limiares, altura=256):
    """Para cada patch anotado (sem dúvida), a fração escura de cada método.

    anot     : tabela de anotações (colunas poco, linha_ini, decisao)
    limiares : DataFrame poço × método (saída de `calcular_limiares`)
    Devolve poco, linha_ini, y (1 = feição clara) e uma coluna por método.
    """
    a = anot[anot.decisao.isin(["feicao_clara", "sem_feicao"])].copy()
    a["y"] = (a.decisao == "feicao_clara").astype(int)
    saida = []
    for poco, g in a.groupby("poco"):
        g = g[["poco", "linha_ini", "y"]].copy()
        for m in limiares.columns:
            g[m] = fracao_escura(npy_dir, poco, g.linha_ini.values, limiares.loc[poco, m], altura)
        saida.append(g)
    return pd.concat(saida).reset_index(drop=True)


def auc_com_ic(y, s, n_boot=2000, semente=0):
    """AUC e intervalo de confiança de 95% por bootstrap (percentis). NaN sem as duas classes.

    Com n_boot=0, devolve só a AUC (o intervalo fica NaN).
    """
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, s)
    if n_boot == 0:
        return float(auc), np.nan, np.nan
    rng = np.random.default_rng(semente)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    if not vals:
        return float(auc), np.nan, np.nan
    return float(auc), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _melhor_corte(y, s):
    """Corte do escore que maximiza o F1 em (y, s)."""
    melhor, corte = -1.0, 0.0
    for c in np.unique(s):
        p = s >= c
        tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > melhor:
            melhor, corte = f1, c
    return corte


def f1_lowo(df, coluna):
    """F1 com corte escolhido nos outros poços (leave-one-well-out) e aplicado no poço de teste.

    Devolve (F1, precisão, revocação) com as previsões de todos os poços juntas.
    """
    pred = np.zeros(len(df), dtype=bool)
    for poco in df.poco.unique():
        teste = (df.poco == poco).values
        corte = _melhor_corte(df.y.values[~teste], df[coluna].values[~teste])
        pred[teste] = df[coluna].values[teste] >= corte
    y = df.y.values == 1
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rev = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return f1, prec, rev


def avaliar(df, colunas):
    """Tabela por método: n, AUC com IC, F1 (corte LOWO), precisão, revocação e AUC por poço."""
    linhas = []
    for c in colunas:
        auc, lo, hi = auc_com_ic(df.y, df[c])
        f1, prec, rev = f1_lowo(df, c)
        por_poco = {p: auc_com_ic(g.y, g[c], n_boot=0)[0] for p, g in df.groupby("poco")}
        linhas.append({"metodo": c, "n": len(df), "auc": auc, "ic95_baixo": lo, "ic95_alto": hi,
                       "f1_lowo": f1, "precisao": prec, "revocacao": rev,
                       **{f"auc_{p}": v for p, v in por_poco.items()}})
    return pd.DataFrame(linhas).round(3)


def avaliar_modelos(escores, coluna="score_soft"):
    """Avalia os escores de modelos (todos os folds juntos) contra o conjunto anotado.

    `escores` tem as colunas poco, y, modelo, semente e a coluna de escore.
    Devolve (tabela por modelo e semente, resumo por modelo com média e desvio entre sementes).
    """
    linhas = []
    for (m, s), g in escores.groupby(["modelo", "semente"]):
        g = g.reset_index(drop=True)
        auc, lo, hi = auc_com_ic(g.y, g[coluna])
        f1, prec, rev = f1_lowo(g, coluna)
        linhas.append({"modelo": m, "semente": s, "n": len(g), "auc": auc, "ic95_baixo": lo,
                       "ic95_alto": hi, "f1_lowo": f1, "precisao": prec, "revocacao": rev})
    por_semente = pd.DataFrame(linhas).round(3)
    resumo = (por_semente.groupby("modelo")[["auc", "f1_lowo", "precisao", "revocacao"]]
              .agg(["mean", "std"]).round(3))
    return por_semente, resumo


def comparar_pareado(escores, base, col_modelo="score_soft", col_base="otsu", n_boot=2000, semente=0):
    """Diferença de AUC (modelo − base) nos mesmos patches, com IC 95% por bootstrap pareado.

    O escore do modelo é a média entre as sementes em cada patch. O bootstrap sorteia patches
    inteiros, então modelo e base são avaliados nas mesmas amostras (comparação pareada).
    Devolve uma linha por modelo: auc_modelo, auc_base, diferenca, ic95_baixo, ic95_alto.
    """
    linhas = []
    for m, g in escores.groupby("modelo"):
        s = g.groupby(["poco", "linha_ini", "y"], as_index=False)[col_modelo].mean()
        d = s.merge(base[["poco", "linha_ini", col_base]], on=["poco", "linha_ini"])
        y, a, b = d.y.values, d[col_modelo].values, d[col_base].values
        rng = np.random.default_rng(semente)
        dif = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) == 2:
                dif.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
        linhas.append({"modelo": m, "n": len(y), "auc_modelo": roc_auc_score(y, a),
                       "auc_base": roc_auc_score(y, b), "diferenca": roc_auc_score(y, a) - roc_auc_score(y, b),
                       "ic95_baixo": np.percentile(dif, 2.5), "ic95_alto": np.percentile(dif, 97.5)})
    return pd.DataFrame(linhas).round(3)
