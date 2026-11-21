"""Leitura das ablações: compara uma configuração com a referência (regra definida em 15/11/2026).

Há efeito se |média da diferença| > 1 desvio-padrão entre sementes E o sinal da diferença
se repete em pelo menos 4 dos 5 poços de teste. Caso contrário: "sem efeito detectável".
"""
import numpy as np
import pandas as pd

METRICAS = ["iou_sint", "dice_sint", "iou_sint_25", "iou_sint_75", "iou_sint_95", "dice_pseudo_teste"]


def ler_ablacao(ref, cand, metrica="iou_sint"):
    """Aplica a regra de leitura a uma métrica. `ref` e `cand` são tabelas de resumos por fold.

    Pareia por modelo, semente e poço de teste. Devolve um dicionário com a diferença média
    (candidata − referência), o desvio entre sementes, o número de poços com o mesmo sinal e o veredito.
    """
    d = ref.merge(cand, on=["modelo", "semente", "poco_teste"], suffixes=("_ref", "_cand"))
    d["delta"] = d[f"{metrica}_cand"] - d[f"{metrica}_ref"]
    por_semente = d.groupby("semente").delta.mean()
    por_poco = d.groupby("poco_teste").delta.mean()
    media, dp = float(por_semente.mean()), float(por_semente.std())
    mesmo_sinal = int((np.sign(por_poco) == np.sign(media)).sum())
    efeito = abs(media) > dp and mesmo_sinal >= 4
    return {"metrica": metrica, "ref": float(d[f"{metrica}_ref"].mean()), "cand": float(d[f"{metrica}_cand"].mean()),
            "delta": media, "dp_sementes": dp, "pocos_mesmo_sinal": f"{mesmo_sinal}/{len(por_poco)}",
            "veredito": "efeito" if efeito else "sem efeito detectável"}


def tabela_ablacao(ref, cand, metricas=METRICAS):
    """Aplica `ler_ablacao` a várias métricas. A principal é `iou_sint`; as outras são de apoio."""
    return pd.DataFrame([ler_ablacao(ref, cand, m) for m in metricas]).round(3)
