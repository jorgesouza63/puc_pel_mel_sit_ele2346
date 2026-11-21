"""Teste sintético: senoides com geometria conhecida inseridas em trechos reais.

As senoides dão máscaras exatas. Elas medem a capacidade de detecção (IoU e Dice) sem
depender de anotação humana. Os parâmetros de geometria e de contraste são escolhas deste
trabalho, dentro de faixas plausíveis, sem validação por especialista (declarar no artigo).
Fonte da forma: plano que corta o poço aparece como senoide; a altura mostra o mergulho
(Fioriti, 2026, p. 52).
"""
import numpy as np
import pandas as pd

SEMENTE_SINT = 123                 # fixa: o mesmo conjunto para todos os modelos e sementes
FATORES = (0.25, 0.75, 0.95)       # valor dentro da senoide = fator × limiar de Otsu do poço


def gerar_sintetico(imagens, indice, limiar, poco, n=24, fatores=FATORES,
                    semente=SEMENTE_SINT, altura=256):
    """Gera n fundos reais do poço (quartil de menor fração escura) × cada contraste.

    Cada fundo recebe 1 ou 2 senoides (período = 180 colunas, ou seja, 1 volta completa).
    O valor dentro da senoide é `fator × limiar`: 0,25 é contraste forte e 0,95 fica quase
    no limiar (difícil). A mesma geometria vale para todos os contrastes.
    Devolve (x, mascara, ignorar, meta): x float32 (N, altura, 180); mascara bool (senoide);
    ignorar bool (pixels que já eram escuros no fundo, fora da senoide: não contam como erro);
    meta DataFrame com poco, linha_ini e fator.
    """
    rng = np.random.default_rng(semente)
    cand = indice[indice.poco == poco].reset_index(drop=True)
    fundos = [np.asarray(imagens[poco][int(r):int(r) + altura], dtype=np.float32)
              for r in cand.linha_ini]
    frac = np.array([(f < limiar).mean() for f in fundos])
    ordem = np.argsort(frac)[:max(n, len(cand) // 4)]
    escolha = rng.choice(ordem, size=min(n, len(ordem)), replace=False)
    colunas = np.arange(fundos[0].shape[1])
    linhas = np.arange(altura)[:, None]
    xs, ms, igs, meta = [], [], [], []
    for i in escolha:
        m = np.zeros((altura, len(colunas)), dtype=bool)
        for _ in range(int(rng.integers(1, 3))):
            centro = rng.uniform(70, altura - 70)
            amp = rng.uniform(10, 60)                  # pico a pico, em linhas
            fase = rng.uniform(0, 2 * np.pi)
            espessura = rng.uniform(4, 8)              # em linhas
            curva = centro + (amp / 2) * np.sin(2 * np.pi * colunas / len(colunas) + fase)
            m |= np.abs(linhas - curva[None, :]) <= espessura / 2
        ignorar = (fundos[i] < limiar) & ~m
        for f in fatores:
            x = fundos[i].copy()
            x[m] = np.clip(f * limiar + rng.normal(0, 0.02, m.sum()), 0, 1)
            xs.append(x)
            ms.append(m)
            igs.append(ignorar)
            meta.append({"poco": poco, "linha_ini": int(cand.linha_ini[i]), "fator": f})
    return np.stack(xs), np.stack(ms), np.stack(igs), pd.DataFrame(meta)


def iou_dice(pred, mascara, ignorar=None):
    """IoU e Dice por patch (pred e mascara bool, forma N × H × W).

    Pixels em `ignorar` são tirados da previsão: marcar uma feição escura que já existia no
    fundo não é erro.
    """
    if ignorar is not None:
        pred = pred & ~ignorar
    inter = (pred & mascara).sum((1, 2)).astype(float)
    uniao = (pred | mascara).sum((1, 2)).astype(float)
    soma = pred.sum((1, 2)) + mascara.sum((1, 2))
    return inter / np.maximum(uniao, 1), 2 * inter / np.maximum(soma, 1)


def resumo_sintetico(pred, mascara, ignorar, meta):
    """IoU e Dice médios por contraste e no total. Devolve um dicionário plano."""
    iou, dice = iou_dice(pred, mascara, ignorar)
    saida = {"iou_sint": float(iou.mean()), "dice_sint": float(dice.mean())}
    for e in sorted(meta.fator.unique()):
        k = (meta.fator == e).values
        saida[f"iou_sint_{int(round(e * 100)):02d}"] = float(iou[k].mean())
        saida[f"dice_sint_{int(round(e * 100)):02d}"] = float(dice[k].mean())
    return saida


def prever_otsu(x, limiar):
    """Previsão do baseline de limiar: pixel abaixo do limiar do poço."""
    return x < limiar
