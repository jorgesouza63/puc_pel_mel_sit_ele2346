"""Normalizações alternativas da amplitude (ablação H3).

A referência (D3) normaliza cada poço pelos seus percentis 1 e 99. Aqui há duas alternativas:
`global` (percentis do conjunto dos 5 poços) e `intervalo` (percentis de cada bloco de
`janela` linhas do poço). Todas partem da amplitude bruta em dB, já reamostrada pela D2,
e ignoram as linhas excluídas pela D1 ao calcular os percentis.
"""
import numpy as np

FATOR_VERTICAL = {"antilope25": 2}   # D2 (igual ao notebook 01)
PERCENTIS = (1, 99)                  # D3 (igual ao notebook 01)
JANELA_INTERVALO = 2000              # linhas por bloco (≈ 10 m); escolha do autor, sem especialista
MIN_LINHAS_VALIDAS = 200             # bloco com menos linhas válidas usa os percentis do poço


def carregar_brutas(npy_dir, pocos):
    """Amplitude bruta (dB, float32) de cada poço, com a D2 aplicada, e a máscara de exclusão (D1)."""
    brutas, excl = {}, {}
    for p in pocos:
        img = np.load(f"{npy_dir}/{p}_img.npy")
        e = np.load(f"{npy_dir}/{p}_excluir_rs.npy")
        f = FATOR_VERTICAL.get(p, 1)
        if f > 1:
            n = (len(img) // f) * f
            img = img[:n].reshape(-1, f, img.shape[1]).mean(axis=1)
        brutas[p], excl[p] = img.astype(np.float32), e
        assert len(brutas[p]) == len(e), f"{p}: linhas da imagem e da máscara diferem"
    return brutas, excl


def _escalar(img, p_baixo, p_alto):
    return np.clip((img - p_baixo) / (p_alto - p_baixo), 0, 1).astype(np.float16)


def normalizar(brutas, excl, modo="poco", janela=JANELA_INTERVALO):
    """Imagens normalizadas (float16, 0 a 1) de todos os poços.

    modo: "poco" (D3, referência), "global" ou "intervalo".
    """
    if modo == "poco":
        return {p: _escalar(im, *np.percentile(im[~excl[p]][::10], PERCENTIS)) for p, im in brutas.items()}
    if modo == "global":
        todos = np.concatenate([im[~excl[p]][::10].ravel() for p, im in brutas.items()])
        pb, pa = np.percentile(todos, PERCENTIS)
        return {p: _escalar(im, pb, pa) for p, im in brutas.items()}
    if modo == "intervalo":
        saida = {}
        for p, im in brutas.items():
            pb0, pa0 = np.percentile(im[~excl[p]][::10], PERCENTIS)
            out = np.empty(im.shape, dtype=np.float16)
            for a in range(0, len(im), janela):
                bloco, ok = im[a:a + janela], ~excl[p][a:a + janela]
                pb, pa = np.percentile(bloco[ok], PERCENTIS) if ok.sum() >= MIN_LINHAS_VALIDAS else (pb0, pa0)
                out[a:a + janela] = _escalar(bloco, pb, pa)
            saida[p] = out
        return saida
    raise ValueError(f"modo desconhecido: {modo}")


def limiares_otsu(imagens, excl):
    """Limiar de Otsu de cada poço sobre as imagens normalizadas (mesma regra da referência)."""
    from src.labels.limiares import limiar_otsu_poco
    return {p: limiar_otsu_poco(im, excl[p]) for p, im in imagens.items()}


def parametros(brutas, excl, modo, janela=JANELA_INTERVALO):
    """Percentis (p_baixo, p_alto) usados em cada linha: dict poço -> array (n_linhas, 2).

    Descreve a mesma normalização de `normalizar`, mas permite aplicá-la depois a imagens
    novas (por exemplo, os patches do teste sintético comum da H3).
    """
    saida = {}
    if modo == "global":
        todos = np.concatenate([im[~excl[p]][::10].ravel() for p, im in brutas.items()])
        glob = np.percentile(todos, PERCENTIS)
    for p, im in brutas.items():
        par = np.zeros((len(im), 2))
        pb0, pa0 = np.percentile(im[~excl[p]][::10], PERCENTIS)
        if modo == "poco":
            par[:] = (pb0, pa0)
        elif modo == "global":
            par[:] = glob
        elif modo == "intervalo":
            for a in range(0, len(im), janela):
                bloco, ok = im[a:a + janela], ~excl[p][a:a + janela]
                par[a:a + janela] = np.percentile(bloco[ok], PERCENTIS) if ok.sum() >= MIN_LINHAS_VALIDAS else (pb0, pa0)
        else:
            raise ValueError(f"modo desconhecido: {modo}")
        saida[p] = par
    return saida


def aplicar_parametros(x_db, linha_ini, par):
    """Normaliza patches em dB (N, altura, colunas) com os percentis das linhas de cada patch.

    `linha_ini` dá a primeira linha de cada patch no poço; `par` vem de `parametros`.
    Devolve float32 depois de passar por float16, como as imagens normalizadas do treino.
    """
    n, altura = x_db.shape[:2]
    idx = np.asarray(linha_ini)[:, None] + np.arange(altura)[None, :]
    pb, pa = par[idx, 0][:, :, None], par[idx, 1][:, :, None]
    x = np.clip((x_db - pb) / (pa - pb), 0, 1)
    return x.astype(np.float16).astype(np.float32)
