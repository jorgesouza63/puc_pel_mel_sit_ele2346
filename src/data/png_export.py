"""Exporta PNGs para inspeção visual das imagens acústicas do WAID."""
import os

import numpy as np
import matplotlib.pyplot as plt


def _escala(v, pct):
    """Limites de cor robustos (percentis) calculados com 1 a cada 10 linhas."""
    vmin, vmax = np.nanpercentile(v[::10], pct)
    return float(vmin), float(vmax)


def salvar_png_poco(df, well_id, out_dir, linhas_por_png=1000, n_png=12,
                    pct=(1, 99), cmap="afmhot", faixas=4, dpi=100):
    """Salva os PNGs de inspeção de um poço.

    df             : DataFrame devolvido por load_well_image (linhas = profundidade)
    out_dir        : pasta base; os arquivos vão para out_dir/<well_id>/
    linhas_por_png : altura de cada trecho em resolução total (linhas)
    n_png          : quantidade de trechos, espaçados de forma uniforme no poço
    pct            : percentis da escala de cor (por poço)
    faixas         : número de faixas lado a lado na visão geral

    Gera 1 arquivo "visao_geral.png" e n_png arquivos "trecho_XX_<prof>m.png".
    Amplitude baixa aparece escura. O eixo horizontal é o índice da coluna
    (0 a 179), não graus, até o mapeamento de azimute ser confirmado.
    Retorna a lista de caminhos salvos.
    """
    v = df.values
    n, ncol = v.shape
    prof = df.index.values.astype("float64")
    vmin, vmax = _escala(v, pct)

    pasta = os.path.join(out_dir, well_id)
    os.makedirs(pasta, exist_ok=True)
    salvos = []

    # 1) visão geral: o poço inteiro em `faixas` colunas, média em blocos de linhas
    h = int(np.ceil(n / faixas))
    fator = max(1, int(np.ceil(h / 1500)))
    fig, eixos = plt.subplots(1, faixas, figsize=(3.2 * faixas, 9), sharey=False)
    for k, ax in enumerate(np.atleast_1d(eixos)):
        a, b = k * h, min((k + 1) * h, n)
        if a >= b:
            ax.axis("off")
            continue
        bloco = v[a:b]
        m = (len(bloco) // fator) * fator
        red = bloco[:m].reshape(-1, fator, ncol).mean(axis=1)
        ax.imshow(red, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        pos = np.linspace(0, len(red) - 1, 6)
        ax.set_yticks(pos)
        ax.set_yticklabels([f"{prof[a + int(p * fator)]:.0f}" for p in pos], fontsize=7)
        ax.set_xticks([0, ncol - 1])
        ax.set_xticklabels(["0", str(ncol - 1)], fontsize=7)
        ax.set_xlabel("coluna", fontsize=7)
    fig.suptitle(f"{well_id} - visão geral (profundidade em m; escala {pct[0]}-{pct[1]}%: "
                 f"{vmin:.1f} a {vmax:.1f} dB)", fontsize=10)
    fig.tight_layout()
    f_geral = os.path.join(pasta, "visao_geral.png")
    fig.savefig(f_geral, dpi=dpi)
    plt.close(fig)
    salvos.append(f_geral)

    # 2) trechos em resolução total, espaçados de forma uniforme
    if n <= linhas_por_png:
        inicios = [0]
    else:
        inicios = np.linspace(0, n - linhas_por_png, n_png).astype(int).tolist()
    for k, a in enumerate(inicios, start=1):
        b = min(a + linhas_por_png, n)
        fig, ax = plt.subplots(figsize=(3.2, 10))
        ax.imshow(v[a:b], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        pos = np.linspace(0, b - a - 1, 6)
        ax.set_yticks(pos)
        ax.set_yticklabels([f"{prof[a + int(p)]:.2f}" for p in pos], fontsize=7)
        ax.set_xticks([0, ncol - 1])
        ax.set_xticklabels(["0", str(ncol - 1)], fontsize=7)
        ax.set_xlabel("coluna")
        ax.set_ylabel("profundidade (m)")
        ax.set_title(f"{well_id}  linhas {a}-{b - 1}", fontsize=9)
        fig.tight_layout()
        f = os.path.join(pasta, f"trecho_{k:02d}_{prof[a]:.2f}m.png")
        fig.savefig(f, dpi=dpi)
        plt.close(fig)
        salvos.append(f)
    return salvos
