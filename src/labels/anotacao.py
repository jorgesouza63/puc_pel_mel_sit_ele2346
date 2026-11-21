"""Anotação simples por patch (critérios em docs/anotacao.md).

Para cada patch sorteado, o autor responde uma pergunta só:
  - "feicao_clara": há feição escura clara (fratura, cavidade ou breakout, pela ficha);
  - "sem_feicao": não há feição escura;
  - "duvida": o patch sai do conjunto.
A curva CAL aparece ao lado como evidência independente (alargamento do poço).
O código não pinta regiões nem sugere resposta, para não induzir a decisão.

O conjunto anotado serve só para validação e teste. Os patches anotados não
entram no treino.
"""
import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.labels.limiares import fracao_escura, limiar_otsu_poco  # noqa: F401 (reexportados)

DECISOES = ["feicao_clara", "sem_feicao", "duvida"]
ARQUIVO = "anotacoes.csv"
COLUNAS = ["rodada", "poco", "linha_ini", "prof_ini_m", "prof_fim_m", "decisao",
           "cal_max", "cal_z", "observacao", "data_hora", "decisao_original"]


def sortear_patches(indice, n_por_poco, semente=0):
    """Sorteia `n_por_poco` patches por poço, espalhados em faixas de profundidade.

    Usa só patches de índice par, para que dois patches sorteados nunca se sobreponham.
    """
    rng = np.random.default_rng(semente)
    partes = []
    for _, g in indice.groupby("poco"):
        g = g.sort_values("linha_ini").iloc[::2]
        faixas = np.array_split(np.arange(len(g)), n_por_poco)
        escolhidos = [int(rng.choice(f)) for f in faixas if len(f)]
        partes.append(g.iloc[escolhidos])
    return pd.concat(partes).reset_index(drop=True)


def _escolher(candidatos, n, rng, ocupados, altura):
    """Sorteia até `n` linhas de início sem sobreposição entre si nem com `ocupados`."""
    ocupados, escolhidos = list(ocupados), []
    for c in rng.permutation(candidatos):
        if len(escolhidos) == n:
            break
        if all(abs(int(c) - o) >= altura for o in ocupados):
            escolhidos.append(int(c))
            ocupados.append(int(c))
    return escolhidos


def sortear_segunda_rodada(indice, usados, n_por_poco=8, n_baixa=None, n_baixa_padrao=4,
                           quantil=0.3, semente=1, altura=256):
    """Segunda rodada: favorece patches com pouca área escura (mais chance de "sem feição").

    indice  : patches válidos, com a coluna `frac_escura`
    usados  : tabela com poco e linha_ini dos patches já sorteados na rodada anterior
    n_baixa : patches tirados do quantil mais baixo de `frac_escura`, por poço
              (os demais vêm do sorteio comum). Isso introduz viés de seleção, e o
              artigo deve declará-lo.
    Os patches novos não se sobrepõem aos usados nem entre si.
    """
    n_baixa = n_baixa or {}
    rng = np.random.default_rng(semente)
    partes = []
    for poco, g in indice.groupby("poco"):
        ocup = usados[usados.poco == poco].linha_ini.tolist()
        g = g.sort_values("linha_ini")
        corte = g.frac_escura.quantile(quantil)
        baixos = g[g.frac_escura <= corte].linha_ini.values
        k = min(n_baixa.get(poco, n_baixa_padrao), n_por_poco)
        esc_b = _escolher(baixos, k, rng, ocup, altura)
        esc_r = _escolher(g.linha_ini.values, n_por_poco - len(esc_b), rng, ocup + esc_b, altura)
        sel = g[g.linha_ini.isin(esc_b + esc_r)].copy()
        sel["origem"] = np.where(sel.linha_ini.isin(esc_b), "baixa_area_escura", "aleatorio")
        partes.append(sel)
    out = pd.concat(partes).reset_index(drop=True)
    out["rodada"] = 2
    return out


def carregar_patch(npy_dir, poco, linha_ini, altura=256):
    """Lê o patch normalizado (float32, valores de 0 a 1) e as profundidades das linhas."""
    norm = np.load(f"{npy_dir}/{poco}_norm.npy", mmap_mode="r")
    prof = np.load(f"{npy_dir}/{poco}_prof_rs.npy", mmap_mode="r")
    patch = np.asarray(norm[linha_ini:linha_ini + altura], dtype=np.float32)
    return patch, np.asarray(prof[linha_ini:linha_ini + altura], dtype=np.float64)


def ler_cal(bsc_dir, poco):
    """Lê a curva CAL do poço (DataFrame com DEPTH e CAL), ou None se não existir."""
    f = f"{bsc_dir}/{poco}_BSC.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, sep=";", decimal=",")
    return d[["DEPTH", "CAL"]].dropna() if "CAL" in d.columns else None


def resumo_cal(cal, prof):
    """CAL máximo no intervalo do patch e o desvio robusto dele em relação à mediana do poço.

    cal_z = (cal_max - mediana) / (1,4826 * MAD). É só informação; não é rótulo.
    Devolve (cal_max, cal_z), ou (nan, nan) sem CAL no intervalo.
    """
    if cal is None:
        return np.nan, np.nan
    sel = cal[(cal.DEPTH >= prof[0]) & (cal.DEPTH <= prof[-1])].CAL
    if len(sel) == 0:
        return np.nan, np.nan
    med = cal.CAL.median()
    mad = 1.4826 * (cal.CAL - med).abs().median()
    return float(sel.max()), (float((sel.max() - med) / mad) if mad > 0 else np.nan)


def anotados(ann_dir, arquivo=ARQUIVO):
    """Conjunto de (poço, linha_ini) já decididos no arquivo."""
    f = f"{ann_dir}/{arquivo}"
    if not os.path.exists(f):
        return set()
    d = pd.read_csv(f)
    return set(zip(d.poco, d.linha_ini))


def salvar_anotacao(ann_dir, linha, decisao, cal_max, cal_z, observacao="", arquivo=ARQUIVO):
    """Atualiza a tabela de anotações (1 linha por patch; repetir o patch substitui a linha)."""
    if decisao not in DECISOES:
        raise ValueError(f"decisão inválida: {decisao}")
    os.makedirs(ann_dir, exist_ok=True)
    reg = {"rodada": int(linha["rodada"]) if "rodada" in linha else 1, "poco": linha["poco"], "linha_ini": int(linha["linha_ini"]),
           "prof_ini_m": linha["prof_ini_m"], "prof_fim_m": linha["prof_fim_m"],
           "decisao": decisao, "cal_max": cal_max, "cal_z": cal_z,
           "observacao": observacao,
           "data_hora": datetime.datetime.now().isoformat(timespec="seconds")}
    f = f"{ann_dir}/{arquivo}"
    tab = _ler(f)
    tab = tab[~((tab.poco == reg["poco"]) & (tab.linha_ini == reg["linha_ini"]))]
    tab = pd.concat([tab, pd.DataFrame([reg])], ignore_index=True)[COLUNAS]
    tab.to_csv(f, index=False)


def _ler(f):
    """Lê a tabela de anotações; completa colunas que faltam em arquivos antigos."""
    if not os.path.exists(f):
        return pd.DataFrame(columns=COLUNAS)
    tab = pd.read_csv(f, dtype={"observacao": str, "decisao_original": str})
    if "rodada" not in tab.columns:
        tab.insert(0, "rodada", 1)
    if "decisao_original" not in tab.columns:
        tab["decisao_original"] = np.nan
    tab["decisao_original"] = tab["decisao_original"].astype(object)
    tab["observacao"] = tab["observacao"].astype(object)
    return tab


def revisar_decisao(ann_dir, poco, linha_ini, nova_decisao, motivo, arquivo=ARQUIVO):
    """Muda a decisão de um patch, guardando a original e o motivo (para o registro)."""
    if nova_decisao not in DECISOES:
        raise ValueError(f"decisão inválida: {nova_decisao}")
    f = f"{ann_dir}/{arquivo}"
    tab = _ler(f)
    m = (tab.poco == poco) & (tab.linha_ini == linha_ini)
    if m.sum() != 1:
        raise ValueError(f"patch não encontrado: {poco} {linha_ini}")
    if (tab.loc[m, "decisao"] == nova_decisao).all() and tab.loc[m, "decisao_original"].notna().all():
        return   # revisão já aplicada; não sobrescreve o motivo
    if pd.isna(tab.loc[m, "decisao_original"]).all():
        tab.loc[m, "decisao_original"] = tab.loc[m, "decisao"].values[0]
    tab.loc[m, "decisao"] = nova_decisao
    tab.loc[m, "observacao"] = motivo
    tab[COLUNAS].to_csv(f, index=False)


def sortear_reanotacao(tab, n_por_poco=None, semente=2):
    """Escolhe patches para reanotar às cegas e embaralha a ordem.

    `n_por_poco` dá o número de patches por poço (ex.: {"tatu22": 4, "outros": 6}).
    Só entram patches decididos como "feicao_clara" ou "sem_feicao". Nos poços
    "outros", metade vem de cada decisão, quando possível.
    """
    rng = np.random.default_rng(semente)
    n_por_poco = n_por_poco or {"tatu22": 4, "outros": 6}
    ok = tab[tab.decisao.isin(["feicao_clara", "sem_feicao"])]
    escolhidos = []
    for poco, n in n_por_poco.items():
        g = ok[ok.poco == poco] if poco != "outros" else ok[~ok.poco.isin(n_por_poco)]
        if poco == "outros":
            metade = n // 2
            for dec, k in [("sem_feicao", metade), ("feicao_clara", n - metade)]:
                h = g[g.decisao == dec]
                escolhidos.append(h.iloc[rng.choice(len(h), min(k, len(h)), replace=False)])
        else:
            escolhidos.append(g.iloc[rng.choice(len(g), min(n, len(g)), replace=False)])
    sel = pd.concat(escolhidos)[["poco", "linha_ini", "prof_ini_m", "prof_fim_m"]]
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    sel["rodada"] = 3
    return sel


def concordancia(ann_dir, arquivo_original=ARQUIVO, arquivo_nova="reanotacao.csv"):
    """Compara a reanotação com a decisão original (só feicao_clara e sem_feicao).

    Devolve (n patches, concordância simples, kappa de Cohen, tabela de decisões).
    """
    a = _ler(f"{ann_dir}/{arquivo_original}")
    b = _ler(f"{ann_dir}/{arquivo_nova}")
    # Usando a decisão original, se o patch foi revisado pela ficha
    a["dec_a"] = a.decisao_original.fillna(a.decisao)
    m = a[["poco", "linha_ini", "dec_a"]].merge(b[["poco", "linha_ini", "decisao"]],
                                                on=["poco", "linha_ini"])
    tabela = pd.crosstab(m.dec_a, m.decisao)
    m = m[m.dec_a.isin(["feicao_clara", "sem_feicao"]) & m.decisao.isin(["feicao_clara", "sem_feicao"])]
    n = len(m)
    if n == 0:
        return 0, np.nan, np.nan, tabela
    po = float((m.dec_a == m.decisao).mean())
    pe = sum(float((m.dec_a == c).mean()) * float((m.decisao == c).mean())
             for c in ["feicao_clara", "sem_feicao"])
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return n, po, kappa, tabela


LEMBRETE = """<div style='font-size:90%'><b>Lembrete da ficha (docs/anotacao.md)</b><br>
<b>Feição clara:</b> par de zonas escuras a ~180° e alongadas na vertical (<i>breakout</i>); traço senoidal contínuo (fratura); mancha escura isolada com borda definida (cavidade).<br>
<b>Sem feição:</b> rocha sem zona escura clara.<br>
<b>Dúvida:</b> faixas diagonais, linhas verticais finas, zona escura extensa, qualquer incerteza.</div>"""


class Anotador:
    """Interface de anotação no Colab (ipywidgets)."""

    def __init__(self, sorteio, npy_dir, ann_dir, bsc_dir, altura=256, arquivo=ARQUIVO):
        import ipywidgets as w
        self.arquivo = arquivo
        self.sorteio, self.npy_dir, self.ann_dir = sorteio, npy_dir, ann_dir
        self.bsc_dir, self.altura = bsc_dir, altura
        self._cal = {}
        self.i = -1
        self.out = w.Output()
        self.titulo = w.HTML()
        self.obs = w.Text(description="obs.", layout=w.Layout(width="420px"))
        b_feicao = w.Button(description="Feição escura clara", button_style="success")
        b_sem = w.Button(description="Sem feição", button_style="info")
        b_duv = w.Button(description="Dúvida (patch sai)", button_style="warning")
        b_pular = w.Button(description="Pular")
        b_feicao.on_click(lambda _: self.salvar("feicao_clara"))
        b_sem.on_click(lambda _: self.salvar("sem_feicao"))
        b_duv.on_click(lambda _: self.salvar("duvida"))
        b_pular.on_click(lambda _: self.proximo())
        self.ui = w.VBox([w.HTML(LEMBRETE), self.titulo, self.out, self.obs,
                          w.HBox([b_feicao, b_sem, b_duv, b_pular])])

    def _curva(self, poco):
        if poco not in self._cal:
            self._cal[poco] = ler_cal(self.bsc_dir, poco)
        return self._cal[poco]

    def proximo(self):
        feitos = anotados(self.ann_dir, self.arquivo)
        for k in range(self.i + 1, len(self.sorteio)):
            r = self.sorteio.iloc[k]
            if (r.poco, r.linha_ini) not in feitos:
                self.i = k
                return self.carregar()
        from IPython.display import clear_output
        self.titulo.value = "<b>Fim da lista.</b> Todos os patches sorteados foram decididos."
        with self.out:
            clear_output()

    def carregar(self):
        r = self.sorteio.iloc[self.i]
        self.patch, self.prof = carregar_patch(self.npy_dir, r.poco, int(r.linha_ini), self.altura)
        self.cal_max, self.cal_z = resumo_cal(self._curva(r.poco), self.prof)
        self.obs.value = ""
        self.titulo.value = (f"<b>Patch {self.i + 1} de {len(self.sorteio)}</b> · {r.poco} · "
                             f"linhas {int(r.linha_ini)}–{int(r.linha_ini) + self.altura - 1} · "
                             f"{r.prof_ini_m:.2f}–{r.prof_fim_m:.2f} m")
        self.desenhar()

    def desenhar(self):
        from IPython.display import clear_output
        r = self.sorteio.iloc[self.i]
        cal = self._curva(r.poco)
        fig, ax = plt.subplots(1, 2, figsize=(7, 7), gridspec_kw={"width_ratios": [1, 0.45]})
        ax[0].imshow(self.patch, aspect="auto", cmap="afmhot", vmin=0, vmax=1,
                     extent=[0, self.patch.shape[1], self.prof[-1], self.prof[0]])
        ax[0].set_xlabel("coluna")
        ax[0].set_ylabel("profundidade (m)")
        if cal is not None:
            sel = cal[(cal.DEPTH >= self.prof[0] - 0.5) & (cal.DEPTH <= self.prof[-1] + 0.5)]
            if len(sel):
                ax[1].plot(sel.CAL, sel.DEPTH, "o-")
            ax[1].axvline(cal.CAL.median(), color="gray", ls="--", lw=1)
        ax[1].set_ylim(self.prof[-1], self.prof[0])
        z = "" if np.isnan(self.cal_z) else f"\nCAL máx {self.cal_max:.1f}, z {self.cal_z:.1f}"
        ax[1].set_title(("CAL (tracejado: mediana do poço)" if cal is not None else "sem CAL") + z,
                        fontsize=8)
        fig.tight_layout()
        with self.out:
            clear_output(wait=True)
            plt.show()
        plt.close(fig)

    def salvar(self, decisao):
        r = self.sorteio.iloc[self.i]
        salvar_anotacao(self.ann_dir, r, decisao, self.cal_max, self.cal_z, self.obs.value, self.arquivo)
        self.proximo()


def abrir_interface(sorteio, npy_dir, ann_dir, bsc_dir, altura=256, arquivo=ARQUIVO):
    """Abre a interface e mostra o primeiro patch ainda não decidido."""
    from IPython.display import display
    a = Anotador(sorteio, npy_dir, ann_dir, bsc_dir, altura, arquivo)
    display(a.ui)
    a.proximo()
    return a
