"""Conjunto de patches para treino e avaliação (pseudo-rótulo de limiar, aumento circular)."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD = 6   # colunas copiadas de cada lado (azimute circular): 180 + 2 × 6 = 192


def carregar_imagens(npy_dir, pocos):
    """Carrega as imagens normalizadas (float16) de todos os poços na memória (cerca de 160 MB)."""
    return {p: np.load(f"{npy_dir}/{p}_norm.npy") for p in pocos}


def excluir_sobrepostos(indice, anot, altura=256):
    """Tira do índice os patches que se sobrepõem a algum patch anotado (inclusive dúvida)."""
    manter = np.ones(len(indice), dtype=bool)
    ini = indice.linha_ini.values
    for poco, g in anot.groupby("poco"):
        m = (indice.poco == poco).values
        for a in g.linha_ini.values:
            manter &= ~(m & (np.abs(ini - a) < altura))
    return indice[manter].reset_index(drop=True)


def indice_sem_exclusao(imagens, anot, altura=256, passo=128):
    """Índice de patches que também cobre as zonas excluídas pela D1 (ablação H5).

    Usa todas as posições (passo de 128 linhas) e tira só os patches que se sobrepõem aos
    anotados. Serve para treinar e validar; a avaliação continua nos patches válidos.
    """
    linhas = [(p, i) for p, im in imagens.items() for i in range(0, len(im) - altura + 1, passo)]
    return excluir_sobrepostos(pd.DataFrame(linhas, columns=["poco", "linha_ini"]), anot, altura)


def preparar_dados(npy_dir, carac_dir, anot_dir, pocos, altura=256):
    """Carrega imagens, limiar de Otsu por poço, índice de treino (sem sobreposição) e anotações.

    Devolve (imagens, limiares, indice_treino, anot). Usa a máscara de exclusão da D1.
    """
    from src.labels.limiares import limiar_otsu_poco
    imagens = carregar_imagens(npy_dir, pocos)
    limiares = {p: limiar_otsu_poco(imagens[p], np.load(f"{npy_dir}/{p}_excluir_rs.npy")) for p in pocos}
    anot = pd.read_csv(f"{anot_dir}/anotacoes.csv")
    indice = pd.read_csv(f"{carac_dir}/patches_indice.csv")
    return imagens, limiares, excluir_sobrepostos(indice, anot, altura), anot


class DatasetPatches(Dataset):
    """Patches de 256 × 180 com pseudo-rótulo (pixel < limiar do poço = feição escura).

    Devolve (x, y) com forma (1, 256, 192): as 180 colunas mais 6 colunas circulares
    de cada lado. Com `deslocar`, cada patch recebe um deslocamento circular aleatório
    no azimute (aumento de dados; ablação H4). Com `imagens_rotulo` (ablação H9), o
    pseudo-rótulo vem dessa outra imagem (por exemplo, a reconstruída por autoencoder),
    com o mesmo deslocamento; a entrada x continua a imagem normalizada.
    """

    def __init__(self, imagens, tabela, limiares, altura=256, deslocar=False, semente=0, imagens_rotulo=None):
        self.imagens, self.tabela = imagens, tabela.reset_index(drop=True)
        self.limiares, self.altura, self.deslocar = limiares, altura, deslocar
        self.imagens_rotulo = imagens_rotulo
        self.rng = np.random.default_rng(semente)

    def __len__(self):
        return len(self.tabela)

    def patch(self, i, imagens=None):
        """Patch i, sem deslocamento e sem preenchimento (float32, 256 × 180)."""
        imagens = self.imagens if imagens is None else imagens
        r = self.tabela.iloc[i]
        return np.asarray(imagens[r.poco][int(r.linha_ini):int(r.linha_ini) + self.altura],
                          dtype=np.float32)

    def __getitem__(self, i):
        x = self.patch(i)
        desloc = int(self.rng.integers(0, x.shape[1])) if self.deslocar else 0
        if self.deslocar:
            x = np.roll(x, desloc, axis=1)
        if self.imagens_rotulo is None:
            fonte = x
        else:
            fonte = np.roll(self.patch(i, self.imagens_rotulo), desloc, axis=1)
        y = (fonte < self.limiares[self.tabela.iloc[i].poco]).astype(np.float32)
        x = np.pad(x, ((0, 0), (PAD, PAD)), mode="wrap")
        y = np.pad(y, ((0, 0), (PAD, PAD)), mode="wrap")
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])
