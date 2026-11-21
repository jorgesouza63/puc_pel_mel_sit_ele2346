"""Limiares de amplitude por poço (Otsu, Multi-Otsu e K-means) para os baselines.

Todos os limiares separam pixels escuros (amplitude baixa) dos demais, sobre a
imagem normalizada (decisão D3). Usam 1 a cada `passo` linhas válidas do poço.
"""
import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu, threshold_otsu
from sklearn.cluster import KMeans

METODOS = ["otsu", "multiotsu3", "kmeans2"]


def valores_poco(norm, excl, passo=10):
    """Valores (float32, 1D) de 1 a cada `passo` linhas válidas do poço."""
    return np.asarray(norm[~excl][::passo], dtype=np.float32).ravel()


def limiar_otsu_poco(norm, excl, passo=10):
    """Limiar de Otsu do poço (2 classes)."""
    return float(threshold_otsu(valores_poco(norm, excl, passo)))


def limiar_multiotsu_poco(norm, excl, classes=3, passo=10):
    """Limiar entre a classe mais escura e as demais, no Multi-Otsu com `classes` classes."""
    return float(threshold_multiotsu(valores_poco(norm, excl, passo), classes=classes)[0])


def limiar_kmeans_poco(norm, excl, k=2, passo=10, n_amostra=200_000, semente=0):
    """Limiar a meio caminho entre os dois menores centros do K-means com `k` grupos."""
    v = valores_poco(norm, excl, passo)
    rng = np.random.default_rng(semente)
    if len(v) > n_amostra:
        v = v[rng.choice(len(v), n_amostra, replace=False)]
    centros = np.sort(KMeans(n_clusters=k, n_init=10, random_state=semente)
                      .fit(v.reshape(-1, 1)).cluster_centers_.ravel())
    return float((centros[0] + centros[1]) / 2)


def calcular_limiares(npy_dir, pocos):
    """Tabela poço × método com os limiares (um por poço e método)."""
    linhas = []
    for p in pocos:
        norm = np.load(f"{npy_dir}/{p}_norm.npy", mmap_mode="r")
        excl = np.load(f"{npy_dir}/{p}_excluir_rs.npy")
        linhas.append({"poco": p,
                       "otsu": limiar_otsu_poco(norm, excl),
                       "multiotsu3": limiar_multiotsu_poco(norm, excl),
                       "kmeans2": limiar_kmeans_poco(norm, excl)})
    return pd.DataFrame(linhas).set_index("poco")


def fracao_escura(npy_dir, poco, linhas_ini, limiar, altura=256):
    """Fração de pixels abaixo do limiar em cada patch (uma entrada por `linhas_ini`)."""
    norm = np.load(f"{npy_dir}/{poco}_norm.npy", mmap_mode="r")
    return np.array([float((np.asarray(norm[i:i + altura], dtype=np.float32) < limiar).mean())
                     for i in linhas_ini])
