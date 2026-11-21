"""Pseudo-rótulo a partir da imagem reconstruída por autoencoder (ablação H9).

Um autoencoder por poço, treinado só na imagem do próprio poço (simplificação do autor: Silva (2026)
treina por intervalo de profundidade). A reconstrução une os patches com uma janela de Hann 2D com piso
de 0,05. O limiar é o Otsu binário da imagem reconstruída, nos pixels válidos (fora da zona D1).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.labels.limiares import limiar_otsu_poco
from src.models.autoencoder import TAM, AutoencoderRuido
from src.train.treino import fixar_semente

PASSO = 8
SIGMA = 0.05
EPS_HANN = 0.05


def _posicoes(n, tam=TAM, passo=PASSO):
    """Posições iniciais dos patches ao longo de um eixo; inclui a última, para cobrir a borda."""
    pos = list(range(0, n - tam + 1, passo))
    if pos[-1] != n - tam:
        pos.append(n - tam)
    return np.array(pos)


def coordenadas_validas(excl, largura, tam=TAM, passo=PASSO):
    """Coordenadas (linha, coluna) dos patches sem nenhuma linha excluída (D1). `excl` é booleano por linha."""
    ac = np.concatenate([[0], np.cumsum(excl.astype(np.int64))])
    linhas = np.array([r for r in _posicoes(len(excl), tam, passo) if ac[r + tam] - ac[r] == 0])
    colunas = _posicoes(largura, tam, passo)
    rr, cc = np.meshgrid(linhas, colunas, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1)


def _lote(img_t, coords, ar):
    """Patches (B, 1, 32, 32) lidos de uma imagem em tensor, sem montar todos na memória."""
    r = coords[:, 0:1, None] + ar[None, :, None]
    c = coords[:, 1:2, None] + ar[None, None, :]
    return img_t[r, c][:, None]


def treinar_ae(img, excl, dispositivo, pasta_ckpt, epocas=60, lote=128, lr=1e-3, semente=0,
               max_epocas_agora=None, verboso=True):
    """Treina o autoencoder de um poço, com ponto de controle por época (retoma se o Colab cair).

    `max_epocas_agora` limita as épocas desta chamada (teste de tempo). Devolve (modelo, histórico).
    """
    fixar_semente(semente)
    os.makedirs(pasta_ckpt, exist_ok=True)
    ck = f"{pasta_ckpt}/ae.pt"
    modelo = AutoencoderRuido().to(dispositivo)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    hist, ep0 = [], 0
    if os.path.exists(ck):
        est = torch.load(ck, map_location=dispositivo)
        modelo.load_state_dict(est["modelo"])
        opt.load_state_dict(est["opt"])
        hist, ep0 = est["hist"], est["epoca"]
    coords = torch.from_numpy(coordenadas_validas(excl, img.shape[1])).to(dispositivo)
    img_t = torch.from_numpy(np.asarray(img, dtype=np.float32)).to(dispositivo)
    ar = torch.arange(TAM, device=dispositivo)
    gerador = torch.Generator(device="cpu").manual_seed(semente + ep0)
    fim = epocas if max_epocas_agora is None else min(epocas, ep0 + max_epocas_agora)
    for ep in range(ep0 + 1, fim + 1):
        t0 = time.time()
        modelo.train()
        ordem = torch.randperm(len(coords), generator=gerador).to(dispositivo)
        soma, n = 0.0, 0
        for i in range(0, len(ordem), lote):
            x = _lote(img_t, coords[ordem[i:i + lote]], ar)
            rec = modelo(x + SIGMA * torch.randn_like(x))
            perda = ((rec - x) ** 2).sum((1, 2, 3)).mean()   # equação 5 de Silva (2026)
            opt.zero_grad()
            perda.backward()
            opt.step()
            soma += float(perda.detach()) * len(x)
            n += len(x)
        hist.append({"epoca": ep, "perda": soma / n, "segundos": round(time.time() - t0, 1)})
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "epoca": ep, "hist": hist}, ck)
        if verboso:
            print(f"  época {ep}/{epocas}: perda {soma / n:.3f}, {hist[-1]['segundos']} s, {n} patches")
    return modelo, pd.DataFrame(hist)


@torch.no_grad()
def reconstruir(modelo, img, dispositivo, lote=512):
    """Imagem reconstruída (float16, mesma forma): média dos patches ponderada por janela de Hann com piso."""
    modelo.eval()
    h, w = img.shape
    img_t = torch.from_numpy(np.asarray(img, dtype=np.float32)).to(dispositivo)
    ar = torch.arange(TAM, device=dispositivo)
    hann = torch.from_numpy(np.hanning(TAM).astype(np.float32)).to(dispositivo)
    janela = torch.clamp(hann[:, None] * hann[None, :], min=EPS_HANN)
    num = torch.zeros(h, w, device=dispositivo)
    den = torch.zeros(h, w, device=dispositivo)
    rr, cc = np.meshgrid(_posicoes(h), _posicoes(w), indexing="ij")
    coords = torch.from_numpy(np.stack([rr.ravel(), cc.ravel()], axis=1)).to(dispositivo)
    for i in range(0, len(coords), lote):
        co = coords[i:i + lote]
        rec = modelo(_lote(img_t, co, ar))[:, 0] * janela
        r = co[:, 0:1, None] + ar[None, :, None]
        c = co[:, 1:2, None] + ar[None, None, :]
        num.index_put_((r.expand_as(rec), c.expand_as(rec)), rec, accumulate=True)
        den.index_put_((r.expand_as(rec), c.expand_as(rec)), janela.expand_as(rec), accumulate=True)
    return (num / den).cpu().numpy().astype(np.float16)


def fracao_escura_poco(img, excl, limiar):
    """Fração de pixels abaixo do limiar nas linhas válidas do poço."""
    return float((np.asarray(img[~excl], dtype=np.float32) < limiar).mean())


def gerar_rotulos_ae(imagens, excl, pocos, dispositivo, pasta, epocas=60, **kw):
    """Treina um autoencoder por poço, reconstrói a imagem e calcula o Otsu binário dela.

    Salva `{pasta}/{poco}_rec.npy` e `limiares_ae.json`; poços prontos são pulados.
    Devolve (imagens reconstruídas, limiares, tabela de diagnóstico).
    """
    os.makedirs(pasta, exist_ok=True)
    rec, lim, linhas = {}, {}, []
    for p in pocos:
        arq = f"{pasta}/{p}_rec.npy"
        if os.path.exists(arq):
            rec[p] = np.load(arq)
        else:
            print(p)
            modelo, _ = treinar_ae(imagens[p], excl[p], dispositivo, f"{pasta}/{p}", epocas=epocas, **kw)
            rec[p] = reconstruir(modelo, imagens[p], dispositivo)
            np.save(arq, rec[p])
        lim[p] = limiar_otsu_poco(rec[p], excl[p])
        l_bruto = limiar_otsu_poco(imagens[p], excl[p])
        linhas.append({"poco": p, "limiar_bruta": round(l_bruto, 3), "limiar_reconstruida": round(lim[p], 3),
                       "fracao_escura_bruta": round(fracao_escura_poco(imagens[p], excl[p], l_bruto), 3),
                       "fracao_escura_reconstruida": round(fracao_escura_poco(rec[p], excl[p], lim[p]), 3),
                       "eqm_reconstrucao": round(float(((np.asarray(imagens[p][~excl[p]], dtype=np.float32)
                                                        - rec[p][~excl[p]].astype(np.float32)) ** 2).mean()), 5)})
    with open(f"{pasta}/limiares_ae.json", "w") as f:
        json.dump(lim, f)
    return rec, lim, pd.DataFrame(linhas)
