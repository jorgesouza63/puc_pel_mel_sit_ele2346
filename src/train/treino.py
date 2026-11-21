"""Treino e avaliação (leave-one-well-out) dos modelos de segmentação."""
import copy
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset_patches import PAD, DatasetPatches, excluir_sobrepostos  # noqa: F401
from src.models.modelos import criar_modelo
from src.synth.sintetico import gerar_sintetico, resumo_sintetico

CFG_PADRAO = {"epocas": 10, "lr": 1e-3, "lote": 16, "paciencia": 4,
              "deslocar": True, "pretreino": True, "altura": 256, "val_cada": 2}


def fixar_semente(s):
    """Fixa as sementes aleatórias (random, numpy, torch e torch.cuda) em um só ponto."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def perda_dice_bce(logits, alvo):
    """Entropia cruzada binária mais perda de Dice (1 - Dice), média no lote."""
    bce = F.binary_cross_entropy_with_logits(logits, alvo)
    p = torch.sigmoid(logits)
    inter = (p * alvo).sum((1, 2, 3))
    dice = (2 * inter + 1) / (p.sum((1, 2, 3)) + alvo.sum((1, 2, 3)) + 1)
    return bce + (1 - dice).mean()


def _central(t):
    """Corta as colunas de preenchimento circular (192 → 180)."""
    return t[..., PAD:-PAD]


@torch.no_grad()
def dice_pseudo(modelo, ds, lote, dispositivo):
    """Dice médio entre a previsão (probabilidade > 0,5) e o pseudo-rótulo."""
    modelo.eval()
    vals = []
    for x, y in DataLoader(ds, batch_size=lote):
        p = (torch.sigmoid(_central(modelo(x.to(dispositivo)))) > 0.5).float()
        y = _central(y.to(dispositivo))
        inter = (p * y).sum((1, 2, 3))
        vals.append(((2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).cpu())
    return float(torch.cat(vals).mean())


def treinar(modelo, ds_treino, ds_val, cfg, semente, dispositivo):
    """Treina com Adam, aquecimento de 1 época e decaimento por cosseno.

    A parada antecipada usa o Dice contra o pseudo-rótulo no poço de validação.
    Devolve o histórico (DataFrame); o modelo fica com os pesos da melhor época.
    """
    fixar_semente(semente)
    modelo.to(dispositivo)
    gerador = torch.Generator().manual_seed(semente)
    dl = DataLoader(ds_treino, batch_size=cfg["lote"], shuffle=True, generator=gerador)
    total, aquec = cfg["epocas"] * len(dl), len(dl)
    opt = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / aquec if i < aquec
        else 0.5 * (1 + math.cos(math.pi * (i - aquec) / max(1, total - aquec))))
    usar_amp = dispositivo.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=usar_amp)
    hist, melhor, melhor_estado, sem_ganho = [], -1.0, None, 0
    for ep in range(1, cfg["epocas"] + 1):
        t0 = time.time()
        modelo.train()
        perdas = []
        for x, y in dl:
            x, y = x.to(dispositivo), y.to(dispositivo)
            with torch.autocast(device_type=dispositivo.type, enabled=usar_amp):
                logits = _central(modelo(x))
            perda = perda_dice_bce(logits.float(), _central(y))
            opt.zero_grad()
            scaler.scale(perda).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            perdas.append(float(perda.detach()))
        # Avaliando a validação a cada `val_cada` épocas e na última
        d = np.nan
        if ep % cfg["val_cada"] == 0 or ep == cfg["epocas"]:
            d = dice_pseudo(modelo, ds_val, cfg["lote"], dispositivo)
            if d > melhor:
                melhor, melhor_estado, sem_ganho = d, copy.deepcopy(modelo.state_dict()), 0
            else:
                sem_ganho += 1
        hist.append({"epoca": ep, "perda": float(np.mean(perdas)), "dice_val": d,
                     "segundos": round(time.time() - t0, 1)})
        if sem_ganho >= cfg["paciencia"]:
            break
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    return pd.DataFrame(hist)


@torch.no_grad()
def prever_pixels(modelo, x, lote, dispositivo):
    """Previsão binária (probabilidade > 0,5) para patches em array (N, H, 180); devolve bool."""
    modelo.eval()
    saida = []
    for i in range(0, len(x), lote):
        b = np.pad(x[i:i + lote], ((0, 0), (0, 0), (PAD, PAD)), mode="wrap")
        b = torch.from_numpy(b[:, None]).to(dispositivo)
        saida.append((torch.sigmoid(_central(modelo(b))) > 0.5)[:, 0].cpu().numpy())
    return np.concatenate(saida)


@torch.no_grad()
def escores_patches(modelo, ds, lote, dispositivo):
    """Escore de cada patch: probabilidade média (`score_soft`) e fração predita (`score`)."""
    modelo.eval()
    soft, frac = [], []
    for x, _ in DataLoader(ds, batch_size=lote):
        p = torch.sigmoid(_central(modelo(x.to(dispositivo))))
        soft += p.mean((1, 2, 3)).cpu().tolist()
        frac += (p > 0.5).float().mean((1, 2, 3)).cpu().tolist()
    return np.array(soft), np.array(frac)


def executar_fold(modelo_nome, poco_teste, poco_val, semente, imagens, indice_treino, anot,
                  limiares, cfg, dispositivo, indice_avaliacao=None, sint=None, imagens_rotulo=None):
    """Treina em 3 poços, valida em 1 e testa no poço restante.

    Devolve (tabela de escores dos patches anotados do poço de teste, resumo, histórico).
    O resumo traz o Dice contra o pseudo-rótulo no poço de teste (medida secundária).
    `indice_avaliacao` (padrão: `indice_treino`) dá os patches do poço de teste e os fundos do
    teste sintético; ablações que mudam só o treino (ex.: H5) mantêm a avaliação fixa.
    `sint` (padrão: None) é um dicionário poço -> (x, máscara, ignorar, meta) com o teste sintético
    já pronto (H3: teste comum às normalizações); sem ele, o teste é gerado a partir de `imagens`.
    `imagens_rotulo` (padrão: None) dá a imagem da qual sai o pseudo-rótulo (H9: imagem reconstruída
    por autoencoder); `limiares` deve ser o limiar dessa imagem. A entrada do modelo é sempre `imagens`.
    """
    indice_avaliacao = indice_treino if indice_avaliacao is None else indice_avaliacao
    t0 = time.time()
    pocos = list(imagens)
    treino = indice_treino[indice_treino.poco.isin([p for p in pocos if p not in (poco_teste, poco_val)])]
    val = indice_treino[indice_treino.poco == poco_val].iloc[::2]
    ds_tr = DatasetPatches(imagens, treino, limiares, cfg["altura"], cfg["deslocar"], semente, imagens_rotulo)
    ds_va = DatasetPatches(imagens, val, limiares, cfg["altura"], imagens_rotulo=imagens_rotulo)
    modelo = criar_modelo(modelo_nome, cfg["pretreino"])
    hist = treinar(modelo, ds_tr, ds_va, cfg, semente, dispositivo)

    # Avaliando nos patches anotados do poço de teste (sem dúvida)
    a = anot[(anot.poco == poco_teste) & anot.decisao.isin(["feicao_clara", "sem_feicao"])].copy()
    a["y"] = (a.decisao == "feicao_clara").astype(int)
    ds_an = DatasetPatches(imagens, a[["poco", "linha_ini"]], limiares, cfg["altura"], imagens_rotulo=imagens_rotulo)
    soft, frac = escores_patches(modelo, ds_an, cfg["lote"], dispositivo) if len(a) else ([], [])
    saida = a[["poco", "linha_ini", "y"]].assign(modelo=modelo_nome, semente=semente,
                                                 score_soft=soft, score=frac)

    # Medindo a concordância com o pseudo-rótulo no poço de teste (1 patch a cada 4)
    teste = indice_avaliacao[indice_avaliacao.poco == poco_teste].iloc[::4]
    d_pl = dice_pseudo(modelo, DatasetPatches(imagens, teste, limiares, cfg["altura"], imagens_rotulo=imagens_rotulo),
                       cfg["lote"], dispositivo)
    # Medindo a detecção de senoides sintéticas com máscara exata no poço de teste
    if sint is not None:
        xs, ms, ig, meta = sint[poco_teste]
    else:
        xs, ms, ig, meta = gerar_sintetico(imagens, indice_avaliacao, limiares[poco_teste], poco_teste,
                                           altura=cfg["altura"])
    res_sint = resumo_sintetico(prever_pixels(modelo, xs, cfg["lote"], dispositivo), ms, ig, meta)
    resumo = {"modelo": modelo_nome, "semente": semente, "poco_teste": poco_teste,
              "poco_val": poco_val, "epocas_feitas": len(hist),
              "melhor_dice_val": float(hist.dice_val.max()), "dice_pseudo_teste": d_pl,
              "segundos": round(time.time() - t0, 1), **res_sint}
    return saida, resumo, hist


def rodar_lowo(modelos, sementes, pocos, imagens, indice_treino, anot, limiares, cfg,
               dispositivo, pasta, indice_avaliacao=None, sint=None, imagens_rotulo=None):
    """Roda o leave-one-well-out para cada modelo e semente, salvando um arquivo por fold.

    Folds já salvos em `pasta` são pulados (a execução pode continuar após uma queda
    do Colab). Devolve (escores, resumos) de todos os folds.
    """
    os.makedirs(pasta, exist_ok=True)
    escores, resumos = [], []
    for m in modelos:
        for s in sementes:
            for k, teste in enumerate(pocos):
                val = pocos[(k + 1) % len(pocos)]
                base = f"{pasta}/{m}_s{s}_{teste}"
                if os.path.exists(f"{base}_resumo.csv"):
                    escores.append(pd.read_csv(f"{base}_escores.csv"))
                    resumos.append(pd.read_csv(f"{base}_resumo.csv"))
                    continue
                esc, res, hist = executar_fold(m, teste, val, s, imagens, indice_treino, anot,
                                               limiares, cfg, dispositivo, indice_avaliacao, sint, imagens_rotulo)
                esc.to_csv(f"{base}_escores.csv", index=False)
                hist.to_csv(f"{base}_historico.csv", index=False)
                pd.DataFrame([res]).to_csv(f"{base}_resumo.csv", index=False)
                escores.append(esc)
                resumos.append(pd.DataFrame([res]))
                print(f"{m} semente {s} teste {teste}: {res['segundos']} s, "
                      f"{res['epocas_feitas']} épocas, Dice pseudo {res['dice_pseudo_teste']:.3f}")
    return pd.concat(escores, ignore_index=True), pd.concat(resumos, ignore_index=True)
