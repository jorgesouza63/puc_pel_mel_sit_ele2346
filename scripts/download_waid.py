#!/usr/bin/env python3
"""Baixa o dataset WAID (Petrobras) direto do GitHub para uma pasta (ex.: Google Drive).

Uso no Colab:
    !python scripts/download_waid.py --dest /content/drive/MyDrive/waid/data/raw

Opções úteis:
    --wells tatu22 coala88   baixa só alguns poços
    --no-img                 baixa só os perfis básicos (BSC, ~1 MB)
    --dry-run                lista o que baixaria, sem baixar

Características:
    - Fixa um commit do repositório (reprodutibilidade do artigo).
    - Pula arquivos já baixados (compara o tamanho com o do servidor).
    - Grava em arquivo .part e renomeia no fim (sem arquivo pela metade).
    - Tenta de novo em caso de falha de rede.

Dados: WAID/PETROBRAS, licença CC BY 4.0 (https://github.com/petrobras/waid).
"""
import argparse
import os
import sys
import time
import urllib.request

# Commit verificado em 2026-10-31. Troque por outro SHA só de forma consciente.
COMMIT = "f8926d060cac24cdee37563546db85daf06a2859"
BASE = f"https://raw.githubusercontent.com/petrobras/waid/f8926d060cac24cdee37563546db85daf06a2859/dataset"

# poço -> número de subarquivos AMP (00 .. n-1)
WELLS = {
    "antilope25": 24,
    "antilope37": 10,
    "botorosa47": 9,
    "coala88": 10,
    "tatu22": 9,
}


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(url, path, retries=4):
    size = remote_size(url)
    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"  ok (já existe)  {os.path.basename(path)}")
        return size
    part = path + ".part"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(part) != size:
                raise IOError("tamanho diferente do esperado")
            os.replace(part, path)
            print(f"  baixado         {os.path.basename(path)}  ({size/1e6:.1f} MB)")
            return size
        except Exception as e:  # noqa: BLE001
            print(f"  falha {attempt}/{retries}: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise SystemExit(f"Não foi possível baixar {url}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="pasta de destino (ex.: .../waid/data/raw)")
    ap.add_argument("--wells", nargs="*", default=list(WELLS), choices=list(WELLS))
    ap.add_argument("--no-img", action="store_true", help="não baixa as imagens AMP")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs = []  # (url, caminho local)
    for w in a.wells:
        jobs.append((f"{BASE}/bsc/{w}_BSC.csv", os.path.join(a.dest, "bsc", f"{w}_BSC.csv")))
        if not a.no_img:
            for i in range(WELLS[w]):
                n = f"{w}_AMP{i:02d}.csv"
                jobs.append((f"{BASE}/img/{n}", os.path.join(a.dest, "img", n)))

    print(f"Commit: f8926d060cac24cdee37563546db85daf06a2859\nArquivos: {len(jobs)}\nDestino: {a.dest}")
    if a.dry_run:
        for url, _ in jobs:
            print("  ", url)
        return
    os.makedirs(os.path.join(a.dest, "bsc"), exist_ok=True)
    os.makedirs(os.path.join(a.dest, "img"), exist_ok=True)
    total = sum(download(u, p) for u, p in jobs)
    print(f"\nPronto. {len(jobs)} arquivos, {total/1e9:.2f} GB.")


if __name__ == "__main__":
    main()
