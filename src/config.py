"""Caminhos e constantes compartilhados pelos notebooks (evita repetir definições)."""

RAIZ = "/content/drive/MyDrive/waid"

POCOS = ["antilope25", "antilope37", "botorosa47", "coala88", "tatu22"]

IMG_DIR = f"{RAIZ}/data/raw/img"
BSC_DIR = f"{RAIZ}/data/raw/bsc"
PNG_DIR = f"{RAIZ}/data/processed/png"
NPY_DIR = f"{RAIZ}/data/processed/npy"
CARAC_DIR = f"{RAIZ}/data/processed/caracterizacao"
ANOT_DIR = f"{RAIZ}/data/annotations"
RES_DIR = f"{RAIZ}/results"

# Decisão D4: patches de 256 linhas, com as 180 colunas
ALTURA_PATCH = 256
