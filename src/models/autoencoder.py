"""Autoencoder convolucional de remoção de ruído (ablação H9), como descrito por Silva (2026).

Patches de 32 × 32; codificador com 3 convoluções de passo 2 (canais 1 → 16 → 32 → 64), achatamento e
projeção para um vetor latente de 64; decodificador espelhado (camada densa e 3 convoluções transpostas
64 → 32 → 16 → 1) com saída sigmoide. A função de ativação intermediária não é dita na fonte; foi
adotada a ReLU (escolha do autor).
"""
import torch
from torch import nn

TAM = 32       # lado do patch
LATENTE = 64   # dimensão do vetor latente


class AutoencoderRuido(nn.Module):
    def __init__(self, latente=LATENTE):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 4 * 4, latente))
        self.dec_fc = nn.Sequential(nn.Linear(latente, 64 * 4 * 4), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        z = self.enc(x)
        return self.dec(self.dec_fc(z).view(-1, 64, 4, 4))
