"""Modelos de segmentação: CNN rasa (baseline) e U-Net com encoder ResNet18."""
from torch import nn


class CNNRasa(nn.Module):
    """CNN totalmente convolucional, sem redução de resolução (baseline).

    Quatro camadas de convolução 3 × 3 com dilatação (1, 2, 4, 1), normalização
    em lote e ReLU, e uma convolução 1 × 1 final. Cada pixel "enxerga" uma janela
    de 17 × 17 pixels. Saída: um logit por pixel (feição escura).
    """

    def __init__(self, canais=(16, 16, 32, 32), dilatacoes=(1, 2, 4, 1)):
        super().__init__()
        camadas, c_in = [], 1
        for c, d in zip(canais, dilatacoes):
            camadas += [nn.Conv2d(c_in, c, 3, padding=d, dilation=d, bias=False),
                        nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_in = c
        camadas.append(nn.Conv2d(c_in, 1, 1))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)


def criar_modelo(nome, pretreino=True):
    """Cria o modelo pelo nome: "cnn_rasa" ou "unet_resnet18".

    A U-Net usa 1 canal de entrada e 1 classe de saída (logit). Com `pretreino`,
    o encoder ResNet18 começa com os pesos do ImageNet (a ablação H7 testa sem).
    """
    if nome == "cnn_rasa":
        return CNNRasa()
    if nome == "unet_resnet18":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name="resnet18", in_channels=1, classes=1,
                        encoder_weights="imagenet" if pretreino else None)
    raise ValueError(f"modelo desconhecido: {nome}")
