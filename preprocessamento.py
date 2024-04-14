import torch
import os
from torchvision import datasets, transforms

# Caminho para a pasta de imagens de treinamento
caminho_pasta = "X:/pablo dev/GITHUB/projeto_gan/imagens_treinamento"

# Tamanho da imagem a ser redimensionada
image_size = 64

# Transformações para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Carrega as imagens de treinamento
dataset_treinamento = datasets.ImageFolder(root=caminho_pasta, transform=transform)

# Define o tamanho do lote (batch size) e cria um DataLoader para iterar sobre os dados de treinamento
tamanho_lote = 64
dataloader_treinamento = torch.utils.data.DataLoader(dataset_treinamento, batch_size=tamanho_lote, shuffle=True)
