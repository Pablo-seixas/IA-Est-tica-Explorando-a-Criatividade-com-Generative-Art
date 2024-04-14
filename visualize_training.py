import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
from meu_gerador import Gerador
import torch.nn as nn

# meu_gerador.py

import torch.nn as nn

class Gerador(nn.Module):
    def __init__(self):
        super(Gerador, self).__init__()
        # Aqui você define a arquitetura do seu gerador
        # Exemplo simplificado:
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# Em algum lugar do código, crie uma instância do Gerador
gerador = Gerador()

# Definir o tamanho do batch e a dimensão do vetor de ruído
batch_size = 64
noise_dim = 100

# Lista para armazenar as imagens geradas em diferentes épocas
generated_images = []

# Defina o número de épocas desejado
num_epochs = 10  # Por exemplo, 10 épocas

# Laço de treinamento
for epoch in range(num_epochs):
    # Gerar ruído aleatório
    noise = torch.randn(batch_size, noise_dim, 1, 1)

    # Verificar se é uma época desejada para salvar imagens geradas
    if epoch in [0, 1, 2, 3, 4]:
        # Gerar imagens com o gerador
        with torch.no_grad():
            # Substitua 'gerador' pelo objeto que gera as imagens
            fake = gerador(noise).detach().cpu()  
        # Adicionar imagens à lista
        generated_images.append(vutils.make_grid(fake, padding=2, normalize=True))

# Criar diretório para salvar as imagens, se não existir
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)

# Salvar imagens geradas em diferentes épocas
for i, img_tensor in enumerate(generated_images):
    vutils.save_image(img_tensor, f'{output_dir}/epoch_{i}.png')

# Visualizar as imagens geradas
fig = plt.figure(figsize=(10, 10))
for i, img_path in enumerate(sorted(os.listdir(output_dir))):
    img = plt.imread(os.path.join(output_dir, img_path))
    fig.add_subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
plt.show()
