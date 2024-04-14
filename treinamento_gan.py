import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from modelo_gan import Gerador

# Definir hiperparâmetros
batch_size = 4  # Reduzindo o tamanho do lote para o mínimo
image_size = 64
nz = 100  # Tamanho do vetor de ruído de entrada do gerador
ngf = 64  # Número de filtros nas camadas convolucionais do gerador
ndf = 64  # Número de filtros nas camadas convolucionais do discriminador
num_epochs = 5
lr = 0.0002
beta1 = 0.5

# Transformações para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Carregar conjunto de dados de treinamento
dataset = torchvision.datasets.ImageFolder(
    root="X:/pablo dev/GITHUB/projeto_gan/codigo_python/imagens/Video Projects",
    transform=transform,
)

# Criar DataLoader para iterar sobre os dados em lotes durante o treinamento
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0
)  # Usando apenas 1 worker para evitar erros

# Dispositivo de treinamento (CPU ou GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Inicializar o gerador
gerador = Gerador(nz, ngf, 3).to(device)

# Carregar o modelo ResNet-50 pré-treinado
resnet = models.resnet50(pretrained=True)

# Congelar parâmetros da ResNet-50
for param in resnet.parameters():
    param.requires_grad = False

# Substituir a camada de classificação final
resnet.fc = nn.Linear(
    resnet.fc.in_features, 1
)  # Saída é um único valor para o discriminador

# Definir o discriminador com ResNet-50
class Discriminador(nn.Module):
    def __init__(self, resnet):
        super(Discriminador, self).__init__()
        self.resnet = resnet

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))

# Inicializar o discriminador
discriminador = Discriminador(resnet).to(device)

# Definir funções de perda e otimizadores
criterion = nn.BCELoss()  # Função de perda de entropia cruzada binária
optimizer_G = torch.optim.Adam(gerador.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = torch.optim.Adam(discriminador.parameters(), lr=lr, betas=(beta1, 0.999))

# Laço de treinamento
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Obter dados
        real_images, _ = data
        real_images = real_images.to(device)

        # Treinar o discriminador com imagens reais
        output_real = discriminador(real_images).view(-1)
        label_real = torch.full((batch_size,), 1.0, device=device)

        # Treinar o discriminador com imagens falsas geradas pelo gerador
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = gerador(noise)
        output_fake = discriminador(fake_images.detach()).view(-1)
        label_fake = torch.full((batch_size,), 0.0, device=device)

        # Calcular a perda do discriminador para imagens reais e falsas
        errD_real = criterion(output_real, label_real)
        errD_fake = criterion(output_fake, label_fake)
        
        # Soma as perdas do discriminador
        errD = errD_real + errD_fake

        # Zerar os gradientes antes de retropropagar
        optimizer_D.zero_grad()

        # Retropropagar o erro
        errD.backward()

        # Atualizar os pesos do discriminador
        optimizer_D.step()

        # Treinar o gerador
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = gerador(noise)
        output = discriminador(fake_images).view(-1)
        label = torch.full((batch_size,), 1.0, device=device)  # Queremos que o gerador engane o discriminador, então usamos rótulos reais
        errG = criterion(output, label)

        # Zerar os gradientes antes de retropropagar no gerador
        optimizer_G.zero_grad()

        # Retropropagar o erro no gerador
        errG.backward()

        # Atualizar os pesos do gerador
        optimizer_G.step()

        # Exibir progresso a cada batch
        if i % 100 == 0:
            print(
                "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f"
                % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item())
            )

# Salvar os modelos após o treinamento
torch.save(gerador.state_dict(), "gerador.pth")
torch.save(discriminador.state_dict(), "discriminador.pth")

print("Treinamento concluído!")
