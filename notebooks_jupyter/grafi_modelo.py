import matplotlib.pyplot as plt

# Substitua esses dados pelos seus dados reais
losses_D = [1.4108, 1.3983, 1.4060, 1.3541, 1.3467]  # Exemplo de perda do discriminador
losses_G = [0.6578, 0.7395, 0.6557, 0.6591, 0.6892]  # Exemplo de perda do gerador

# Número de épocas
epochs = range(1, len(losses_D) + 1)

# Plotando as perdas
plt.plot(epochs, losses_D, label='Loss Discriminador')
plt.plot(epochs, losses_G, label='Loss Gerador')

# Adicionando rótulos e título
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Treinamento da GAN')

# Adicionando legenda
plt.legend()

# Exibindo o gráfico
plt.show()
