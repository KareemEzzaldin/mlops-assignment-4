import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')
data = df.values
data = np.pad(data, ((0, 0), (0, 2)), mode='constant')
data = data.reshape(-1, 1, 4, 4)

data_min = data.min()
data_max = data.max()
data = (data - data_min) / (data_max - data_min) * 2 - 1

tensor_data = torch.FloatTensor(data)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 4, 4)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

netG = Generator()
netD = Discriminator()

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 100

for epoch in range(epochs):

    acc_real_total = 0
    acc_fake_total = 0
    total_samples = 0

    for i, data_batch in enumerate(dataloader):
        real_data = data_batch[0]
        b_size = real_data.size(0)
        total_samples += b_size

        netD.zero_grad()

        label_real = torch.ones(b_size, 1)
        output_real = netD(real_data)
        errD_real = criterion(output_real, label_real)
        errD_real.backward()

        preds_real = (output_real > 0.5).float()
        acc_real_total += (preds_real == label_real).sum().item()

        noise = torch.randn(b_size, 10)
        fake_data = netG(noise)
        label_fake = torch.zeros(b_size, 1)
        output_fake = netD(fake_data.detach())
        errD_fake = criterion(output_fake, label_fake)
        errD_fake.backward()
        optimizerD.step()

        preds_fake = (output_fake > 0.5).float()
        acc_fake_total += (preds_fake == label_fake).sum().item()

        netG.zero_grad()
        label_g = torch.ones(b_size, 1)
        output_g = netD(fake_data)
        errG = criterion(output_g, label_g)
        errG.backward()
        optimizerG.step()

    d_acc_overall = (acc_real_total + acc_fake_total) / (2 * total_samples)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Discriminator Accuracy: {d_acc_overall * 100:.2f}%")

noise = torch.randn(1, 10)
fake_image = netG(noise).detach().numpy().reshape(4, 4)

plt.imshow(fake_image, cmap='gray')
plt.axis('off')
plt.savefig('generated_sample.png')
