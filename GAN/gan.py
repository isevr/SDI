import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from dataloader import get_data

data_path = 'heart.csv'
data = get_data(data_path)

input_dim = 100
output_dim = data[0].shape()[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


generator = Generator(input_dim, output_dim).to(device)
discriminator = Discriminator(output_dim).to(device)


print(f"Generator model initialized with input_dim={input_dim} and moved to {device}.")
print(f"Discriminator model initialized and moved to {device}.")

criterion = nn.BCELoss()
lr = 2e-5

optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    for i, (real_samples, _) in enumerate(data[1]):
        batch_size = real_samples.size(0)
        real_samples = real_samples.to(device)


        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
        fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_d.zero_grad()

        # Noise
        z = torch.randn(batch_size, input_dim).to(device)

        # Batch Generation
        gen_samples = generator(z)

        # Loss 
        real_loss = criterion(discriminator(real_samples), valid)
        fake_loss = criterion(discriminator(gen_samples.detach()), fake)
        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_d.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_g.zero_grad()

        # Loss
        g_loss = criterion(discriminator(gen_samples), valid)

        g_loss.backward()
        optimizer_g.step()


        if i % 100 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(data[1])}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")