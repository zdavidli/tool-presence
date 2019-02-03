
# coding: utf-8

# # Tutorial
#
# Based off: https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
#
# Data used: Sinus Endoscopy Video from https://www.youtube.com/watch?v=6niL7Poc_qQ
#
# Simplifying Decisions:
#
# * Downscale image to 64x64 with center crop (not perfect)

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm



class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=32*32, z_dim=32):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.conv5 = nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2)
        self.conv6 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.conv7 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2)
        self.conv8 = nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=1)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu, logvar = self.fc1(x), self.fc2(x)
        return mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 1024, 1, 1)
        z = F.relu(self.conv5(z))
        z = F.relu(self.conv6(z))
        z = F.relu(self.conv7(z))
        z = torch.sigmoid(F.relu(self.conv8(z)))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sampling(mu, logvar)
        z = self.decode(z)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Variables
batch_size=32
epochs = 50
image_channels=3

# Dataset
dataset = torchvision.datasets.ImageFolder('data/',
                                           transform=transforms.Compose([
                                               transforms.Resize(32),
                                               transforms.CenterCrop(32),
                                               transforms.ToTensor()
                                           ]))

data_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=False)

fixed_x, _ = dataset[0]

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


for epoch in tqdm(range(epochs)):
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda()
        recon_images, mu, logvar = model(images)
        loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1,
                                                                    epochs,
                                                                    loss.item()/batch_size,
                                                                    bce.item()/batch_size,
                                                                    kld.item()/batch_size)
        tqdm.write(to_print)

torch.save(model.state_dict(), 'vae_tools.torch')

fixed_x = dataset[np.random.randint(1, 100)][0].unsqueeze(0)

recon_x = model(fixed_x)[0].squeeze()

compare = torch.cat([fixed_x.squeeze(), recon_x], dim=2)
