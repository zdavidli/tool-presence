import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class VAE(nn.Module):
    def __init__(self, image_channels, image_size, h_dim1, h_dim2,
                 zdim, conv_channels=[16, 16]):
        super(VAE, self).__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.h_dim1 = h_dim1
        self.h_dim2 = h_dim2
        self.zdim = zdim
        self.conv_channels = conv_channels

        # Encoder
        self.conv1 = nn.Conv2d(
            image_channels, conv_channels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            conv_channels[0], conv_channels[1], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2)

        # Latent vectors
        self.fc1 = nn.Linear(image_size//2 * image_size //
                             2 * conv_channels[1], h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, zdim)
        self.fc32 = nn.Linear(h_dim2, zdim)

        # Decoder
        self.fc3 = nn.Linear(zdim, h_dim2)
        self.fc4 = nn.Linear(h_dim2, h_dim1)
        self.fc5 = nn.Linear(h_dim1, image_size//2 *
                             image_size//2 * conv_channels[1])

        self.conv3 = nn.ConvTranspose2d(
            conv_channels[1], conv_channels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv4 = nn.ConvTranspose2d(
            conv_channels[0], image_channels, kernel_size=3, stride=1,
            padding=1, bias=False)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.dropout(self.pool1(x))
        x = x.view(-1, self.image_size//2 * self.image_size //
                   2 * self.conv_channels[-1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc31(x), self.fc32(x)

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = z.view(-1, self.conv_channels[-1],
                   self.image_size//2, self.image_size//2)
        z = F.interpolate(z, scale_factor=2)
        z = F.relu(self.conv3(z))
        z = torch.sigmoid(self.conv4(z))
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return self.decode(z), mu, log_var


def vae_loss(recon_x, x, mu, log_var, input_size=1, zdim=1, beta=1):
    RL = F.binary_cross_entropy(recon_x, x, reduction='sum')/input_size
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())/zdim
    loss = RL + KLD * beta
    return loss, RL, KLD
