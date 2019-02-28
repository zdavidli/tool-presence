
# coding: utf-8

# ## Imports

# In[37]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm, tnrange

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


batch_size = 32
epochs = 50
seed = 1
log_interval = 50

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


# In[40]:


# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform = transforms.Compose([
                                   transforms.Lambda(lambda x: x.convert('RGB')),
                                   transforms.Resize(128),
                                   transforms.ToTensor(),
                               ]),
                               download=False)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.Compose([
                                  transforms.Lambda(lambda x: x.convert('RGB')),
                                  transforms.Resize(128),
                                  transforms.ToTensor(),
                              ]),
                              download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# In[41]:


plt.imshow(train_dataset[1][0].numpy().transpose(1,2,0))


# In[63]:


class VAE(nn.Module):
    def __init__(self, zdim, hdim1, hdim2, hdim3=128):
        super(VAE, self).__init__()
        self.zdim = zdim
        self.hdim1 = hdim1
        self.hdim2 = hdim2
        self.hdim3 = hdim3

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.pool1 = nn.MaxPool2d(2)

        # Latent vectors
        self.fc1 = nn.Linear(self.hdim1, self.hdim2)
        self.fc2 = nn.Linear(self.hdim2, self.hdim3)
        self.fc31 = nn.Linear(self.hdim3, zdim)
        self.fc32 = nn.Linear(self.hdim3, zdim)

        # Decoder
        self.fc3 = nn.Linear(zdim, self.hdim3)
        self.fc4 = nn.Linear(self.hdim3, self.hdim2)
        self.fc5 = nn.Linear(self.hdim2, self.hdim1)

        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=2, bias=False)
        self.conv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(self.pool1(x))
        x = x.view(-1, self.hdim1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc31(x), self.fc32(x)

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = z.view(-1, 32, 32, 32)
        z = F.interpolate(z, scale_factor=2)
        z = F.relu(self.conv3(z))
        z = torch.sigmoid(self.conv4(z))
        return z

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sampling(mu, log_var)
        return self.decode(z), mu, log_var


model = VAE(zdim=2, hdim1=32*32*32, hdim2=1024, hdim3=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)



def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

val_losses = []
train_losses = []


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            tqdm.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    tqdm.write('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    train_losses.append(train_loss / len(train_loader.dataset))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 3, 128, 128)[:n]])
                save_image(comparison.cpu(),
                           './reconstruction_upsample_128_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    tqdm.write('====> Test set loss: {:.4f}'.format(test_loss))
    val_losses.append(test_loss)


# In[71]:


for epoch in tqdm(range(epochs)):
    train(epoch+1)
    test(epoch+1)


torch.save(model.state_dict(), "./cnn_vae_upsample_128_50epochs_zdim2.torch")


plt.plot(val_losses)
plt.title('Validation Loss\nCNN VAE Upsample 128x128\nz=2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('./upsample_128_validation_loss.png')


# In[ ]:


plt.plot(train_losses)
plt.title('Training Loss\nCNN VAE Upsample 128x128\nz=2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('./upsample_128_training_loss.png')


# In[ ]:


# model.load_state_dict(torch.load("../../weights/cnn_vae_50epochs_dropout.torch"))


# In[ ]:


with torch.no_grad():
    z = torch.randn(64, 2)
    sample = model.decode(z.cuda())
#     plt.imshow(sample.cpu().numpy())
    save_image(sample.view(64, 3, 128, 128).cpu(), './sample_zdim_{}'.format(2) + '.png')


# In[ ]:

n = 15
digit_size=128

u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)

x_decoded = model.decode(torch.from_numpy(z_grid.reshape(n*n, 2)).float().cuda())
x_decoded = x_decoded.reshape(n, n, digit_size, digit_size)

plt.figure(figsize=(10, 10))
plt.imshow(np.block(list(map(list, x_decoded.detach().cpu().numpy()))), cmap='gray')
plt.show()
plt.savefig('./upsample_128_latent_dimension_sample.png')

