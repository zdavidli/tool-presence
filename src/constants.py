import os

import torch

# Training arguments
image_channels = 3
image_size = 64
batch_size = 32
epochs = 50
save_every = 50
test_every = 10

seed = 1
torch.manual_seed(seed)

# Cuda
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Paths
root = '/users/dli44/tool-presence/'
data_home = os.path.join(root, 'data/')
