import torch

image_channels = 3
image_size = 64

batch_size = 32
epochs = 100
seed = 1
log_interval = 5
validation_split = .2

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

data_home = '/users/dli44/tool-presence/data/'