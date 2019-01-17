import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class EndoscopyDataset(Dataset):
    """Endoscopy Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) if os.path.isfile(self.root_dir+name)])
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "frame_{0:04d}.png".format(idx))
        image = io.imread(img_name)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample