import argparse
import os

import src.constants as c
import src.model as m
import torch
from torchvision import datasets, transforms


def torch_to_numpy(tensor):
    """
    input: torch tensor
    output: numpy array
    """
    return tensor.detach().cpu().numpy()


def to_image(im):
    """
    input: result of torch_to_numpy()
    output: rgb image instead of brg
    """
    return im.squeeze().transpose(1, 2, 0)


def torch_to_image(tensor):
    """
    input: torch tensor
    output: numpy array as rgb image
    """
    return to_image(torch_to_numpy(tensor))


def setup_argparse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--retrain-if-model-exists',
                        type=int, default=0, help=' ')
    parser.add_argument('--root', type=str,
                        default=os.path.abspath('.'),
                        help='Root directory of tool-presence')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(c.data_home, 'surgical_data'),
                        help=' ')
    parser.add_argument('--output-dir', type=str,
                        default='', help='Directory to save outputs')
    parser.add_argument('--output-name', type=str,
                        default='', help='Filename to save outputs')
    parser.add_argument('--image-channels', type=int,
                        default=c.image_channels, help=' ')
    parser.add_argument('--image-size', type=int,
                        default=c.image_size, help=' ')
    parser.add_argument('--batch-size', type=int, default=32, help=' ')
    parser.add_argument('--z-dim', type=int, default=64, help=' ')
    parser.add_argument('--learning-rate', type=float,
                        default=1e-3, help=' ')
    parser.add_argument('--epochs', type=int, default=c.epochs, help=' ')
    parser.add_argument('--save-model-interval',
                        type=int, default=50, help=' ')
    parser.add_argument('--sample-model-interval',
                        type=int, default=10, help=' ')
    parser.add_argument('--betas', type=str,
                        default='5,20', help='beta values delimited by ,')
    parser.add_argument('--loss-function', choices=['mmd', 'vae'],
                        default='vae', help='Loss function choice')
    parser.add_argument('-v', '--verbose', help="increase output verbosity",
                        action="store_true")
    return parser


def select_loss_function(choice):
    if choice == 'mmd':
        return m.mmd_loss
    elif choice == 'vae':
        return m.vae_loss
    else:
        raise ValueError("Unrecognized loss function")


def setup_data(args):
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15),
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.root,
                                                           args.data_dir,
                                                           x),
                                              data_transforms)
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
                   for x in ['train', 'val']}

    return image_datasets, dataloaders
