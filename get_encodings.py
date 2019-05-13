import os

import argparse
import pandas as pd
import numpy as np
import torch
from src import constants as c
from src import utils
from src import visualization as v
from src import model as m
from tqdm import tqdm, trange


def main(args):
    datasets, _ = utils.setup_data(args)

    model = m.VAE(image_channels=args.image_channels,
                  image_size=args.image_size,
                  h_dim1=1024,
                  h_dim2=128,
                  zdim=args.z_dim).to(c.device)

    model.load_state_dict(torch.load(os.path.join(args.root, args.path)))

    output_string = args.output_header + "_{}.csv"
    train, test = utils.get_encodings(
        datasets, model, args, output_string, save=False)


if __name__ == '__main__':
    # set up argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str,
                        default=os.path.abspath('.'),
                        help='Root directory of tool-presence')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(c.data_home, 'youtube_data'),
                        help=' ')
    parser.add_argument('--output-header', type=str,
                        default=os.path.abspath('inference/encodings'),
                        help=' ')
    parser.add_argument('--path', type=str, default='',
                        help='Location of weights file')
    parser.add_argument('--image-channels', type=int,
                        default=c.image_channels, help=' ')
    parser.add_argument('--image-size', type=int,
                        default=c.image_size, help=' ')
    parser.add_argument('--batch-size', type=int, default=32, help=' ')
    parser.add_argument('--z-dim', type=int, help='latent dimensions')
    parser.add_argument('--weights', type=str, default='',
                        help='path of weights file')
    parser.add_argument('-v', '--verbose', help="increase output verbosity",
                        action="store_true")
    parser.add_argument('-a', '--augmentation',
                        help='dataset augmentation', action='store_true')
    args = parser.parse_args()

    args.data_dir = os.path.abspath(os.path.join(args.root, args.data_dir))
    os.makedirs(os.path.join(args.root, args.output_dir), exist_ok=True)

    if args.verbose:
        print("Saving data to:",
              os.path.join(args.root, args.output_dir))
        print("Reading model from:",
              os.path.join(args.root, args.path))

    # pass args to main
    main(args)
