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
    datasets, dataloaders = utils.setup_data(args)

    model = m.VAE(image_channels=args.image_channels,
                  image_size=args.image_size,
                  h_dim1=1024,
                  h_dim2=128,
                  zdim=args.z_dim).to(c.device)

    model.load_state_dict(torch.load(os.path.join(args.root, args.path)));

    train_encoding = []
    test_encoding = []
    with torch.no_grad():
        for index in trange(len(datasets['train'])):
            data = datasets['train'][index][0].view(-1, args.image_channels, args.image_size, args.image_size).to(c.device)
            latent_vector = utils.torch_to_numpy(model.sampling(*model.encode(data)))
            train_encoding.extend([ar[0] for ar in np.split(latent_vector, data.shape[0])])

        for index in trange(len(datasets['val'])):
            data = datasets['val'][index][0].view(-1, args.image_channels, args.image_size, args.image_size).to(c.device)
            latent_vector = utils.torch_to_numpy(model.sampling(*model.encode(data)))
            test_encoding.extend([ar[0] for ar in np.split(latent_vector, data.shape[0])])
    train = pd.DataFrame(train_encoding)
    test = pd.DataFrame(test_encoding)
    if args.verbose:
        print(train.head(5))

    print("Saving to", os.path.join(args.root, args.output_dir, "".join(os.path.splitext(os.path.basename(args.path))[:-1]) + '_{train, test}.csv'))
    train.to_csv(os.path.join(args.root, args.output_dir, "".join(os.path.splitext(os.path.basename(args.path))[:-1]) + '_train.csv'))
    test.to_csv(os.path.join(args.root, args.output_dir, "".join(os.path.splitext(os.path.basename(args.path))[:-1]) + '_test.csv'))

if __name__=='__main__':
    # set up argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str,
                        default=os.path.abspath('.'),
                        help='Root directory of tool-presence')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(c.data_home, 'youtube_data'),
                        help=' ')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.abspath('.'),
                        help=' ')
    parser.add_argument('--path', type=str, default='', help='Location of weights file')
    parser.add_argument('--image-channels', type=int,
                        default=c.image_channels, help=' ')
    parser.add_argument('--image-size', type=int,
                        default=c.image_size, help=' ')
    parser.add_argument('--batch-size', type=int, default=32, help=' ')
    parser.add_argument('--z-dim', type=int, help='latent dimensions')
    parser.add_argument('--weights', type=str, default='', help='path of weights file')
    parser.add_argument('-v', '--verbose', help="increase output verbosity",
                        action="store_true")
    parser.add_argument('-a', '--augmentation', help='dataset augmentation', action='store_true')
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
