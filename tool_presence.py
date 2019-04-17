import os

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from src import constants as c
from src import utils
from src import visualization as v
from src import model as m
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm, trange


def main(args):
    datasets, dataloaders = utils.setup_data(args)
    args.betas = [float(x) for x in args.betas.split(',')]
    for beta in tqdm(args.betas):
        output_name = '{}_beta_{}_epoch_{{}}.{{}}'.format(
            args.output_name, beta)
        losses = {'kl': [], 'rl': []}
        model = m.VAE(image_channels=args.image_channels,
                      image_size=args.image_size,
                      h_dim1=1024,
                      h_dim2=128,
                      zdim=args.z_dim).to(c.device)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        tbar = trange(args.epochs)
        for epoch in tbar:
            """
            Training
            """
            model.train()
            train_loss, kl, rl = 0, 0, 0
            t2 = tqdm(dataloaders['train'])
            for batch_idx, (data, _) in enumerate(t2):
                data = data.to(c.device)
                optimizer.zero_grad()
                recon_batch, z, mu, logvar = model(data)

                loss_params = {'recon': recon_batch,
                               'x': data,
                               'z': z,
                               'mu': mu,
                               'logvar': logvar,
                               'batch_size': args.batch_size,
                               'input_size': args.image_size,
                               'zdim': args.z_dim,
                               'beta': 5}

                loss, r, k = args.loss_function(**loss_params)
                loss.backward()

                train_loss += loss.item()
                kl += k.item()
                rl += r.item()

                optimizer.step()

                t2.set_postfix(
                    {"Reconstruction Loss": r.item(),
                     "KL Divergence": k.item()})

            losses['kl'].append(kl)
            losses['rl'].append(rl)

            tbar.set_postfix({"KL Divergence":
                              kl/len(dataloaders['train'].dataset),
                              "Reconstruction Loss":
                              rl/len(dataloaders['train'].dataset)})

            """
            Testing
            """
            if (epoch + 1) % args.sample_model_interval == 0:
                model.eval()
                with torch.no_grad():
                    it = iter(dataloaders['val'])
                    data, _ = next(it)
                    data = data.to(c.device)
                    recon_batch, z, mu, logvar = model(data)
                    loss_params = {'recon': recon_batch,
                                   'x': data,
                                   'z': z,
                                   'mu': mu,
                                   'logvar': logvar,
                                   'batch_size': args.batch_size,
                                   'input_size': args.image_size,
                                   'zdim': args.z_dim,
                                   'beta': 5}
                    loss, r, k = args.loss_function(**loss_params)
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(args.batch_size,
                                                             c.image_channels,
                                                             args.image_size,
                                                             args.image_size)
                                            [:n]
                                            ])

                    save_image(comparison.cpu(),
                               os.path.join(args.output_dir,
                                            output_name.format(
                                                epoch+1,
                                                'png')
                                            ),
                               nrow=n)

            if (epoch + 1) % args.save_model_interval == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir,
                                        output_name.format(
                                            epoch+1,
                                            'torch')))


# set up argparse
parser = utils.setup_argparse()
args = parser.parse_args()
# args = parser.parse_args(['--data-dir=./data/surgical_data/',
#                           '--output-dir=./data/beta_vae/2fc',
#                           '--epochs=1',
#                           '--image-size=16',
#                           '--betas=2,5',
#                           '--sample-model-interval=1'])
args.data_dir = os.path.abspath(args.data_dir)
os.makedirs(args.output_dir, exist_ok=True)
args.loss_function = utils.select_loss_function(args.loss_function)

if args.verbose:
    print("Using loss function:", args.loss_function.__name__)
    print("Saving data to:",
          os.path.join("./", args.output_dir, args.output_name))

# pass args to main
main(args)

