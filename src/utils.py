from scipy.special import logsumexp
from scipy.stats import norm
import argparse
import numpy as np
import pandas as pd
import os
from collections import OrderedDict


import src.constants as c
import src.model as m
import torch
from torchvision import datasets, transforms

from tqdm import tqdm, trange

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
                        default=os.path.join(c.data_home, 'youtube_data'),
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
    parser.add_argument('--z-dim', type=str, default='64', help=' ')
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
    parser.add_argument('-a', '--augmentation', help="data augmentation",
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
    transformations = []
    if args.augmentation:
        transformations.extend([
            transforms.RandomHorizontalFlip(),
        ])

    transformations.extend([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()])

    data_transforms = transforms.Compose(transformations)

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


def compute_samples(data, model, num_samples):
    """
    Sample from importance distribution z_samples ~ q(z|X) and
    compute p(z_samples), q(z_samples) for importance sampling
    Adapted from http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
    """

    z_mean, z_log_sigma = model.encode(data.to(c.device))
    z_mean, z_log_sigma = torch_to_numpy(
        z_mean), torch_to_numpy(z_log_sigma)
    z_samples = []
    qz = []

    for m, s in zip(z_mean, z_log_sigma):
        z_vals = [np.random.normal(m[i], np.exp(s[i]), num_samples)
                  for i in range(len(m))]
        qz_vals = [norm.pdf(z_vals[i], loc=m[i], scale=np.exp(s[i]))
                   for i in range(len(m))]
        z_samples.append(z_vals)
        qz.append(qz_vals)

    z_samples = np.array(z_samples)
    pz = norm.pdf(z_samples)
    qz = np.array(qz)

    z_samples = np.swapaxes(z_samples, 1, 2)
    pz = np.swapaxes(pz, 1, 2)
    qz = np.swapaxes(qz, 1, 2)

    return z_samples, pz, qz


def get_encodings(datasets, model, args, save=True):
    """
    Generate Latent space encodings from dataset
    """
    train_encoding = []
    test_encoding = []
    with torch.no_grad():
        for index in trange(len(datasets['train'])):
            data = datasets['train'][index][0].view(
                -1, args.image_channels, args.image_size, args.image_size).to(c.device)
            latent_vector = torch_to_numpy(
                model.sampling(*model.encode(data)))
            train_encoding.extend(
                [ar[0] for ar in np.split(latent_vector, data.shape[0])])

        for index in trange(len(datasets['val'])):
            data = datasets['val'][index][0].view(
                -1, args.image_channels, args.image_size, args.image_size).to(c.device)
            latent_vector = torch_to_numpy(
                model.sampling(*model.encode(data)))
            test_encoding.extend(
                [ar[0] for ar in np.split(latent_vector, data.shape[0])])
    train = pd.DataFrame(train_encoding)
    test = pd.DataFrame(test_encoding)
    if args.verbose:
        print(train.head(5))

    if save:
        print("Saving to", os.path.join(
            args.root, args.output_dir, args.output_header))
        train.to_csv(os.path.join(args.root, args.output_dir,
                                  args.output_header + "_train.csv"))
        test.to_csv(os.path.join(args.root, args.output_dir,
                                 args.output_header + "_test.csv"))

    return train, test


def pystan_vb_extract(results):
    """
    VB extract function
    """
    param_specs = results['sampler_param_names']
    samples = results['sampler_params']
    n = len(samples[0])

    # first pass, calculate the shape
    param_shapes = OrderedDict()
    for param_spec in param_specs:
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            # no +1 for shape calculation because pystan already returns 1-based indexes for vb!
            idxs = [int(i) for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        param_shapes[name] = np.maximum(idxs, param_shapes.get(name, idxs))

    # create arrays
    params = OrderedDict([(name, np.nan * np.empty((n, ) + tuple(shape)))
                          for name, shape in param_shapes.items()])

    # second pass, set arrays
    for param_spec, param_samples in zip(param_specs, samples):
        splt = param_spec.split('[')
        name = splt[0]
        if len(splt) > 1:
            # -1 because pystan returns 1-based indexes for vb!
            idxs = [int(i) - 1 for i in splt[1][:-1].split(',')]
        else:
            idxs = ()
        params[name][(..., ) + tuple(idxs)] = param_samples

    return params


def estimate_logpx(dataloader, model, args, num_samples):
    """
    Adapted from http://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/

    Calculate importance sample
    \log p(x) = E_p[p(x|z)]
    = \log(\int p(x|z) p(z) dz)
    = \log(\int p(x|z) p(z) / q(z|x) q(z|x) dz)
    = E_q[p(x|z) p(z) / q(z|x)]
    ~= \log(1/n * \sum_i p(x|z_i) p(z_i)/q(z_i))
    = \log p(x) = \log(1/n * \sum_i e^{\log p(x|z_i) + \log p(z_i) - \log q(z_i)})
    = \log p(x) = -\logn + \logsumexp_i(\log p(x|z_i) + \log p(z_i) - \log q(z_i))
    See: scipy.special.logsumexp
    """

    result = []
    for batch_idx, (data, _) in enumerate(dataloader):
        z_samples, pz, qz = compute_samples(data, model, num_samples)
        assert z_samples.shape == pz.shape
        assert pz.shape == qz.shape
        for i in range(len(data)):
            datum = torch_to_numpy(data[i]).reshape(
                args.image_size * args.image_size * args.image_channels)
            x_predict = model.decode(torch.from_numpy(
                z_samples[i]).float().to(c.device))
            x_predict = torch_to_numpy(
                x_predict).reshape(-1, args.image_size * args.image_size * args.image_channels)
            x_predict = np.clip(x_predict, np.finfo(
                float).eps, 1. - np.finfo(float).eps)
            p_vals = pz[i]
            q_vals = qz[i]

            # \log p(x|z) = Binary cross entropy
            logp_xz = np.sum(datum * np.log(x_predict) +
                             (1. - datum) * np.log(1.0 - x_predict), axis=-1)
            logpz = np.sum(np.log(p_vals), axis=-1)
            logqz = np.sum(np.log(q_vals), axis=-1)
            argsum = logp_xz + logpz - logqz
            logpx = -np.log(num_samples) + logsumexp(argsum)
            result.append(logpx)

    return np.array(result)
