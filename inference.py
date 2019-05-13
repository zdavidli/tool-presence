import os
import argparse
import pystan
import pandas as pd
import numpy as np
import pickle

from src import constants as c
from src import utils
from src import visualization as v
from src import model as m


def main(args):
    train = pd.read_csv(os.path.join(
        args.root, args.data_dir, args.data_name + '_train.csv'))
    test = pd.read_csv(os.path.join(
        args.root, args.data_dir, args.data_name + '_tests.csv'))
    print("Loaded data")

    if args.recompile:
        compiled_model = os.path.join(args.root, args.model_save_path)
    else:
        compiled_model = os.path.join(args.root, args.model_path)
    sampled_fit = os.path.join(args.root, args.fit_save_path)

    # Read data into pandas dataframe
    test_labels = pd.read_csv(os.path.join(args.root, args.test_labels_file),
                              index_col=0)
    test_labels = pd.concat([test, test_labels], axis=1).dropna()

    # Want to learn tool/no tool (2 latent groups)
    data = {"N": len(train.index),
            "N2": len(test_labels),
            "x": train,
            "x_test": test_labels.values[:, :10],
            "K": 2,
            "D": len(train.columns)}

    # stan parameters
    iters = 1000

    if args.recompile:
        sm = pystan.StanModel(file=os.path.join(args.root, args.stan_model))
        with open(compiled_model, 'wb') as f:
            pickle.dump(sm, f)
    else:
        with open(compiled_model, 'rb') as f:
            sm = pickle.load(f)

    if args.vb:
        fit = sm.vb(data=data, algorithm='meanfield')
    else:
        fit = sm.sampling(data=data, warmup=500, iter=iters, chains=2, thin=1)

    with open(sampled_fit, 'wb') as f:
        pickle.dump(fit, f)


if __name__ == '__main__':
    # set up argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str,
                        default=os.path.abspath('.'),
                        help='Root directory of tool-presence')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.abspath('.'), 'inference'), help='')
    parser.add_argument('--data-name', type=str, default='', help='')
    parser.add_argument('--test-labels', type=str, default='')
    parser.add_argument('--stan-model', type=str,
                        default='', help='Stan code')
    parser.add_argument('--model-path', type=str, default='',
                        help='Where to read pickled model')
    parser.add_argument('--model-save-path', type=str, default='',
                        help='Where to save pickled model')
    parser.add_argument('--fit-save-path', type=str, default='',
                        help='Where to save pickled fit')
    parser.add_argument(
        '--vb', help='Where to save pickled fit', action='store_true')
    parser.add_argument('-v', '--verbose', help="increase output verbosity",
                        action="store_true")
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--recompile', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        print("Reading train data from:",
              os.path.join(args.root, args.data_dir, args.data_name + '_train.csv'))
        print("Reading test data from:",
              os.path.join(args.root, args.data_dir, args.data_name + '_val.csv'))

    # pass args to main
    main(args)
