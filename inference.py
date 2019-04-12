import os
import argparse
import pystan
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import seaborn as sns

def confusion(result, test_labels):
    print(type(test_labels))
    predictions = np.zeros((len(test_labels), ))
    for i, row in enumerate(test_labels.itertuples()):
        pz = result['theta'][-1][:,0] #mixing probabilities

        py_z0 = norm.pdf(row[:10],
                        loc=result['mu'][-1][0],
                        scale=result['sigma'][-1][0])
        py_z1 = norm.pdf(row[:10],
                        loc=result['mu'][-1][1],
                        scale=result['sigma'][-1][1])
        posterior0 = np.dot(pz,py_z0)
        posterior1 = np.dot((1-pz),py_z1)

        predictions[i] = int(posterior0 > posterior1)

    confusion = confusion_matrix(test_labels['Tool'].values, predictions)
    return confusion

def main(args):
    #  constants
    model = """
    data {
    int N; // number of observations
    int D; // dimension of observed vars
    int K; // number of clusters
    vector[D] x[N]; // training data
    }

    parameters {
    ordered[K] mu; // locations of hidden states
    vector<lower=0>[K] sigma; // variances of hidden states
    simplex[K] theta[D]; // mixture components
    }

    model {
    matrix[K,D] obs = rep_matrix(0.0, K, D);
    // priors
    for(k in 1:K){
      mu[k] ~ normal(0,10);
      sigma[k] ~ inv_gamma(1,1); //prior of normal distribution
    }
    for (d in 1:D){
      theta[d] ~ dirichlet(rep_vector(2.0, K)); //prior of categorical distribution
    }
    // likelihood
    for(i in 1:N) {
      vector[D] increments;
      for(d in 1:D){
        increments[d]=log_mix(theta[d][1],
        normal_lpdf(x[i][d] | mu[1], sigma[1]), normal_lpdf(x[i][d] | mu[2], sigma[2]));
      }
      target += log_sum_exp(increments);
    }
    }

    generated quantities {
    }


    """
    train_data_file = os.path.join(args.root, args.train)
    test_data_file = os.path.join(args.root, args.test)
    test_labels_file = os.path.join(args.root, 'data/youtube_data/val/labels.csv')
    compiled_model = os.path.join(args.root, args.model_path)
    sampled_fit = os.path.join(args.root, args.fit_path)

    # Read data into pandas dataframe
    train = pd.read_csv(train_data_file, index_col=0)
    test = pd.read_csv(test_data_file, index_col=0)
    test_labels = pd.read_csv(test_labels_file, index_col=0)
    test_labels = pd.concat([test, test_labels], axis=1).dropna()

    # Want to learn tool/no tool (2 latent groups)
    data = {"N": len(train.index),
        "x":train,
        "K":2,
        "D":len(train.columns)}

    # stan parameters
    iters = 1000

    if args.recompile:
        sm = pystan.StanModel(model_code=model)
        with open(compiled_model, 'wb') as f:
            pickle.dump(sm, f)
    else:
        with open(compiled_model, 'rb') as f:
            sm = pickle.load(f)

    if args.refit:
        fit = sm.sampling(data=data, warmup=500, iter=iters, chains=2, thin=1)
        with open(sampled_fit, 'wb') as f:
            pickle.dump(fit, f)
    else:
        with open(sampled_fit, 'rb') as f:
            fit = pickle.load(f)

    result = fit.extract()

    c = confusion(result, test_labels)
    sns.heatmap(c.astype('float') / c.sum(axis=1)[:, np.newaxis], cmap=sns.color_palette("Blues"),
                xticklabels=['No Tool', 'Tool'], yticklabels=['No Tool', 'Tool'], annot=c, fmt='g',cbar=False)
    plt.ylabel("Predictions")
    plt.xlabel("Actual")
    plt.title(r"$\beta$-VAE Confusion Matrix\\"
              r"$\beta=1, z=10$");
    plt.savefig('beta_vae_beta1_confusion.png')

if __name__=='__main__':
    # set up argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str,
                        default=os.path.abspath('.'),
                        help='Root directory of tool-presence')
    parser.add_argument('--train', type=str, default='')
    parser.add_argument('--test', type=str, default='')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--fit-path', type=str, default='')
    parser.add_argument('-v', '--verbose', help="increase output verbosity",
                        action="store_true")
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--recompile', action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.root, args.model_path), exist_ok=True)
    os.makedirs(os.path.join(args.root, args.fit_path), exist_ok=True)

    if args.verbose:
        print("Reading train data from:",
              os.path.join(args.root, args.train))
        print("Reading test data from:",
              os.path.join(args.root, args.test))
        print("Saving inference model to:",
              os.path.join(args.root, args.model_path))
        print("Saving fit to:",
              os.path.join(args.root, args.fit_path))

    # pass args to main
    main(args)
