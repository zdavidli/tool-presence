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

