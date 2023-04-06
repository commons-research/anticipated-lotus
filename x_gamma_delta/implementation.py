import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#import personal packages
import utils.sigma_utils as sigma_utils
from utils.utils import *
from utils.MCMC import run_mcmc_with_gibbs

#initialize size of simulated data
T = ['m', 's']
n_t = [100, 10]
assert len(T)==len(n_t)

#simulate priors
print("generating prior...")
mu, alpha, beta, sigma = simulate_from_prior(T, n_t)

#calculate sum of all possible combination of mus
sum_mu = compute_sum_of_mus(mu)

print("Calculating epsilon_c...")
epsilon_matrix = sigma_utils.compute_epsilon(alpha, sigma)
epsilon_c = simulate_epsilon_c(epsilon_matrix)

#get probability of x and simulate x
prob_x = compute_prob_X(sum_mu, epsilon_c, n_t)
x = simulate_X(prob_x)

#get probability of Lotus and simulate Lotus
prob_Lotus, n_papers, gamma, delta = compute_prob_L(x)
lotus_binary, lotus_n_papers = simulate_lotus(prob_Lotus, n_papers)

# Run the MCMC chain
n_iter = 100000
gamma_init = 1
delta_init = 1
x_init = np.zeros_like(lotus_binary, dtype=np.float64)
print("Running MCMC")
samples, x_samples, accept_gamma, accept_delta = run_mcmc_with_gibbs(lotus_n_papers, x_init, n_iter,
                                                                     gamma_init, delta_init,
                                                                     sum_mu.reshape(n_t),
                                                                     epsilon_c.reshape(n_t))

burn_in = int(0.5 * n_iter)  # Remove the first 50% of the samples
post_burn_in_samples = samples[burn_in:]

# Extract the posterior mean estimates for gamma and delta
gamma_posterior_mean = np.mean(post_burn_in_samples[:, 0])
delta_posterior_mean = np.mean(post_burn_in_samples[:, 1])

print("True gamma: ", gamma)
print("Estimated gamma: ", gamma_posterior_mean)
print("True delta: ", delta)
print("Estimated delta: ",delta_posterior_mean)
print("rate accept gamma : ", accept_gamma)
print("rate accept delta : ", accept_delta)

print(np.corrcoef(x.flatten(), x_samples[-1].flatten()))

#sns.set(style="darkgrid")
#fig, axs = plt.subplots(ncols=2)
#sns.scatterplot(x=range(len(post_burn_in_samples)),
#                y = post_burn_in_samples[:,0],
#                ax=axs[0]).set_title(f'True gamma : {gamma}')
#sns.lineplot(x=range(len(post_burn_in_samples)),
#             y=[float(gamma) for i in range(len(post_burn_in_samples[:,0]))],
#             ax=axs[0],
#             color='r')
#
#sns.scatterplot(x=range(len(post_burn_in_samples)),
#                y=post_burn_in_samples[:,1],
#                ax=axs[1]).set_title(f'True delta : {delta}')
#sns.lineplot(x=range(len(post_burn_in_samples)),
#             y=[float(delta) for i in range(len(post_burn_in_samples[:,1]))],
#             ax=axs[1],
#             color='r')
#plt.show()


## DONE simulate \epsilson_c from multivariate normal distribution with mean = 0 and cov epsilon 
## DONE Once we have epsilon_c and our sum of mus, we have the probability of X. We can draw X with a Bernoulli varaibles and we get our data X. 
## DONE we can simulate the number of papers with a Poisson distribution --> 1+ Poisson
## DONE We can then simulate the Lotus data by drawing from X. 
## DONE from x simulate Lotus from x and P and Q --> R 
## DONE from R draw Bernoulli with P = R
## --> simulated data

# try to infer \gamma and \delta from MCMC --> should find our \delta and \gamma