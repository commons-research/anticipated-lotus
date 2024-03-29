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
n_iter = 10000
gamma_init = 1
delta_init = 1
x_init = np.zeros_like(lotus_binary, dtype=np.float64)
#x_init = x
mu_init = [np.zeros(i) for i in n_t]
print("Running MCMC")
print("N iterations : ", n_iter)
samples, x_samples, mu_samples, accept_gamma, accept_delta = run_mcmc_with_gibbs(lotus_n_papers, x_init, n_iter,
                                                                     gamma_init, delta_init,
                                                                     mu_arrays=mu_init,
                                                                     e=epsilon_c.reshape(n_t))

burn_in = int(0.5 * len(samples))  # Remove the first 50% of the samples
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

#check correlation of x (should be 1 if x is fixed, check in the MCMC file)
x_samples_burn = x_samples[burn_in:]
average_x = np.sum(x_samples_burn, axis=0)/burn_in
print(np.corrcoef(x.flatten(), average_x.flatten()))


mu_samples_burn = mu_samples[burn_in:]
average_estimations = {}
for key in mu:
    key_index = list(mu.keys()).index(key)
    key_estimations = [estimation[key_index] for estimation in mu_samples_burn]
    key_avg = np.mean(key_estimations, axis=0)
    average_estimations[key] = key_avg
    print(f"Correlation mu '{key}' : ", np.corrcoef(mu[key], average_estimations[key]))


sns.set(style="darkgrid")
fig, axs = plt.subplots(ncols=2)

#sns.scatterplot(x=[i for i in range(n_iter)],
#                y=[item[0][0] for item in mu_samples])
#sns.lineplot(x=[i for i in range(n_iter)],
#             y=[mu['m'][0] for i in range(n_iter)])

sns.scatterplot(x=range(len(post_burn_in_samples)),
                y = post_burn_in_samples[:,0],
                ax=axs[0]).set_title(f'True gamma : {gamma}')
sns.lineplot(x=range(len(post_burn_in_samples)),
             y=[float(gamma) for i in range(len(post_burn_in_samples[:,0]))],
             ax=axs[0],
             color='r')

sns.scatterplot(x=range(len(post_burn_in_samples)),
                y=post_burn_in_samples[:,1],
                ax=axs[1]).set_title(f'True delta : {delta}')
sns.lineplot(x=range(len(post_burn_in_samples)),
             y=[float(delta) for i in range(len(post_burn_in_samples[:,1]))],
             ax=axs[1],
             color='r')
plt.show()