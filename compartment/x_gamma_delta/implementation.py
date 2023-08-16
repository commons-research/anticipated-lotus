import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.random.seed(10)

# import personal packages
import utils.sigma_utils as sigma_utils
from utils.utils import *
from utils.MCMC import run_mcmc_with_gibbs

# initialize size of simulated data
T = ["m", "s"]
n_t = [100, 16]
blocks = [4, 2]
assert len(T) == len(n_t)
assert len(n_t) == len(blocks)
for i in range(len(n_t)):
    assert n_t[i] % blocks[i] == 0

print("Simulating data from prior...")
mu, alpha, beta = simulate_from_prior(T, n_t)
sigma = simulate_from_prior_sigma(T, n_t, blocks)
sigma_blocks = simulate_sigma_blocks(blocks)
sigma_blocks_multivar = simulate_multivariate_normal_distribution(sigma_blocks)

sum_mus = compute_sum_of_mus(mu)
epsilon_c = sigma_utils.compute_epsilon(alpha, sigma)
epsilon_c_multivar = [simulate_multivariate_normal_distribution(i) for i in epsilon_c]

# simulate X
prob_X = compute_prob_X(sum_mus, epsilon_c_multivar, sigma_blocks_multivar, blocks, n_t)
x = simulate_X(prob_X)

# get probability of Lotus and simulate Lotus
prob_Lotus, n_papers, gamma, delta = compute_prob_L(x)
lotus_binary, lotus_n_papers = simulate_lotus(prob_Lotus, n_papers)

# Run the MCMC chain
n_iter = 20000
gamma_init = 1
delta_init = 1
x_init = np.zeros_like(lotus_binary, dtype=np.float64)
# x_init = x
print("Running MCMC... ")
samples, x_samples, accept_gamma, accept_delta = run_mcmc_with_gibbs(
    lotus_n_papers,
    x_init,
    n_iter,
    gamma_init,
    delta_init,
    sum_mus,
    epsilon_c_multivar,
    sigma_blocks_multivar,
    blocks,
)

burn_in = int(0.5 * n_iter)  # Remove the first 50% of the samples
post_burn_in_samples = samples[burn_in:]
x_samples_burn = x_samples[burn_in:]
average_x = np.sum(x_samples_burn, axis=0) / burn_in

# Extract the posterior mean estimates for gamma and delta
gamma_posterior_mean = np.mean(post_burn_in_samples[:, 0])
delta_posterior_mean = np.mean(post_burn_in_samples[:, 1])

print("True gamma: ", gamma)
print("Estimated gamma: ", gamma_posterior_mean)
print("True delta: ", delta)
print("Estimated delta: ", delta_posterior_mean)
print("rate accept gamma : ", accept_gamma)
print("rate accept delta : ", accept_delta)

# print correlation between calculated probabilites of X and true values of X
print(np.corrcoef(x.flatten(), average_x.flatten()))

sns.set(style="darkgrid")
fig, axs = plt.subplots(ncols=2)
sns.scatterplot(
    x=range(len(post_burn_in_samples)),
    y=post_burn_in_samples[:, 0],
    ax=axs[0],
).set_title(f"True gamma : {gamma[0]}")
sns.lineplot(
    x=range(len(post_burn_in_samples)),
    y=[float(gamma) for i in range(len(post_burn_in_samples[:, 0]))],
    ax=axs[0],
    color="r",
)

sns.scatterplot(
    x=range(len(post_burn_in_samples)),
    y=post_burn_in_samples[:, 1],
    ax=axs[1],
).set_title(f"True delta : {delta[0]}")
sns.lineplot(
    x=range(len(post_burn_in_samples)),
    y=[float(delta) for i in range(len(post_burn_in_samples[:, 1]))],
    ax=axs[1],
    color="r",
)
plt.show()
