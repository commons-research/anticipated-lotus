import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import personal packages
import utils.sigma_utils as sigma_utils
from utils.utils import *
from utils.MCMC import run_mcmc

# initialize size of simulated data
T = ["m", "s"]
n_t = [100, 10]
blocks = [4, 2]
assert len(T) == len(n_t)
assert len(n_t) == len(blocks)
for i in range(len(n_t)):
    assert n_t[i] % blocks[i] == 0

print("Simulating data from prior...")
mu, alpha, beta = simulate_from_prior(T, n_t)
sigma = simulate_from_prior_sigma(T, n_t, blocks)
sigma_blocks = simulate_sigma_blocks(blocks)
sigma_blocks_multivar = np.random.multivariate_normal(
    mean=np.zeros(sigma_blocks.shape[0]), cov=sigma_blocks
)

sum_mus = compute_sum_of_mus(mu)
epsilon_c = sigma_utils.compute_epsilon(alpha, sigma)
epsilon_c_multivar = [
    np.random.multivariate_normal(mean=np.zeros(i.shape[0]), cov=i) for i in epsilon_c
]

# simulate X
prob_X = compute_prob_X(sum_mus, epsilon_c_multivar, sigma_blocks_multivar, blocks, n_t)
x = simulate_X(prob_X)

# get probability of Lotus and simulate Lotus
prob_Lotus, n_papers, gamma, delta = compute_prob_L(x)
lotus_binary, lotus_n_papers = simulate_lotus(prob_Lotus, n_papers)

print("Running MCMC...")
# Run the MCMC chain
n_iter = 20000
gamma_init = 1
delta_init = 1
# samples, accept_gamma, accept_delta = utils.MCMC.run_mcmc(lotus_n_papers, x, n_iter, gamma_init, delta_init)
samples, accept_gamma, accept_delta = run_mcmc(
    lotus_n_papers, x, n_iter, gamma_init, delta_init
)

burn_in = int(0.5 * n_iter)  # Remove the first 50% of the samples
post_burn_in_samples = samples[burn_in:]

# Extract the posterior mean estimates for gamma and delta
gamma_posterior_mean = np.mean(post_burn_in_samples[:, 0])
delta_posterior_mean = np.mean(post_burn_in_samples[:, 1])

print("True gamma: ", gamma)
print("Estimated gamma: ", gamma_posterior_mean)
print("True delta: ", delta)
print("Estimated delta: ", delta_posterior_mean)
print("rate accept gamma : ", accept_gamma)
print("rate accept delta : ", accept_delta)

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
