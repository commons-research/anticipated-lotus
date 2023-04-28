import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#import personal packages
import utils.sigma_utils as sigma_utils
from utils.utils import *
#from utils.MCMC import run_mcmc_with_gibbs

#initialize size of simulated data
T = ['m', 's']
n_t = [100, 10]
blocks = [2, 2]
assert len(T)==len(n_t)
assert len(n_t)==len(blocks)


mu, alpha, beta = simulate_from_prior(T, n_t)
sigma = simulate_from_prior_sigma(T, n_t, blocks) 
sigma_blocks = simulate_sigma_blocks(blocks)
sigma_blocks_multivar = np.random.multivariate_normal(mean=np.zeros(sigma_blocks.shape[0]), cov=sigma_blocks)

sum_mus = compute_sum_of_mus(mu)
epsilon_c = sigma_utils.compute_epsilon(alpha, sigma)
epsilon_c_multivar = [np.random.multivariate_normal(mean=np.zeros(i.shape[0]), cov=i) for i in epsilon_c]

#simulate X
prob_X = compute_prob_X(sum_mus, epsilon_c_multivar, sigma_blocks_multivar, blocks, n_t)
x = simulate_X(prob_X)

#get probability of Lotus and simulate Lotus
prob_Lotus, n_papers, gamma, delta = compute_prob_L(x)
lotus_binary, lotus_n_papers = simulate_lotus(prob_Lotus, n_papers)

print(x.shape)