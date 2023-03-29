import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#import personal packages
import utils.sigma_utils as sigma_utils
from utils.utils import *

#initialize size of simulated data
T = ['m', 's']
n_t = [100, 10]
assert len(T)==len(n_t)

#simulate priors
print("generating prior...")
mu, alpha, beta, sigma = simulate_from_prior(T, n_t)

gamma = np.random.exponential(scale=1, size=1)
delta = np.random.exponential(scale=0.1, size=1)
Q_s = 1 + np.random.poisson(lam=3, size=n_t[1])
P_m = 1 + np.random.poisson(lam=2, size=n_t[0])

#calculate sum of all possible combination of mus
sum_mu = compute_sum_of_mus(mu)

print("Calculating epsilon_c...")
epsilon_matrix = sigma_utils.compute_epsilon(alpha, sigma)
epsilon_c = simulate_epsilon_c(epsilon_matrix)

#get probability of x and simulate x
prob_x = compute_prob_X(sum_mu, epsilon_c, n_t)
x = simulate_X(prob_x)

#get probability of Lotus and simulate Lotus
prob_Lotus = compute_prob_L(x, gamma, delta, P_m, Q_s)
lotus = simulate_lotus(prob_Lotus)


## DONE simulate \epsilson_c from multivariate normal distribution with mean = 0 and cov epsilon 
## DONE Once we have epsilon_c and our sum of mus, we have the probability of X. We can draw X with a Bernoulli varaibles and we get our data X. 
## DONE we can simulate the number of papers with a Poisson distribution --> 1+ Poisson
## DONE We can then simulate the Lotus data by drawing from X. 
## DONE from x simulate Lotus from x and P and Q --> R 
## DONE from R draw Bernoulli with P = R
## --> simulated data

# try to infer \gamma and \delta from MCMC --> should find our \delta and \gamma