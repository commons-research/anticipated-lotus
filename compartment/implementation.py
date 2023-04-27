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

sigma = simulate_from_prior_sigma(T, n_t, blocks) 

alpha = {'m': 1, 
         's': 1}
