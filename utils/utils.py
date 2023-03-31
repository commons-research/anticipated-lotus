import numpy as np
import scipy.stats as stats
import scipy.special
import itertools

def simulate_from_prior(T, n_t) -> any:
    mu = {T[i]: np.random.normal(loc=0, scale=1, size=n_t[i]).astype("float32") for i in range(len(T))}
    
    alpha = {T[i]: np.random.exponential(scale=2, size=1).astype("float32") for i in range(len(T))}
    
    beta = {T[i]: np.random.exponential(scale=3, size=n_t[i]).astype("float32") for i in range(len(T))}
    
    sigma = {T[i]: stats.wishart.rvs(df=n_t[i],
                                     scale=1/n_t[i]*np.eye(n_t[i]),
                                     size=1).astype("float32") for i in range(len(T))}
    return mu, alpha, beta, sigma

def compute_sum_of_mus(mu: dict) -> np.ndarray:
    arrays = list(mu.values())

    # Create an iterator for every combination of values
    combinations = itertools.product(*arrays)
    
    # Sum the values in each combination and store in a list
    sum_combinations = [np.sum(comb) for comb in combinations]
    
    # Convert the list to a numpy array with float32 dtype
    sum_combinations = np.array(sum_combinations, dtype="float32")
    
    return sum_combinations

def simulate_epsilon_c(epsilon: np.ndarray) -> np.ndarray:
    return np.random.multivariate_normal(mean=np.zeros(epsilon.shape[0]), cov=epsilon)

def compute_prob_X(mus: np.ndarray, epsilon_c: np.ndarray, dim_size:list) -> np.ndarray :
    result = mus + epsilon_c
    result = result.reshape(dim_size)
    return scipy.special.expit(result)

def simulate_X(prob_x: np.ndarray) -> np.ndarray :
    return np.random.binomial(n=1, p=prob_x, size=prob_x.shape).astype('uint8')

#look at the condition in the model to calculate the proba of L_sm
def compute_prob_L(x):
    n_papers = 1 + np.random.poisson(lam=1, size=x.shape)
    
    #set presence or absence in nature
    not_present_in_nature = np.where(x == 0)
    present_in_x = np.where(x == 1)
    
    #change number of papers to 0 in the n_papers array if the entry is not present in nature
    n_papers[not_present_in_nature] = 0

    #generate gamma and delta, and calculate the 
    gamma = np.random.exponential(scale=0.5, size=1)
    delta = np.random.exponential(scale=0.1, size=1)
    P_m = np.sum(n_papers, axis=1)
    Q_s = np.sum(n_papers, axis=0)
    
    prob_L = np.zeros(x.shape)
    R_ms = 1 - np.exp(-gamma * P_m[:, None] - delta * Q_s[None, :], dtype="float32")


    prob_L[not_present_in_nature] = 0
    
    prob_L[present_in_x] = R_ms[present_in_x]
    
    return prob_L.astype("float32"), n_papers.astype('float32'), gamma, delta

def simulate_lotus(prob_lotus: np.ndarray, n_papers) -> np.ndarray: 
    Lotus_binary = np.random.binomial(n=1, p=prob_lotus, size=prob_lotus.shape).astype('uint8')

    not_present_in_lotus = np.where(Lotus_binary == 0)
    #Lotus_N_papers = np.zeros(Lotus_binary.shape)
    Lotus_N_papers = n_papers
    Lotus_N_papers[not_present_in_lotus] = 0
    
    return Lotus_binary, Lotus_N_papers.astype('int32')
