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

def research_effort(gamma, delta, P_m, Q_s):
    return 1 - np.exp(-gamma * P_m[:, None] - delta * Q_s[None, :])

#look at the condition in the model to calculate the proba of L_sm
def compute_prob_L(x, gamma, delta, P_m, Q_s):
    prob_L = np.zeros(x.shape)
    R_ms = 1 - np.exp(-gamma * P_m[:, None] - delta * Q_s[None, :], dtype="float32")
    
    not_present_in_x = np.where(x == 0)
    present_in_x = np.where(x == 1)
    
    prob_L[not_present_in_x] = 0
    prob_L[present_in_x] = R_ms[present_in_x]
    
    return prob_L.astype("float32")

def compute_sum_of_mus(mu: dict) -> np.ndarray:
    arrays = list(mu.values())

    # Create an iterator for every combination of values
    combinations = itertools.product(*arrays)
    
    # Sum the values in each combination and store in a list
    sum_combinations = [np.sum(comb) for comb in combinations]
    
    # Convert the list to a numpy array with float32 dtype
    sum_combinations = np.array(sum_combinations, dtype="float32")
    
    return sum_combinations

def compute_prob_X(mus: np.ndarray, epsilon_c: np.ndarray, dim_size:list) -> np.ndarray :
    result = mus + epsilon_c
    result = result.reshape(dim_size)
    return scipy.special.expit(result)

def simulate_X(prob_x: np.ndarray) -> np.ndarray :
    return np.random.binomial(n=1, p=prob_x, size=prob_x.shape).astype('uint8')

def simulate_epsilon_c(epsilon: np.ndarray) -> np.ndarray:
    return np.random.multivariate_normal(mean=np.zeros(epsilon.shape[0]), cov=epsilon)

def simulate_lotus(prob_lotus: np.ndarray) -> np.ndarray: 
    return np.random.binomial(n=1, p=prob_lotus, size=prob_lotus.shape).astype('uint8')