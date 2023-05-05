import numpy as np
import scipy.stats as stats
import scipy.special
import itertools

def _create_blocks(t, n, block_count):
    block_size = n // block_count
    blocks = []
    for i in range(block_count):
        start = i * block_size
        end = start + block_size if i != block_count - 1 else n
        block = stats.wishart.rvs(df=end - start, scale=1/(end - start) * np.eye(end - start), size=1).astype("float32")
        blocks.append(block)
    return blocks

def simulate_from_prior(T, n_t) -> any:
    mu = {T[i]: np.random.normal(loc=0, scale=1, size=n_t[i]).astype("float32") for i in range(len(T))}
    
    alpha = {T[i]: np.random.exponential(scale=2, size=1).astype("float32") for i in range(len(T))}
    
    beta = {T[i]: np.random.exponential(scale=3, size=n_t[i]).astype("float32") for i in range(len(T))}
    return mu, alpha, beta

def simulate_from_prior_sigma(T, n_t, blocks):

    sigma = {T[i]: {f"{T[i]}_{j}": block for j, block in enumerate(_create_blocks(T[i], n_t[i], blocks[i]))} for i in range(len(T))}
    
    return sigma

def simulate_sigma_blocks(blocks):
        return stats.wishart.rvs(df=np.prod(blocks),
                                 scale=1/np.prod(blocks)*np.eye(np.prod(blocks)),
                                 size=1).astype("float32")

def compute_sum_of_mus(mu: dict) -> np.ndarray:
    arrays = list(mu.values())

    # Create an iterator for every combination of values
    combinations = itertools.product(*arrays)
    
    # Sum the values in each combination and store in a list
    sum_combinations = [np.sum(comb) for comb in combinations]
    
    # Convert the list to a numpy array with float32 dtype
    sum_combinations = np.array(sum_combinations, dtype="float32")
    
    return sum_combinations

def simulate_multivariate_normal_distribution(epsilon: np.ndarray) -> np.ndarray:
    return np.random.multivariate_normal(mean=np.zeros(epsilon.shape[0]), cov=epsilon)

def find_mu_blocks(mu_flat, blocks, original_shape):
    block_rows, block_cols = blocks
    row_size, col_size = original_shape

    rows_per_block = row_size // block_rows
    cols_per_block = col_size // block_cols

    row_indices, col_indices = np.unravel_index(np.arange(mu_flat.size), original_shape)
    block_row_indices = row_indices // rows_per_block
    block_col_indices = col_indices // cols_per_block

    mu_blocks = block_row_indices * block_cols + block_col_indices

    return mu_blocks.tolist()

def find_corresponding_sigmas(mu_flat, blocks, original_shape, sigmas):
    block_rows, block_cols = blocks
    row_size, col_size = original_shape

    rows_per_block = row_size // block_rows
    cols_per_block = col_size // block_cols

    row_indices, col_indices = np.unravel_index(np.arange(mu_flat.size), original_shape)
    block_row_indices = row_indices // rows_per_block
    block_col_indices = col_indices // cols_per_block
    within_block_row_indices = row_indices % rows_per_block
    within_block_col_indices = col_indices % cols_per_block

    block_numbers = block_row_indices * block_cols + block_col_indices
    within_block_indices = within_block_row_indices * cols_per_block + within_block_col_indices

    sigmas_array = np.array(sigmas)
    corresponding_sigmas = sigmas_array[block_numbers, within_block_indices]

    return corresponding_sigmas.tolist()

def compute_prob_X(sum_mus, sigmas, sigma_blocks, blocks, dim_size):
    in_what_block_is_mu = np.array(find_mu_blocks(sum_mus, blocks, dim_size))
    corresponding_sigmas = np.array(find_corresponding_sigmas(sum_mus, blocks, dim_size, sigmas))
    
    result = sum_mus + np.array(sigma_blocks)[in_what_block_is_mu] + corresponding_sigmas

    return scipy.special.expit(result.reshape(dim_size))

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
    np.random.seed(10)
    gamma = np.random.exponential(scale=0.1, size=1)
    delta = np.random.exponential(scale=0.01, size=1)
    P_m = np.sum(n_papers, axis=1)
    Q_s = np.sum(n_papers, axis=0)

    R_ms = 1 - np.exp(-gamma * P_m[:, None] - delta * Q_s[None, :], dtype="float32")
    prob_L = np.where(x == 1, R_ms, 0)
    
    return prob_L.astype("float32"), n_papers.astype('float32'), gamma, delta

def simulate_lotus(prob_lotus: np.ndarray, n_papers) -> np.ndarray: 
    Lotus_binary = np.random.binomial(n=1, p=prob_lotus, size=prob_lotus.shape).astype('uint8')

    not_present_in_lotus = np.where(Lotus_binary == 0)
    #Lotus_N_papers = np.zeros(Lotus_binary.shape)
    Lotus_N_papers = n_papers
    Lotus_N_papers[not_present_in_lotus] = 0
    
    return Lotus_binary, Lotus_N_papers.astype('int32')