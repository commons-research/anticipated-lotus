import numpy as np
import scipy.stats as stats
import scipy.special
import itertools
from closures import *
np.random.seed(10)

def simulate_from_prior(T, n_t) -> any:
    mu = {T[i]: np.random.normal(loc=0, scale=1, size=n_t[i]).astype("float32") for i in range(len(T))}
    
    alpha = {T[i]: np.random.exponential(scale=2, size=1).astype("float32") for i in range(len(T))}
    
    beta = {T[i]: np.random.exponential(scale=3, size=n_t[i]).astype("float32") for i in range(len(T))}
    return mu, alpha, beta

def simulate_from_prior_sigma(T, n_t, blocks):

    sigma = {T[i]: {f"{T[i]}_{j}": block for j, block in enumerate(create_blocks(T[i], n_t[i], blocks[i]))} for i in range(len(T))}
    sigma['blocks'] = stats.wishart.rvs(df=np.prod(blocks),
                                        scale=1/np.prod(blocks)*np.eye(np.prod(blocks)),
                                        size=1).astype("float32")
    
    return sigma


def are_elements_in_same_block(array1, array2, element1_idx, element2_idx, element3_idx, element4_idx, block_indices):
    """
    This function checks if two elements of the first array and two elements of the second array are in the same blocks.
    
    Parameters:
    array1 (list): The first one-dimensional array.
    array2 (list): The second one-dimensional array.
    element1_idx (int): The index of the first element in array1.
    element2_idx (int): The index of the second element in array1.
    element3_idx (int): The index of the first element in array2.
    element4_idx (int): The index of the second element in array2.
    block_indices (list of tuples): List containing tuples with start and end indices of each block.

    Returns:
    int: Returns 1 if all elements are in the same block, otherwise returns 0.
    """

    block1_idx = find_block_idx(element1_idx, block_indices)
    block2_idx = find_block_idx(element2_idx, block_indices)
    block3_idx = find_block_idx(element3_idx, block_indices)
    block4_idx = find_block_idx(element4_idx, block_indices)

    if block1_idx == block2_idx and block3_idx == block4_idx and block1_idx == block3_idx:
        return True
    else:
        return False
