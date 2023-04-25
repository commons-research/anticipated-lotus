import numpy as np
import itertools
from numba import jit, prange, config, set_num_threads, get_num_threads
config.THREADING_LAYER = 'threadsafe'
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def prepare_sigma(alpha: dict, sigma: dict):
    arrays = {key: alpha[key] * sigma[key] for key in alpha.keys()}
    cov_matrices = list(arrays.values())
    matrix_dims = [matrix.shape[0] for matrix in cov_matrices]
    index_list = list(itertools.product(*[range(dim) for dim in matrix_dims] * 2))
    return cov_matrices, np.array(matrix_dims), np.array(index_list)

@jit(nopython=True, cache=True)
def ravel_multi_index(coords, dims):
    idx = 0
    for i in range(len(coords)):
        idx *= dims[i]
        idx += coords[i]
    return idx

set_num_threads(config.NUMBA_DEFAULT_NUM_THREADS - 1)
@jit(nopython=True, parallel=True, cache=True)
def sigma_numba(cov_matrices:list, matrix_dims: np.ndarray, index_list: np.ndarray) -> np.ndarray :
    summed_matrix_dim = np.prod(matrix_dims)
    
    # Initialize the summed matrix
    summed_matrix = np.zeros((summed_matrix_dim, summed_matrix_dim))
    
    #index_list = np.array(index_list)
    for i in prange(len(index_list)):
        input_idx = index_list[i][:len(matrix_dims)]
        output_idx = index_list[i][len(matrix_dims):]
        
        # Compute the sum using a loop instead of a generator expression
        element_sum = 0
        for j, matrix in enumerate(cov_matrices):
            element_sum += matrix[input_idx[j], output_idx[j]]
        # Compute the linear index using the custom function
        input_linear_idx = ravel_multi_index(input_idx, matrix_dims)
        output_linear_idx = ravel_multi_index(output_idx, matrix_dims)
        
        summed_matrix[input_linear_idx, output_linear_idx] = element_sum
    
    return summed_matrix

def compute_epsilon(alpha: dict, sigma: dict) -> np.ndarray :
    cov_matrices, matrix_dims, index_list = prepare_sigma(alpha, sigma)

    return sigma_numba(cov_matrices, matrix_dims, index_list)