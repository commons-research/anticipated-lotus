import numpy as np
import scipy.stats as stats

def create_blocks(t, n, block_count):
    block_size = n // block_count
    blocks = []
    for i in range(block_count):
        start = i * block_size
        end = start + block_size if i != block_count - 1 else n
        block = stats.wishart.rvs(df=end - start, scale=1/(end - start) * np.eye(end - start), size=1).astype("float32")
        blocks.append(block)
    return blocks

def find_block_idx(element_idx, block_indices):
    for idx, (start, end) in enumerate(block_indices):
        if start <= element_idx < end:
            return idx
    return -1