import pytest
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hypnp
import hypothesis as hyp

import numpy as np

MAX_DIM = 5



@st.composite
def hermitian(draw, dim=None):
    if dim is None:
        dim = draw(st.tuples(st.integers(min_value=1, max_value=MAX_DIM), st.integers(min_value=1, max_value=MAX_DIM)))
    matrix = draw(hypnp.arrays(dtype=complex, 
                    shape=dim,
                    elements=st.complex_numbers(allow_infinity=False, allow_nan=False)))

    return (matrix + matrix.conj().T) / 2

@st.composite
def symmetric(draw, dim=None):
    if dim is None:
        dim = draw(st.tuples(st.integers(min_value=1, max_value=MAX_DIM), st.integers(min_value=1, max_value=MAX_DIM)))
    matrix = draw(hypnp.arrays(dtype=float, 
                    shape=st.tuples(dim[0], dim[1]),
                    elements=st.floats(allow_infinity=False, allow_nan=False)))

    return (matrix + matrix.T) / 2


def psd_real(draw, dim=None, rank=None):
    """
    dim is an integer
    rank is an integer between 1 and dim 
    rank greater than dim is not strictly disallowed, 
    but makes no practical difference

    return positive semidefinite matrix of shape (dim, dim)
    """
    if dim is None:
        dim = draw(st.integers(min_value=1, max_value=MAX_DIM))

    if rank is None:
        rank = draw(st.integers(min_value=1, max_value=dim))
    
    MIN_EIGENVALUE = 1e-5
    MAX_EIGENVALUE = 1e5

    seed= draw(st.integers(min_value=1, max_value = 100000))
    rng = np.random.default_rng(seed)
    # eigenvectors = rng.normal(size=(dim, dim))
    # eigenvalues = draw(hypnp.arrays(dtype=float, shape=(dim), 
    #                     elements=st.floats(min_value=MIN_EIGENVALUE, max_value=MAX_EIGENVALUE, 
    #                                         allow_infinity=False, allow_nan=False)))
    mat = np.zeros((dim, dim), dtype=float)
    for r in range(rank):
        vec = rng.normal(size=(dim, 1))
        mat += vec @ vec.T
    return mat

# def get_random_spatial_cov(num_zones, num_ls):
#     rng = np.random.default_rng(SEED)
#     rank = rng.integers(1, num_ls)
#     R = np.zeros((num_zones, num_ls, num_ls), dtype=complex)
#     for k in range(num_zones):
#         for r in range(rank):
#             v = rng.normal(size=(num_ls, 1)) + 1j*rng.normal(size=(num_ls, 1))
#             R[k,:,:] += v @ v.conj().T

#     assert np.allclose(R, np.moveaxis(R.conj(), 1,2))
#     return R

# def get_random_beamformer(num_zones, num_ls):
#     rng = np.random.default_rng(SEED+10)
#     w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, num_ls))
#     return w

# def get_random_beamformer_normalized(num_zones, num_ls):
#     rng = np.random.default_rng(SEED+235)
#     w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, num_ls)) + 1j*rng.normal(scale=rng.uniform(high=5),size=(num_zones, num_ls))
#     w /= np.sqrt(np.sum(np.abs(w)**2, axis=-1, keepdims=True))
#     return w

# def get_random_noise_pow(num_zones):
#     rng = np.random.default_rng(SEED+2354)
#     return rng.uniform(low=1e-3, high=10, size=num_zones)

# def get_random_sinr_targets(num_zones):
#     rng = np.random.default_rng(SEED+12)
#     return rng.uniform(low=0.3, high=5, size=num_zones)

# def get_random_pow_vec(num_zones):
#     rng = np.random.default_rng(SEED+2358)
#     return rng.uniform(low=1e-3, high=10, size=num_zones)
