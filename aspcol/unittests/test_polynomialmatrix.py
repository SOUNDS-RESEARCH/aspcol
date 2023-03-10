from operator import matmul
import numpy as np
import hypothesis as hyp
import hypothesis.extra.numpy as hypnp
#from hypothesis import given
import hypothesis.strategies as st

import aspcol.polynomialmatrix as pmt

DTYPE = st.shared(st.sampled_from((float, complex)))
MAX_DIM = 5
MAX_TAPS = 5

@st.composite
def POLYMAT(draw, dtype=None, num_taps=None, mat_dim=None):
    if dtype is None:
        dtype = draw(DTYPE)

    if dtype == float:
        elements = st.floats(min_value=-1e10, max_value=1e10, allow_infinity=False, allow_nan=False)
    elif dtype == complex:
        elements = st.complex_numbers(max_magnitude=1e10, allow_infinity=False, allow_nan=False)
    else:
        raise ValueError("Invalid dtype")

    if num_taps is None:
        num_taps = 2*draw(st.integers(min_value=1, max_value=MAX_TAPS)) - 1
    else:
        assert num_taps % 2 == 1

    if mat_dim is None:
        mat_dim = draw(st.tuples(st.integers(min_value=1, max_value=MAX_DIM), 
                                st.integers(min_value=1, max_value=MAX_DIM)))

    return draw(hypnp.arrays(dtype=dtype, elements=elements, shape=(num_taps, mat_dim[0], mat_dim[1])))

@st.composite
def PARAHERMITIAN(draw, mat_dim = None, **kwargs):
    if mat_dim is None:
        mat_dim = draw(st.integers(min_value=1, max_value=MAX_DIM))
        mat_dim = (mat_dim, mat_dim)
    elif isinstance(mat_dim, int):
        mat_dim = (mat_dim, mat_dim)
    else:
        assert mat_dim[0] == mat_dim[1]

    mat = draw(POLYMAT(mat_dim=mat_dim, **kwargs))
    mat = (mat + pmt.paraconjugate(mat)) / 2
    assert pmt.is_parahermitian(mat)
    return mat

@st.composite
def COMPATIBLE_POLYMATS(draw, taps_l=None, taps_r=None, **kwargs):
    diml = draw(st.integers(min_value=1,max_value=MAX_DIM))
    dimr = draw(st.integers(min_value=1, max_value=MAX_DIM))
    inner_dim = draw(st.integers(min_value=1, max_value=MAX_DIM))
    
    matl = draw(POLYMAT( num_taps=taps_l, mat_dim = (diml, inner_dim), **kwargs))
    matr = draw(POLYMAT( num_taps=taps_r, mat_dim = (inner_dim, dimr), **kwargs))
    return matl, matr



@hyp.settings(deadline=None)
@hyp.given(COMPATIBLE_POLYMATS(taps_l=1, taps_r=1))
def test_matmul_single_tap_equals_normal_matmul(mats):
    mat1, mat2 = mats
    result_mat = pmt.matmul(mat1, mat2)
    compare_mat = mat1 @ mat2

    assert np.allclose(result_mat, compare_mat, equal_nan=True)

@hyp.settings(deadline=None)
@hyp.given(COMPATIBLE_POLYMATS(taps_r=1))
def test_matmul_by_single_tap_equals_normal_matmul_for_each_tap(mats):
    """
    mat2 is a single tap, so the result should be equal to normal 
        matmul for each tap of mat1
    """
    mat1, mat2 = mats
    result_mat = pmt.matmul(mat1, mat2)
    for tap in range(mat1.shape[0]):
        compare_mat = mat1[tap,:,:] @ mat2
        assert np.allclose(result_mat[tap,:,:], compare_mat, equal_nan=True)

@hyp.settings(deadline=None)
@hyp.given(mat1 = POLYMAT())
def test_matmul_by_unit_delay(mat1):
    delay_mat = np.zeros((3, mat1.shape[-1], mat1.shape[-1]), dtype=mat1.dtype)
    delay_mat[-1,:,:] = np.eye(mat1.shape[-1], dtype=mat1.dtype)
    result_mat = pmt.matmul(mat1, delay_mat)

    for tap in range(mat1.shape[0]):
        assert np.allclose(mat1[tap,:,:], result_mat[tap+2,:,:], equal_nan=True)



@hyp.settings(deadline=None)
@hyp.given(PARAHERMITIAN(num_taps=1))
def test_pevd_single_tap_equals_evd(mat):
    decomp = pmt.pevd_sbr2(mat, 1e-3, 100, 0.99)
