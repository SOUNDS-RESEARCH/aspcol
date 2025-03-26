import pytest
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hypnp
import hypothesis as hyp
import cvxpy as cp

import numpy as np
#import soundfieldcontrol.szc.sinr_opt as sinropt
#import soundfieldcontrol.szc.sinr_opt_time as sinrt

import aspcol.soundfieldcontrol as sfc
import matrix_strategies as mst


def get_random_spatial_cov_full_rank(num_freq, num_ls):
    rng = np.random.default_rng()
    rank = num_ls + 4
    R = np.zeros((num_freq, num_ls, num_ls), dtype=complex)
    for k in range(num_freq):
        for r in range(rank):
            v = rng.normal(size=(num_ls, 1)) + 1j*rng.normal(size=(num_ls, 1))
            R[k,:,:] += v @ v.conj().T

    assert np.allclose(R, np.moveaxis(R.conj(), 1,2))
    return R





@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
    st.floats(min_value=0.5, max_value=5),
)
def test_mwf_and_gevd_with_full_rank_are_equivalent(num_zones, num_ls, mu):
    Rb = get_random_spatial_cov_full_rank(num_zones, num_ls)
    Rd = get_random_spatial_cov_full_rank(num_zones, num_ls)

    mwf = sfc.szc_transform_mwf(Rb, Rd, mu)
    gevd = sfc.szc_transform_mwf_gevd(Rb, Rd, num_ls, mu)
    assert np.allclose(mwf, gevd)