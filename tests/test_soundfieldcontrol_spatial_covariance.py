import pytest
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hypnp

import numpy as np
import aspcol.soundfieldcontrol as sfc
import matrix_strategies as mst




def test_frequency_domain_spatial_covariance_functions_give_save_results():
    rng = np.random.default_rng()
    num_src = 5
    num_freq = 32
    num_mic = 8

    rir_freq = rng.normal(size=(num_freq, num_src, num_mic)) + 1j*rng.normal(size=(num_freq, num_src, num_mic))

    cov1 = sfc.spatial_cov_freq(rir_freq)
    cov2, _ = sfc.spatial_cov_freq_superpos(np.moveaxis(rir_freq, 1,2), np.moveaxis(rir_freq, 1, 2))

    assert np.allclose(cov1, cov2.conj())