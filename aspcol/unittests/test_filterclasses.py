import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest

import ancsim.signal.sources as sources
import aspcol.filterclasses as fc
import aspcol.utilities as util


@hyp.settings(deadline=None)
@hyp.given(num_ch = st.integers(min_value=1, max_value=5), 
            nfft_exp = st.integers(min_value=3, max_value=7))
def test_wola_perfect_reconstruction(num_ch, nfft_exp):
    block_size = 2 ** nfft_exp
    overlap = block_size // 2
    num_blocks = 10
    #block_size = 32
    num_samples = num_blocks * block_size

    src = sources.WhiteNoiseSource(num_ch,1)
    wola = fc.WOLA(num_ch, block_size, overlap)

    sig = src.get_samples(num_samples)
    sig_out = np.zeros((num_ch, num_samples))

   

    for i in util.block_process_idxs(num_samples, block_size, overlap):
        wola.analysis(sig[:,i:i+block_size])
        sig_out[:,i:i+block_size] = wola.synthesis()

    sig_in = sig[:,overlap:-block_size]
    sig_out = sig_out[:,overlap:-block_size]

    assert np.allclose(sig_in, sig_out)
