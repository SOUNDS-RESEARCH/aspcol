import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest

import aspsim.signal.sources as sources
import aspcol.filterclasses as fc
import aspcol.utilities as util


@hyp.settings(deadline=None)
@hyp.given(num_ch = st.integers(min_value=1, max_value=5),
            num_out = st.integers(min_value=1, max_value=5), 
            nfft_exp = st.integers(min_value=3, max_value=7))
def test_wola_perfect_reconstruction(num_ch, num_out, nfft_exp):
    block_size = 2 ** nfft_exp
    overlap = block_size // 2
    num_blocks = 10
    #block_size = 32
    num_samples = num_blocks * block_size

    src = sources.WhiteNoiseSource(num_ch,1)
    wola = fc.WOLA(num_ch, num_out, block_size, overlap)

    sig = src.get_samples(num_samples)
    sig_out = np.zeros((num_ch, num_out, num_samples))

    for i in util.block_process_idxs(num_samples, wola.hop, 0):
        wola.analysis(sig[:,i:i+wola.hop])
        sig_out[:,:,i:i+wola.hop] = wola.synthesis()

    sig_in = sig[:,overlap:-block_size]
    sig_out = sig_out[:,:,overlap:-block_size]

    for i in range(num_out):
        assert np.allclose(sig_in[:,:-wola.hop], sig_out[:,i,wola.hop:])
