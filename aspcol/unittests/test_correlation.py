import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest

import ancsim.signal.sources as sources
import ancsim.signal.correlation as cr


@hyp.settings(deadline=None)
@hyp.given(max_lag = st.integers(min_value=1, max_value=5),
            num_channels = st.integers(min_value=1, max_value=3),
            amp = st.floats(min_value=1, max_value=1))
def test_autocorr_pulse_train_gives_identity_single_block(max_lag, num_channels, amp):
    
    #rng = np.random.default_rng()
    #amp = rng.normal(loc=1, scale=0.5)

    src = sources.PulseTrain(num_channels, amp, max_lag, 0)
    #src = sources.WhiteNoiseSource(num_channels, 1, rng)
    num_samples = 15 * max_lag
    sig = src.get_samples(num_samples)

    ac = cr.Autocorrelation(1, max_lag, num_channels)
    ac.update(sig)
    r = ac.corr.state
    r2 = cr.autocorr_ref(sig, max_lag)
    r3 = cr.autocorr_ref_spsig(sig, max_lag)
    assert np.allclose(r, r2)
    assert np.allclose(r2, r3)
    assert np.allclose(r[:,:,0], np.ones((num_channels, num_channels)) * amp**2 / max_lag )
    assert np.allclose(r[:,:,1:], np.zeros((num_channels, num_channels, max_lag-1)))


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5),
            max_lag = st.integers(min_value=1, max_value=5),
            num_channels = st.integers(min_value=1, max_value=3),
            amp = st.floats(min_value=1, max_value=1))
def test_autocorr_pulse_train_gives_identity_block_processing(bs, max_lag, num_channels, amp):
    src = sources.PulseTrain(num_channels, amp, max_lag, 0)
    num_samples = 5 * max_lag
    sig = src.get_samples(num_samples)

    ac = cr.Autocorrelation(1, max_lag, num_channels)
    for i in range(num_samples // bs):
        ac.update(sig[:,i*bs:(i+1)*bs])
    ac.update(sig[:,(i+1)*bs:])

    r = ac.corr.state
    r2 = cr.autocorr_ref(sig, max_lag)
    assert np.allclose(r, r2)
    assert np.allclose(r[:,:,0], np.ones((num_channels, num_channels)) * amp**2 / max_lag )
    assert np.allclose(r[:,:,1:], np.zeros((num_channels, num_channels, max_lag-1)))


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5),
            max_lag = st.integers(min_value=5, max_value=5),
            num_channels = st.integers(min_value=2, max_value=5),
            amp = st.floats(min_value=1, max_value=1))
def test_autocorr_delayed_pulse_train_gives_offdiagonal(bs, max_lag, num_channels, amp):
    dly=1
    max_dly = abs((num_channels-1) * dly)
    period = max_lag + max_dly
    src = sources.PulseTrain(num_channels, amp, period, np.arange(num_channels)*dly)
    num_samples = 5 * period
    sig = src.get_samples(num_samples)

    ac = cr.Autocorrelation(1, max_lag, num_channels)
    for i in range(num_samples // bs):
        ac.update(sig[:,i*bs:(i+1)*bs])
    ac.update(sig[:,(i+1)*bs:])

    r = ac.corr.state
    r2 = cr.autocorr_ref(sig, max_lag)
    assert np.allclose(r, r2)
    for i in range(max_lag):
        assert np.allclose(r[:,:,i], np.eye(num_channels, k=-i*dly)*amp**2/period)



@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5),
            max_lag = st.integers(min_value=5, max_value=5),
            num_channels = st.integers(min_value=2, max_value=2),
            amp = st.floats(min_value=1, max_value=1))
def test_autocorr_mat_delayed_pulse_train(bs, max_lag, num_channels, amp):
    dly=1
    max_dly = abs((num_channels-1) * dly)
    period = max_lag + max_dly
    src = sources.PulseTrain(num_channels, amp, period, np.arange(num_channels)*dly)
    num_samples = 5 * period
    sig = src.get_samples(num_samples)

    ac = cr.Autocorrelation(1, max_lag, num_channels)
    ac.update(sig)
    R = ac.get_corr_mat()
    R2 = cr.corr_mat_new_first_ref(sig, max_lag)
    R3 = cr.corr_matrix(sig, sig, max_lag, max_lag)
    assert np.allclose(R, R2)
    assert np.allclose(R, R3)

    R = ac.get_corr_mat(new_first=False)
    R2 = cr.corr_mat_old_first_ref(sig, max_lag)
    assert np.allclose(R, R2)

    #assert False
    #for i in range(max_lag):
    #    assert np.allclose(r[:,:,i], np.eye(num_channels, k=-i*dly)*amp**2/period)