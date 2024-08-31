import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np
import aspcol.fouriertransform as ft


@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            fft_len = st.integers(min_value=16, max_value=256))
def test_fft_ifft_returns_original_signal(num_ch1, num_ch2, num_ch3, fft_len):
    rng = np.random.default_rng()
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, fft_len))
    fft_signal = ft.fft(signal)
    ifft_signal = ft.ifft(fft_signal)
    assert np.allclose(signal, ifft_signal)

@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            half_fft_len = st.integers(min_value=16, max_value=256))
def test_rfft_irfft_returns_original_signal(num_ch1, num_ch2, num_ch3, half_fft_len):
    rng = np.random.default_rng()
    fft_len = 2*half_fft_len
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, fft_len))
    fft_signal = ft.rfft(signal)
    ifft_signal = ft.irfft(fft_signal)
    assert np.allclose(signal, ifft_signal)

@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            sig_len = st.integers(min_value=16, max_value=256))
def test_rfft_irfft_with_padding_returns_original_signal(num_ch1, num_ch2, num_ch3, sig_len):
    rng = np.random.default_rng()
    fft_len = 2*sig_len
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, sig_len))
    fft_signal = ft.rfft(signal, n = fft_len)
    ifft_signal = ft.irfft(fft_signal)[...,:sig_len]
    assert np.allclose(signal, ifft_signal)



@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            fft_len = st.integers(min_value=16, max_value=256))
def test_rfft_equals_first_half_of_fft(num_ch1, num_ch2, num_ch3, fft_len):
    rng = np.random.default_rng()
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, fft_len))
    fft_signal = ft.fft(signal)
    rfft_signal = ft.rfft(signal)
    assert np.allclose(fft_signal[:rfft_signal.shape[0],...], rfft_signal)



@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            sig_len = st.integers(min_value=16, max_value=256))
def test_convolution_with_rfft_gives_real_valued_output(num_ch1, sig_len):
    rng = np.random.default_rng()
    fft_len = 2*sig_len
    signal = rng.normal(size=(num_ch1, sig_len))
    fft_signal = ft.rfft(signal, n = fft_len)
    freq_filter = rng.normal(size = fft_signal.shape) + 1j * rng.normal(size = fft_signal.shape)
    freq_filter[0,:] = np.real(freq_filter[0,:])
    freq_filter[-1,:] = np.real(freq_filter[-1,:])

    filtered_sig = fft_signal * freq_filter
    ifft_signal = ft.irfft(filtered_sig)[...,:sig_len]
    assert np.allclose(0, np.imag(ifft_signal))


@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            fft_len = st.integers(min_value=16, max_value=256))
def test_multiplying_by_dft_vector_gives_same_result_as_fft(num_ch1, num_ch2, num_ch3, fft_len):
    rng = np.random.default_rng()
    signal = rng.normal(size=(num_ch1, num_ch2, num_ch3, fft_len))
    fft_signal = ft.fft(signal)
    for n in range(fft_len):
        manual_dft = np.sum(signal * ft.dft_vector(n, fft_len)[None,None,None,:], axis=-1)
        assert np.allclose(fft_signal[n,...], manual_dft)
    
@hyp.settings(deadline=None)
@hyp.given(num_ch1 = st.integers(min_value=1, max_value=3),
            num_ch2 = st.integers(min_value=1, max_value=3), 
            num_ch3 = st.integers(min_value=1, max_value=3), 
            fft_len = st.integers(min_value=16, max_value=256))
def test_multiplying_by_idft_vector_gives_same_result_as_ifft(num_ch1, num_ch2, num_ch3, fft_len):
    rng = np.random.default_rng()
    freq_signal = rng.normal(size=(fft_len, num_ch1, num_ch2, num_ch3))
    signal = ft.ifft(freq_signal)
    for n in range(fft_len):
        manual_idft = np.sum(freq_signal * ft.idft_vector(n, fft_len)[:,None,None,None], axis=0)
        assert np.allclose(signal[...,n], manual_idft)





@hyp.settings(deadline=None)
@hyp.given(samplerate = st.integers(min_value=1, max_value=128),
           fft_len = st.integers(min_value=1, max_value=128))
def test_get_real_freqs_is_equivalent_to_np_rfftfreq(samplerate, fft_len):
    #ng = np.random.default_rng()
    freqs = ft.get_real_freqs(fft_len, samplerate)
    np_freqs = np.fft.rfftfreq(fft_len, 1/samplerate)
    assert np.allclose(freqs, np_freqs)