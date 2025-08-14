import numpy as np
import aspcore.fouriertransform as ft

import matplotlib.pyplot as plt

import aspcol.soundfieldestimation.moving_microphone as mm
import aspcol.soundfieldestimation_jax.moving_microphone_jax as mm_jax


def test_stft_bayesian_numpy_and_jax_are_equivalent():
    rng = np.random.default_rng()
    seq_len = 64
    num_periods = 3
    num_real_freqs = len(ft.get_real_freqs(seq_len, 1.0))
    signal = rng.normal(size=(seq_len))

    signal_f = mm._seq_stft_bayesian_multiperiod(signal, num_periods)[:num_real_freqs,...]
    signal_f_jax = mm_jax._seq_stft_bayesian_multiperiod(signal, num_periods)

    np.testing.assert_allclose(signal_f, signal_f_jax, rtol=1e-5, atol=1e-8)

def test_stft_krr_numpy_and_jax_are_equivalent():
    rng = np.random.default_rng()
    seq_len = 250
    num_periods = 10
    signal = rng.normal(size=(seq_len))

    signal_f = mm._seq_stft_krr_multiperiod(signal, num_periods)
    signal_f_jax = mm_jax._seq_stft_krr_multiperiod(signal, num_periods)

    np.testing.assert_allclose(signal_f, signal_f_jax, rtol=1e-5, atol=1e-8)


def test_stft_bayesian_and_stft_krr_numpy_are_scaled_conjugates():
    rng = np.random.default_rng()
    seq_len = 64
    num_periods = 3
    num_real_freqs = len(ft.get_real_freqs(seq_len, 1.0))

    signal = rng.normal(size=(seq_len))
    signal_f_bayesian = mm._seq_stft_bayesian_multiperiod(signal, num_periods)[:num_real_freqs,...]
    signal_f_krr = mm._seq_stft_krr_multiperiod(signal, num_periods)

    np.testing.assert_allclose(signal_f_bayesian, signal_f_krr.conj() / seq_len, rtol=1e-5, atol=1e-8)



def test_stft_bayesian_and_stft_krr_jax_are_scaled_conjugates():
    rng = np.random.default_rng()

    seq_len = 512
    num_periods = 3
    signal = rng.normal(size=(seq_len))

    signal_f_bayesian = mm_jax._seq_stft_bayesian_multiperiod(signal, num_periods)
    signal_f_krr = mm_jax._seq_stft_krr_multiperiod(signal, num_periods)

    np.testing.assert_allclose(signal_f_bayesian, signal_f_krr.conj() / seq_len, rtol=1e-4, atol=1e-6)