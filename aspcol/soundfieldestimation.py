"""Collection of algorithms for sound field estimation

* Kernel interpolation [uenoKernel2018]
* Infinite dimensional spherical harmonic analysis for moving microphones [brunnstromBayesianSubmitted]
* Spatial spectrum estimation for moving microphones [katzbergSpherical2021]

References
----------
[uenoKernel2018] N. Ueno, S. Koyama, and H. Saruwatari, “Kernel ridge regression with constraint of Helmholtz equation for sound field interpolation,” in 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan: IEEE, Sep. 2018, pp. 436–440. doi: 10.1109/IWAENC.2018.8521334. `[link] <https://doi.org/10.1109/IWAENC.2018.8521334>`__ \n
[brunnstromBayesianSubmitted] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted. \n
[katzbergSpherical2021] F. Katzberg, M. Maass, and A. Mertins, “Spherical harmonic representation for dynamic sound-field measurements,” in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Jun. 2021, pp. 426–430. doi: 10.1109/ICASSP39728.2021.9413708. `[link] <https://doi.org/10.1109/ICASSP39728.2021.9413708>`__ \n
"""
import numpy as np
import scipy.spatial.distance as spdist

import aspcol.kernelinterpolation as ki
import aspcol.pseq as pseq
import aspcol.sphericalharmonics as sph
import aspcol.fouriertransform as ft



#============= FREQUENCY DOMAIN METHODS - STATIONARY MICROPHONES =============
def est_ki_diffuse_freq(p_freq, pos, pos_eval, k, reg_param):
    """Estimates the RIR in the frequency domain using kernel interpolation
    
    Uses the frequency domain sound pressure as input

    Parameters
    ----------
    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    k : ndarray of shape (num_freq)
        wavenumbers
    reg_param : float
        regularization parameter for kernel interpolation

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    References
    ----------
    [uenoKernel2018]
    """
    est_filt = ki.get_krr_parameters(ki.kernel_helmholtz_3d, reg_param, pos_eval, pos, k)
    p_ki = est_filt @ p_freq[:,:,None]
    return np.squeeze(p_ki, axis=-1)


def nearest_neighbour_freq(p_freq, pos, pos_eval):
    """
    Estimates the sound field at the evaluation points by simply selecting the value
    associated with the nearest microphone position. 

    Parameters
    ----------
    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    dist = spdist.cdist(pos, pos_eval)
    min_idx = np.argmin(dist, axis=0)

    num_real_freqs = p_freq.shape[0]
    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for pos_idx in range(pos_eval.shape[0]):
        est_sound_pressure[:,pos_idx] = p_freq[:,min_idx[pos_idx]]
    return est_sound_pressure


def inf_dim_shd_analysis(p_freq, pos, pos_eval, wave_num, dir_coeffs, reg_param):
    """Estimates the sound field using directional microphones 

    Since we reconstruct the sound pressure immediately, without explicitly computing
    the spherical harmonic coefficients, no truncation or expansion center must be set.
    
    Parameters
    ----------
    p_freq : ndarray of shape (num_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    wave_num : ndarray of shape (num_freq)
        wavenumbers
    dir_coeffs : ndarray of shape (num_freq, num_mic, num_coeffs)
        harmonic coefficients of the directionality of the microphones
    reg_param : float
        regularization parameter. Must be non-negative
    
    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    num_mic = pos.shape[0]

    psi = sph.translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, wave_num)
    psi_plus_noise_cov = psi + np.eye(num_mic) * reg_param
    regression_vec = np.linalg.solve(psi_plus_noise_cov, p_freq)[...,None]

    omni_dir = sph.directivity_omni()
    estimator_matrix = sph.translated_inner_product(pos_eval, pos, omni_dir, dir_coeffs, wave_num)
    est_sound_pressure = np.squeeze(estimator_matrix @ regression_vec, axis=-1)
    return est_sound_pressure
# ============= TIME DOMAIN METHODS - STATIONARY MICROPHONES =============


def pseq_nearest_neighbour(p, seq, pos, pos_eval):
    """
    Estimates the sound field at the evaluation points by simply selecting the value
    associated with the nearest microphone position. 
    Assumes that the sequence is a perfect periodic sequence.

    Parameters
    ----------
    p : ndarray of shape (num_mic, seq_len)
        sound pressure in time domain at num_mic microphone positions
    seq : ndarray of shape (seq_len) or (1, seq_len)
        training signal used for the measurements
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    rir = pseq.decorrelate(p, seq)
    rir_freq = ft.rfft(rir)

    dist = spdist.cdist(pos, pos_eval)
    min_idx = np.argmin(dist, axis=0)

    num_real_freqs = rir_freq.shape[0]
    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for pos_idx in range(pos_eval.shape[0]):
        est_sound_pressure[:,pos_idx] = rir_freq[:,min_idx[pos_idx]]
    return est_sound_pressure


def est_ki_diffuse(p, seq, pos, pos_eval, samplerate, c, reg_param):
    """
    Estimates the RIR in the frequency domain using kernel interpolation
    Assumes seq is a perfect periodic sequence

    Parameters
    ----------
    p : ndarray of shape (M, seq_len)
        sound pressure in time domain at M microphone positions
    seq : ndarray of shape (seq_len)
        the training signal used for the measurements
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    samplerate : int
    c : float
        speed of sound
    reg_param : float
        regularization parameter for kernel interpolation

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    References
    ----------
    [uenoKernel2018]
    """
    rir = pseq.decorrelate(p, seq)

    fft_len = rir.shape[-1]
    rir_freq = ft.rfft(rir)
    k = ft.get_real_wavenum(fft_len, samplerate, c)

    return est_ki_diffuse_freq(rir_freq, pos, pos_eval, k, reg_param)










