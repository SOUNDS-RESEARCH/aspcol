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

import aspcore.pseq as pseq
import aspcore.fouriertransform as ft
import aspcore.montecarlo as mc
import aspcore.matrices as aspmat

import aspcol.kernelinterpolation as ki
import aspcol.sphericalharmonics as sph
import aspcol.planewaves as pw





#============= FREQUENCY DOMAIN METHODS - STATIONARY MICROPHONES =============
def est_ki_freq(p_freq, pos, pos_eval, wave_num, reg_param, kernel_func = None, kernel_args = None):
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
    # if kernel_func is None:
    #     kernel_func = ki.kernel_helmholtz_3d
    #     assert kernel_args is None, "kernel_args must be None if kernel_func is None"
    # if kernel_args is None:
    #     kernel_args = []

    a = ki.get_krr_params(p_freq, pos, wave_num, reg_param, kernel_func=kernel_func, kernel_args=kernel_args)
    p_ki = ki.reconstruct_freq(a, pos_eval, pos, wave_num, kernel_func, kernel_args)
    return p_ki


    # est_filt = ki.get_interpolation_params(kernel_func, reg_param, pos_eval, pos, k, *kernel_args)
    # if est_filt.ndim == 4:
    #     est_filt = np.squeeze(est_filt, axis=1)
    # p_ki = est_filt @ p_freq[:,:,None]
    # return np.squeeze(p_ki, axis=-1)


def est_ki_diffuse_freq(p_freq, pos, pos_eval, k, reg_param):
    """DEPRECATED: Use est_ki_freq instead. It defaults to the diffuse Helmholtz kernel, hence is equivalent. 
    Only kept for backwards compatibility.
    """
    return est_ki_freq(p_freq, pos, pos_eval, k, reg_param)


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




def est_ki_freq_rff(p_freq, pos, pos_eval, k, reg_param, num_basis = 64, rng = None, direction=None, beta = None):
    """Estimates the RIR in the frequency domain using random Fourier features
    
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
    num_basis : int
        number of basis directions to use for the random fourier features
    direction : ndarray of shape (3,)
        direction of the directional weighting. 
        This should be towards the source, i.e. in the opposite of the propagation direction
    beta : float
        strength of the directional component

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    Notes
    -----
    If direction is set, the basis directions are sampled from a von Mises-Fisher distribution, 
    with pdf p(x) = e^{-beta * direction^T x}.

    References
    ----------
    [uenoKernel2018]
    """
    if rng is None:
        rng = np.random.default_rng()

    if direction is None:
        assert beta is None, "beta must be None if direction is None"
        basis_directions = mc.uniform_random_on_sphere(num_basis, rng)
    else:
        assert beta is not None, "beta must be set if direction is set"
        basis_directions = mc.vonmises_fisher_on_sphere(num_basis, -direction, beta, rng)

    Z = pw.plane_wave(pos, basis_directions, k) / np.sqrt(num_basis)
    system_mat = np.moveaxis(Z.conj(), 1, 2) @ Z
    system_mat += reg_param * np.eye(num_basis, dtype=system_mat.dtype)[None,...]

    #system_mat = aspmat.regularize_matrix_with_condition_number(system_mat, 1e8) # safety to ensure the system is not singular
    
    projected_data = np.moveaxis(Z.conj(),1,2) @ p_freq[:,:,None]
    params = np.linalg.solve(system_mat, projected_data)

    z_eval = pw.plane_wave(pos_eval, basis_directions, k) / np.sqrt(num_basis)
    p_est = z_eval @ params
    return np.squeeze(p_est, axis=-1)











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


def est_ki(p, seq, pos, pos_eval, samplerate, c, reg_param, kernel_func = None, kernel_args = None):
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
    kernel_func : callable
        kernel function to use, defaults to the diffuse Helmholtz kernel.
    kernel_args : tuple
        additional arguments to pass to the kernel function, defaults to None.

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

    return est_ki_freq(rir_freq, pos, pos_eval, k, reg_param, kernel_func, kernel_args)

def est_ki_diffuse(p, seq, pos, pos_eval, samplerate, c, reg_param):
    """DEPRECATED: Use est_ki instead. It defaults to the diffuse Helmholtz kernel, hence is equivalent.
    Only kept for backwards compatibility.
    """
    return est_ki(p, seq, pos, pos_eval, samplerate, c, reg_param)



def est_ki_rff(p, seq, pos, pos_eval, samplerate, c, reg_param, kernel_func = None, kernel_args = None):
    """Estimates the RIR in the frequency domain using random Fourier features
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
    kernel_func : callable
        kernel function to use, defaults to the diffuse Helmholtz kernel.
    kernel_args : tuple
        additional arguments to pass to the kernel function, defaults to None.

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

    return est_ki_freq_rff(rir_freq, pos, pos_eval, k, reg_param, kernel_func, kernel_args)




