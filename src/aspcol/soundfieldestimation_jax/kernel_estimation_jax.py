
import numpy as np
import scipy.linalg as splin

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


from functools import partial

import aspcol.kernelinterpolation_jax as kernel
import aspcore.fouriertransform_jax as ft
import aspcore.matrices_jax as aspmat

@partial(jax.jit, static_argnames=["batch_size"])
def reconstruct_diffuse(pos_eval, pos_mic, wave_num, krr_params, batch_size=20):
    """
    
    pos_eval : np.ndarray of shape (num_eval, 3)
        The position of the evaluation points.
    pos_mic : np.ndarray of shape (num_mics, 3)
        The position of the microphones.
    wave_num : np.ndarray of shape (num_real_freqs,)
        The wave numbers defined as 2 * np.pi * freqs / c
    krr_params : np.ndarray of shape (num_pos, ir_len) or (num_pos * ir_len,)
        The kernel ridge regression parameters, denoted by a in the paper [brunnströmTimedomain2025]'
    batch_size : int,
        The number of evaluation points to process in parallell. Default is 20.
        A higher number generally increases the speed, but also the memory usage, 
        the latter of which can be extremely high for many evaluation points. 

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.

    References
    ----------
    [brunnströmTimedomain2025]
    """
    num_mic = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    
    if krr_params.ndim == 1:
        krr_params = krr_params.reshape(num_mic, -1)
    assert krr_params.ndim == 2
    ir_len = krr_params.shape[-1]
    even_ir_length = ir_len % 2 == 0
    #even_ir_length = False

    def _reconstruct_diffuse_inner_loop(pos_eval_batch):
        gamma_eval = kernel.kernel_time_domain_diffuse(pos_eval_batch[None,:], pos_mic, wave_num, real_nyquist=even_ir_length)
        estimate = jnp.squeeze(aspmat.matmul_param(gamma_eval, krr_params[:,None,:,None]), axis=(0,1,3))
        return estimate

    estimate = jax.lax.map(_reconstruct_diffuse_inner_loop, pos_eval, batch_size=batch_size)
    return estimate


def reconstruct_from_kernel(gamma_eval, krr_params):
    """Reconstructs the sound field using an already calculated kernel matrix.
    
    gamma_eval : np.ndarray of shape (num_eval, num_mic, ir_len, ir_len)
        The kernel matrix evaluated at the evaluation points. 
    krr_params : np.ndarray of shape (num_pos, ir_len) or (num_pos * ir_len,)
        The kernel ridge regression parameters, denoted by a in the paper [brunnströmTimedomain2025]

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.

    References
    ----------
    [brunnströmTimedomain2025]
    """
    num_mic = gamma_eval.shape[1]
    
    if krr_params.ndim == 1:
        krr_params = krr_params.reshape(num_mic, -1)
    assert krr_params.ndim == 2

    estimate_each_mic = jnp.stack([gamma_eval[:,m,...] @ krr_params[m,:] for m in range(num_mic)], axis=0)
    estimate = jnp.sum(estimate_each_mic, axis=0)
    return estimate

@partial(jax.jit, static_argnames=["num_pos", "ir_len"])
def _data_weighting_argument_parsing(data_weighting, num_pos, ir_len):
    if data_weighting.ndim == 1 and data_weighting.shape[-1] == ir_len:
        data_weighting = jnp.tile(data_weighting, (num_pos))
    elif data_weighting.ndim == 2 and data_weighting.shape == (1, ir_len):
        data_weighting = jnp.tile(data_weighting, (num_pos, 1))
    data_weighting = data_weighting.reshape(-1)
    return data_weighting

@partial(jax.jit, static_argnames=["verbose"])
def krr_stationary_mics(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, verbose=False, data_weighting=None, freq_weighting=None):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Parameters
    ----------
    ir_mic : np.ndarray of shape (num_mics, ir_len)
        The impulse responses measure  at the microphones.
    pos_mic : np.ndarray of shape (num_mics, 3)
        The position of the microphones.
    pos_eval : np.ndarray of shape (num_eval, 3)
        The position of the evaluation points.
    c : float
        The speed of sound.
    reg_param : float
        The regularization parameter

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    even_dft_length = ir_len % 2 == 0
    #even_dft_length = False  # Check if the DFT length is even

    mat_size = num_pos * ir_len
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    gamma = kernel.kernel_time_domain_diffuse(pos_mic, pos_mic, wave_num, real_nyquist=even_dft_length)

    krr_params = _calc_krr_parameters(gamma, ir_mic, reg_param, data_weighting=data_weighting, freq_weighting=freq_weighting)
    estimate = reconstruct_diffuse(pos_eval, pos_mic, wave_num, krr_params)

    if verbose:
        return estimate, krr_params, gamma
    return estimate


@jax.jit
def _calc_krr_parameters(gamma, ir_mic, reg_param, data_weighting=None, freq_weighting=None):
    num_pos = ir_mic.shape[0]
    ir_len = ir_mic.shape[-1]
    mat_size = num_pos * ir_len
    gamma = aspmat.param2blockmat(gamma)

    if data_weighting is not None:
        data_weighting = _data_weighting_argument_parsing(data_weighting, num_pos, ir_len)
        data_weighting = jnp.diag(1 / data_weighting)
        reg_matrix = data_weighting * reg_param
    else:
        reg_matrix = reg_param * jnp.eye(mat_size)

    if freq_weighting is not None:
        freq_mat = jnp.squeeze(kernel.freq_to_time_domain_kernel_matrix(freq_weighting[None, None,:]), axis=(0,1))
        freq_mat_inv = jnp.squeeze(kernel.freq_to_time_domain_kernel_matrix(1/freq_weighting[None, None,:]), axis=(0,1))
        freq_mat = aspmat.block_diagonal_same(freq_mat, num_pos)
        freq_mat_inv = aspmat.block_diagonal_same(freq_mat_inv, num_pos)
        reg_matrix = reg_matrix @ freq_mat

    system_matrix_reg = gamma + reg_matrix

    data_vector = ir_mic.reshape(-1)
    krr_params = jax.scipy.linalg.solve(system_matrix_reg, data_vector, assume_a="pos")
    return krr_params

@partial(jax.jit, static_argnames=["verbose"])
def krr_stationary_mics_directional_vonmises(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, direction, beta, verbose=False, data_weighting=None, freq_weighting=None):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Parameters
    ----------
    ir_mic : np.ndarray of shape (num_mics, ir_len)
        The impulse responses measure  at the microphones.
    pos_mic : np.ndarray of shape (num_mics, 3)
        The position of the microphones.
    pos_eval : np.ndarray of shape (num_eval, 3)
        The position of the evaluation points.
    c : float
        The speed of sound.
    reg_param : float
        The regularization parameter

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    ir_len = ir_mic.shape[-1]
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)

    gamma = kernel.kernel_time_domain_directional_vonmises(pos_mic, pos_mic, wave_num, direction, beta)
    krr_params = _calc_krr_parameters(gamma, ir_mic, reg_param, data_weighting, freq_weighting)
    estimate = reconstruct_directional_vonmises(pos_eval, pos_mic, wave_num, krr_params, direction, beta)

    if verbose:
        return estimate, krr_params, gamma

    return estimate

@partial(jax.jit, static_argnames=["batch_size"])
def reconstruct_directional_vonmises(pos_eval, pos_mic, wave_num, krr_params, direction, beta, batch_size=20):
    """
    
    pos_eval : np.ndarray of shape (num_eval, 3)
        The position of the evaluation points.
    pos_mic : np.ndarray of shape (num_mics, 3)
        The position of the microphones.
    wave_num : np.ndarray of shape (num_real_freqs,)
        The wave numbers defined as 2 * np.pi * freqs / c
    krr_params : np.ndarray of shape (num_pos, ir_len) or (num_pos * ir_len,)
        The kernel ridge regression parameters, denoted by a in the paper [brunnströmTimedomain2025]'
    batch_size : int,
        The number of evaluation points to process in parallell. Default is 20.
        A higher number generally increases the speed, but also the memory usage, 
        the latter of which can be extremely high for many evaluation points. 

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.

    References
    ----------
    [brunnströmTimedomain2025]
    """
    num_mic = pos_mic.shape[0]
    
    if krr_params.ndim == 1:
        krr_params = krr_params.reshape(num_mic, -1)
    assert krr_params.ndim == 2

    def _reconstruct_inner_loop(pos_eval_batch):
        gamma_eval = kernel.kernel_time_domain_directional_vonmises(pos_eval_batch[None,:], pos_mic, wave_num, direction, beta)
        estimate = jnp.squeeze(aspmat.matmul_param(gamma_eval, krr_params[:,None,:,None]), axis=(0,1,3))
        return estimate

    estimate = jax.lax.map(_reconstruct_inner_loop, pos_eval, batch_size=batch_size)
    return estimate





@partial(jax.jit, static_argnames="score")
def cross_validation_krr_stationary_mics(data, pos, samplerate, c, reg_params, data_weighting=None, score="gcv", **kwargs):
    """Computes the GCV score for time-domain sound field kernel interpolation.

    data : np.ndarray of shape (num_data, data_dim)
        For sound field estimation, this is the sound pressure at the measurement positions.
        for time-domain kernel interpolation, data_dim is the length of the impulse response.
        for frequency-domain kernel interpolation, data_dim is 1
    pos : np.ndarray of shape (num_data, 3)
        Positions of the measurement points.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    reg_param : float
        regularization parameter to evaluate the GCV score for.

    score : str in {'gcv', 'ml', 'loocv', 'loocv_spatially_blocked'}
        decides which type of score to use to determine the best regularization parameter.
        gcv is generalized cross-validation
        ml is maximum likelihood
        loocv is leave-one-out cross-validation

    Returns
    -------
    gcv_score : float
        GCV score for the given regularization parameter.

    References
    ----------
    Time-domain sound field estimation using kernel ridge regression, J Brunnström, M. B. Møller, J Østergaard, S Koyama, T van Waterschoot, M Moonen, 2025
    """
    num_pos = pos.shape[0]
    ir_len = data.shape[-1]
    even_dft_length = ir_len % 2 == 0

    mat_size = num_pos * ir_len
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    K = kernel.kernel_time_domain_diffuse(pos, pos, wave_num, real_nyquist=even_dft_length)
    K = aspmat.param2blockmat(K)

    if data_weighting is not None:
        data_weighting = _data_weighting_argument_parsing(data_weighting, num_pos, ir_len)
        data_weighting = jnp.diag(1 / data_weighting)
        reg_matrix_base = data_weighting
    else:
        reg_matrix_base = jnp.eye(mat_size)

    def compute_score(reg):
        reg_matrix = reg * reg_matrix_base
        if score == "gcv":
            score_value = kernel.gcv_score(K, reg_matrix, data)
        elif score == "ml":
            score_value = kernel.ml_score(K, reg_matrix, data)
        elif score == "loocv":
            score_value = kernel.loocv_score(K, reg_matrix, data)
        elif score == "loocv_spatially_blocked":
            score_value = kernel.loocv_score_spatially_blocked(K, reg_matrix, data, pos, block_radius=kwargs["block_radius"])
        else:
            raise ValueError("unrecognized score parameter")
        return score_value
    
    score_values = jax.lax.map(compute_score, reg_params, batch_size=1)  

    best_index = jnp.argmin(score_values)
    best_reg_param = reg_params[best_index]

    return best_reg_param, score_values