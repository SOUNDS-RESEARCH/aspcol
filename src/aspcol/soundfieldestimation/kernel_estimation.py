
import numpy as np
import scipy.linalg as splin
import json

import aspcol.kernelinterpolation as kernel
import aspcore.fouriertransform as ft
import aspcore.matrices as aspmat


def reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel_func, kernel_args):
    """Reconstruct the sound field function from estimated kernel ridge regression parameters.
    
    pos_eval : np.ndarray of shape (num_eval, 3)
        The position of the evaluation points.
    pos_mic : np.ndarray of shape (num_mics, 3)
        The position of the microphones.
    wave_num : np.ndarray of shape (num_real_freqs,)
        The wave numbers defined as 2 * np.pi * freqs / c
    krr_params : np.ndarray of shape (num_pos, ir_len) or (num_pos * ir_len,)
        The kernel ridge regression parameters, denoted by a in the paper [brunnströmTimedomain2025]
    kernel_func : function
        The kernel function defined by the function space and the regularization specifically (R* R) Gamma(r, r')
        The function should have the signature kernel_func(pos1, pos2, wave_num, *args) and return a kernel matrix
        which is a np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs). See documentation
        in kernel.py for more information.
    kernel_args : list
        Additional arguments to the kernel function.

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.

    References
    ----------
    [brunnströmTimedomain2025]
    """
    num_eval = pos_eval.shape[0]
    num_mic = pos_mic.shape[0]
    
    if krr_params.ndim == 1:
        krr_params = krr_params.reshape(num_mic, -1)
    assert krr_params.ndim == 2
    ir_len = krr_params.shape[-1]

    out_gamma = kernel_func(pos_eval, pos_mic, wave_num, *kernel_args)
    estimate = np.zeros((num_eval, ir_len), dtype=krr_params.dtype)
    for m in range(num_mic):
        estimate += out_gamma[:,m,...] @ krr_params[m,:]
    return estimate

def krr_stationary_mics(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, kernel_func=None, kernel_args=None, verbose=False, max_cond = None, data_weighting = None, freq_weighting = None):
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

    data_weighting : np.ndarray of shape (ir_len,) or (num_mics, ir_len)
        The data term in the optimization problem will be the l2 norm weighted by this vector (the matrix constructed from
        this vector on the diagonal). All values should be positive
    freq_weighting : np.ndarray of shape (num_real_freqs,)
        The frequency weighting of the regularization term. If None, no frequency weighting is applied.

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    mat_size = num_pos * ir_len
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)

    if kernel_func is None:
        kernel_func = kernel.kernel_time_domain_diffuse
    if kernel_args is None:
        kernel_args = []

    gamma = kernel_func(pos_mic, pos_mic, wave_num, *kernel_args)
    gamma = aspmat.param2blockmat(gamma)

    data_weighting = _parse_data_weighting(data_weighting, num_pos)
    if data_weighting is None:
        reg_matrix = reg_param * np.eye(mat_size)
    else:
        data_weighting = data_weighting.reshape(-1)
        reg_matrix = reg_param * np.diag(1 / data_weighting)

    if freq_weighting is not None:
        freq_mat = np.squeeze(kernel.freq_to_time_domain_kernel_matrix(freq_weighting[None, None,:]), axis=(0,1))
        freq_mat_inv = np.squeeze(kernel.freq_to_time_domain_kernel_matrix(1 / freq_weighting[None, None,:]), axis=(0,1))
        reg_matrix = aspmat.block_diag_multiply(reg_matrix, block_right=freq_mat)

    system_matrix = gamma + reg_matrix

    if max_cond is not None:
        if isinstance(max_cond, (list, tuple, np.ndarray)): #frequency dependent
            assert len(max_cond) == 2 
            system_matrix = regularize_matrix_frequency_dependent(system_matrix, max_cond, ir_len, time_domain=True)
        else: #scaled identity matrix
            system_matrix = aspmat.regularize_matrix_with_condition_number(system_matrix, max_cond)
    data_vector = ir_mic.reshape(-1)
    krr_params = np.linalg.solve(system_matrix, data_vector)

    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel_func, kernel_args)

    if verbose:
        metadata = aspmat.psd_matrix_metadata({"gamma":gamma, "system_matrix" : system_matrix})
        return estimate, krr_params, gamma, metadata
    return estimate



def krr_stationary_mics_regularized(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, kernel_func, kernel_args, verbose=False, max_cond=None, data_weighting = None):
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
    kernel_func : function
        The kernel function defined by the function space and the regularization specifically (R* R) Gamma(r, r') 
        The function should have the signature kernel_func(pos1, pos2, wave_num, *args) and return a kernel matrix
        which is a np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs). See documentation
        in kernel.py for more information.
    kernel_args : list
        Additional arguments to the kernel function.
    reg_kernel_func : function
        The kernel function defined by the function space and the regularization, specifically (R* R)^3 Gamma(r, r'). Format
        is the same as kernel_func.
    reg_kernel_args : list
        Additional arguments to the reg_kernel_func.
    verbose : bool
        If True, returns additional metadata and intermediate results.
    data_weighting : np.ndarray of shape (ir_len,) or (num_mics, ir_len)
        The data term in the optimization problem will be the l2 norm weighted by this vector (the matrix constructed from
        this vector on the diagonal). All values should be positive.
    freq_weighting : np.ndarray of shape (num_real_freqs,)
        The frequency weighting of the regularization term. If None, no frequency weighting is applied.
    
    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    gamma = kernel_func(pos_mic, pos_mic, wave_num, *kernel_args)

    data_weighting = _parse_data_weighting(data_weighting, num_pos)
    if data_weighting is not None:
        gamma_weighted = gamma * data_weighting[None,:,None,:] #matrix multiplication from the right
    else:
        gamma_weighted = gamma
    gamma_bar = aspmat.matmul_param(gamma_weighted, gamma)

    gamma = aspmat.param2blockmat(gamma)
    gamma_weighted = aspmat.param2blockmat(gamma_weighted)

    system_matrix = gamma + reg_param * np.eye
    if max_cond is not None:
        if isinstance(max_cond, (list, tuple, np.ndarray)): #frequency dependent
            assert len(max_cond) == 2 
            system_matrix = regularize_matrix_frequency_dependent(system_matrix, max_cond, ir_len, time_domain=True)
        else: #scaled identity matrix
            system_matrix = aspmat.regularize_matrix_with_condition_number(system_matrix, max_cond)


    data_vector = ir_mic.reshape(-1)
    weighted_data_vector = gamma_weighted @ data_vector
    krr_params = np.linalg.solve(system_matrix, weighted_data_vector)
    #krr_params = np.linalg.lstsq(system_matrix_reg, weighted_data_vector, rcond=1e-10)[0]
    krr_params = krr_params.reshape(num_pos, ir_len)

    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel_func, kernel_args)

    if verbose:
        metadata = aspmat.psd_matrix_metadata({"gamma":gamma, "gamma_bar" : gamma_bar, "system_matrix" : system_matrix})
        return estimate, krr_params, gamma, metadata
    return estimate



# def krr_stationary_mics_direction_regularized_changedip(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, direction, beta):
#     """Estimates the impulse responses at the evaluation points using kernel ridge regression.

#     Adds a directional weighting to regularize the sound field. This is done by just changing out the inner product
#     in the RKHS to a weighted inner product, the same as was done in [koyamaSpatial2021]. The result is that the solution
#     has the same form as the unregularized solution, but with a different kernel function.

#     Parameters
#     ----------
#     ir_mic : np.ndarray of shape (num_mics, ir_len)
#         The impulse responses measure  at the microphones.
#     pos_mic : np.ndarray of shape (num_mics, 3)
#         The position of the microphones.
#     pos_eval : np.ndarray of shape (num_eval, 3)
#         The position of the evaluation points.
#     c : float
#         The speed of sound.
#     reg_param : float
#         The regularization parameter. Scales the regularization term in the optimization.
#     direction : np.ndarray of shape (1, 3)
#         a unit vector describing the direction of the weighting. The direction should be from (0,0,0) towards
#         the source. 
#     beta : float
#         The strength of the weighting. A larger value will give more regularization.

#     Returns
#     -------
#     ir_eval : np.ndarray of shape (num_eval, ir_len)
#         The estimated impulse responses at the evaluation points.
#     """
#     num_pos = pos_mic.shape[0]
#     num_eval = pos_eval.shape[0]
#     ir_len = ir_mic.shape[-1]
#     wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
#     gamma = kernel.time_domain_directional_kernel_vonmises(pos_mic, pos_mic, wave_num, direction, beta)

#     gamma = aspmat.param2blockmat(gamma)
#     system_matrix = gamma + reg_param * np.eye(gamma.shape[-1])

#     data_vector = ir_mic.reshape(-1)
#     krr_params = np.linalg.solve(system_matrix, data_vector)

#     estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel.time_domain_directional_kernel_vonmises, [direction, beta])
#     return estimate



def _parse_data_weighting(data_weighting, num_pos):
    """
    Parameters
    ----------
    data_weighting : np.ndarray of shape (ir_len,) or (1, ir_len) or (num_pos, ir_len)
        The data term in the optimization problem will be the l2 norm weighted by this vector (the matrix constructed from
        this vector on the diagonal). All values should be positive
    num_pos : int
        The number of microphones

    Returns
    -------
    data_weighting : np.ndarray of shape (num_pos, ir_len)
        weighting for each microphone and time sample
    """
    if data_weighting is None:
        return None
    
    assert data_weighting.ndim == 1 or data_weighting.ndim == 2
    assert np.all(data_weighting >= 0)
    
    if data_weighting.ndim == 1:
        data_weighting = data_weighting[None,:]
    if data_weighting.shape[0] == 1:
        data_weighting = np.tile(data_weighting, (num_pos, 1))
    return data_weighting


def regularize_matrix_frequency_dependent(mat, max_cond, num_blocks, time_domain=False):
    """Adds a regularization matrix to the input matrix in order to limit the condition number
    
    Parameters
    ----------
    max : ndarray of shape (mat_size, mat_size)
        the matrix to regularize. Is assumed to be Hermitian
    max_cond : tuple, list or ndarray of length 2
        specifies the max condition number at the lowest frequency, and the max condition number
        that will be used for higher frequencies
    time_domain : bool, optional
        default is False. If True, the matrix will be multiplied by the DFT matrix from either side in 
        order to produce a circulant matrix to regularize with. That will correspond to the frequency domain
        frequency-dependent regularization. 

    Returns
    -------
    regularized_matrix : ndarray of shape (mat_size, mat_size)
    """

    mat_size = mat.shape[-1]
    assert mat_size % num_blocks == 0
    values_per_block = mat_size // num_blocks
        
    fade_len = values_per_block // 5

    fade_values = np.logspace(np.log10(max_cond[0]), np.log10(max_cond[1]), fade_len)
    all_values = np.concatenate((fade_values, max_cond[1] * np.ones(values_per_block-fade_len)))
    all_values = np.tile(all_values, num_blocks)

    max_ev = splin.eigvalsh(mat, subset_by_index=(mat_size-1, mat_size-1))
    reg_matrix = np.diag(max_ev / all_values)

    if time_domain:
        F = ft.dft_mat(values_per_block)
        reg_matrix = aspmat.block_diag_multiply(reg_matrix, F.conj().T, F)
       # reg_matrix = F @ (reg_matrix) @ F.conj().T / 

    mat_reg = mat + reg_matrix
    return mat_reg


