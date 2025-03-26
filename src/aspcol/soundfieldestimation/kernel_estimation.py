
import numpy as np
import scipy.linalg as splin
import json

import aspcol.kernelinterpolation as kernel
import aspcore.fouriertransform as ft
import aspcore.matrices as aspmat

def regularize_matrix_frequency_dependent(mat, max_cond, num_blocks, time_domain=False):
    """
    
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

   # num_blocks = mat_size // dft_len
   # assert mat_size % dft_len == 0

    # if time_domain:
    #     num_tot_values = len(ft.get_real_freqs(dft_len, 1))
    # else:
    #     num_tot_values = mat_size
        
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
        kernel_func = kernel.time_domain_diffuse_kernel
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
    
        #gamma_weighted = gamma @ np.diag(data_weighting)
        #data_vector = gamma_weighted @ data_vector
        #data_matrix = gamma_weighted @ gamma 
        #reg_matrix = reg_param * gamma
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

    #data_vector = ir_mic.reshape(-1)
    data_vector = ir_mic.reshape(-1)
    krr_params = np.linalg.solve(system_matrix, data_vector)
    #if freq_weighting is not None:
    #    krr_params = (freq_mat_inv @ krr_params.reshape(num_pos, ir_len).T).T.flatten()

    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel_func, kernel_args)

    if verbose:
        metadata = aspmat.psd_matrix_metadata({"gamma":gamma, "system_matrix" : system_matrix})
        return estimate, krr_params, gamma, metadata
    return estimate



def krr_stationary_mics_regularized(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, kernel_func, kernel_args, reg_kernel_func, reg_kernel_args, verbose=False, max_cond=None, data_weighting = None):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Uses a regularization defined by a linear operator.

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
    gamma_r3 = reg_kernel_func(pos_mic, pos_mic, wave_num, *reg_kernel_args)

    data_weighting = _parse_data_weighting(data_weighting, num_pos)
    if data_weighting is not None:
        gamma_weighted = gamma * data_weighting[None,:,None,:] #matrix multiplication from the right
    else:
        gamma_weighted = gamma
    gamma_bar = aspmat.matmul_param(gamma_weighted, gamma)

    gamma = aspmat.param2blockmat(gamma)
    gamma_weighted = aspmat.param2blockmat(gamma_weighted)
    gamma_r3 = aspmat.param2blockmat(gamma_r3)
    gamma_bar = aspmat.param2blockmat(gamma_bar)


    system_matrix = gamma_bar + reg_param * gamma_r3
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
        metadata = aspmat.psd_matrix_metadata({"gamma":gamma, "gamma_r3" : gamma_r3, "gamma_bar" : gamma_bar, "system_matrix" : system_matrix})
        return estimate, krr_params, gamma, gamma_r3, metadata
    return estimate



def krr_stationary_mics_regularized_changedip(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, kernel_func, kernel_args, verbose=False, max_cond=None, data_weighting = None):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Uses a regularization defined by a linear operator.

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
    #gamma_r3 = reg_kernel_func(pos_mic, pos_mic, wave_num, *reg_kernel_args)

    data_weighting = _parse_data_weighting(data_weighting, num_pos)
    if data_weighting is not None:
        gamma_weighted = gamma * data_weighting[None,:,None,:] #matrix multiplication from the right
    else:
        gamma_weighted = gamma
    gamma_bar = aspmat.matmul_param(gamma_weighted, gamma)

    gamma = aspmat.param2blockmat(gamma)
    gamma_weighted = aspmat.param2blockmat(gamma_weighted)
    #gamma_r3 = aspmat.param2blockmat(gamma_r3)
    #gamma_bar = aspmat.param2blockmat(gamma_bar)


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
        metadata = aspmat.psd_matrix_metadata({"gamma":gamma, "gamma_r3" : gamma_r3, "gamma_bar" : gamma_bar, "system_matrix" : system_matrix})
        return estimate, krr_params, gamma, gamma_r3, metadata
    return estimate







def krr_stationary_mics_regularized_with_l2_norm(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param1, reg_param2,
                                                 kernel_func, kernel_r_func, kernel_r2_func, kernel_r3_func, 
                                                 kernel_args, kernel_r_args,
                                                    verbose=False, max_cond=1e10):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Uses a regularization defined by a linear operator, as well as the standard l2 norm. 

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
    reg_param1 : float
        The regularization parameter, scaling factor for the regularization term defined by kernel_r_func
    reg_param2 : float
        The regularization parameter, scaling factor for the l2 norm regularization term
    kernel_func : function
        The kernel function defined by the function space and the regularization specifically (R* R) Gamma(r, r') 
        The function should have the signature kernel_func(pos1, pos2, wave_num, *args) and return a kernel matrix
        which is a np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs). See documentation
        in kernel.py for more information.
    kernel_args : list
        Additional arguments to the kernel function.
    verbose : bool
        If True, returns additional metadata and intermediate results.
    
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
    gamma_r = kernel_r_func(pos_mic, pos_mic, wave_num, *kernel_r_args)
    gamma_r2 = kernel_r2_func(pos_mic, pos_mic, wave_num, *kernel_r_args)
    gamma_r3 = kernel_r3_func(pos_mic, pos_mic, wave_num, *kernel_r_args)

    G = [aspmat.param2blockmat(mat) for mat in (gamma, gamma_r, gamma_r2, gamma_r3)]
    I = np.eye(num_pos*ir_len)

    l1 = reg_param1
    l2 = reg_param2

    gamma_r_term = G[1] @ ((3 * l1 * l2**2) * I + l1**2 * G[1] + l1*l2*G[0])
    gamma_term = G[0] @ (l2**3 * I + l2**2 * G[0] + l1*l2*G[1])
    system_matrix = l1**3 * G[3] + (3 * l1**2 * l2) * G[2] + gamma_r_term + gamma_term

    # Regularize the inverse problem
    system_matrix = aspmat.regularize_matrix_with_condition_number(system_matrix, max_cond)

    data_vector = ir_mic.reshape(-1)

    data_weighting = l1 * G[1] + l2 * G[0]
    weighted_data_vector = data_weighting @ data_vector

    if max_cond is not None:
        if isinstance(max_cond, (list, tuple, np.ndarray)): #frequency dependent
            assert len(max_cond) == 2 
            system_matrix = regularize_matrix_frequency_dependent(system_matrix, max_cond, ir_len, time_domain=True)
        else: #scaled identity matrix
            system_matrix = aspmat.regularize_matrix_with_condition_number(system_matrix, max_cond)

    krr_params = np.linalg.solve(system_matrix, weighted_data_vector)
    krr_params = krr_params.reshape(num_pos, ir_len)

    if kernel_args: #if not empty
        raise NotImplementedError
    def combined_kernel(pos1, pos2, wave_num, *args):
        return l2 * kernel_func(pos1, pos2, wave_num) + l1 * kernel_r_func(pos1, pos2, wave_num, *args)
    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, combined_kernel, kernel_r_args)

    if verbose:
        metadata = aspmat.psd_matrix_metadata({"gamma":G[0], "gamma_r" : G[1], "gamma_r2" : G[2], "gamma_r3" : G[1], "system_matrix" : system_matrix})
        return estimate, krr_params, G[1], G[3], metadata
    return estimate











def krr_stationary_mics_rdft(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, freqs_to_remove=0, kernel_func=None, kernel_args=None, verbose=False, max_cond = None):
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
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)[freqs_to_remove:]
    num_freqs = wave_num.shape[0]

    if kernel_func is None:
        kernel_func = kernel.multifreq_diffuse_kernel
    if kernel_args is None:
        kernel_args = []

    gamma = kernel_func(pos_mic, pos_mic, wave_num, *kernel_args)
    #dft_weight_mat = np.ones(num_freqs) / 2
    #dft_weight_mat[-1] = 1
    #%gamma = np.diag(dft_weight_mat)[None,None,:,:] @ gamma

    gamma = aspmat.param2blockmat(gamma)

    mat_size = num_pos * num_freqs
    system_matrix = gamma + reg_param * np.eye(mat_size)
    if max_cond is not None:
        if isinstance(max_cond, (list, tuple, np.ndarray)): #frequency dependent
            assert len(max_cond) == 2 
            system_matrix = regularize_matrix_frequency_dependent(system_matrix, max_cond, num_freqs, time_domain=False)
        else: #scaled identity matrix
            system_matrix = aspmat.regularize_matrix_with_condition_number(system_matrix, max_cond)


    ir_mic_freq = ft.rfft(ir_mic, num_freqs_removed_low=freqs_to_remove)
    data_vector = ir_mic_freq.T.reshape(-1)
    krr_params = np.linalg.solve(system_matrix, data_vector)

    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel_func, kernel_args).T

    if verbose:
        metadata = aspmat.psd_matrix_metadata({"gamma": gamma, "system_matrix" : system_matrix})
        return estimate, krr_params, gamma, metadata
    return estimate







def krr_stationary_mics_regularized_rdft(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, freqs_to_remove=0, kernel_func=None, kernel_args=None, reg_kernel_func=None, reg_kernel_args=None, verbose=False, max_cond=None, data_weighting = None):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Uses a regularization defined by a linear operator.

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
    
    Returns
    -------
    ir_eval : np.ndarray of shape (num_freqs, num_eval)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)[freqs_to_remove:]
    num_freqs = wave_num.shape[0]

    if kernel_func is None:
        kernel_func = kernel.multifreq_diffuse_kernel
    if kernel_args is None:
        kernel_args = []
    if reg_kernel_func is None:
        reg_kernel_func = kernel.multifreq_diffuse_kernel
    if reg_kernel_args is None:
        reg_kernel_args = []

    gamma_r = kernel_func(pos_mic, pos_mic, wave_num, *kernel_args)
    gamma_r3 = reg_kernel_func(pos_mic, pos_mic, wave_num, *reg_kernel_args)
    gamma_r = aspmat.param2blockmat(gamma_r)
    gamma_r3 = aspmat.param2blockmat(gamma_r3)

    C = _real_dft_weighting(num_pos, num_freqs, ir_len, freqs_to_remove)

    
    data_weighting = _parse_data_weighting(data_weighting, num_pos)
    Q = _time_to_freq_data_weighting(data_weighting, num_freqs, freqs_to_remove)
    if data_weighting is not None:
        middle_weighting = C @ Q + Q @ C
        gamma_r_weighted = gamma_r @ middle_weighting
        #gamma_weighted = gamma * data_weighting[None,:,None,:]
    else:
        gamma_r_weighted = gamma_r @ C
        #gamma_bar = gamma_r @ C  + C @ gamma_r
    gamma_bar = gamma_r_weighted @ gamma_r
        

    #gamma_bar = kernel._matmul_param(gamma_r @ C[None,None,:,:], gamma_r)
    gamma_reg = 0.5 * (C @ gamma_r3 + gamma_r3 @ C)
    system_matrix = gamma_bar + reg_param * gamma_reg

    ir_mic_freq = ft.rfft(ir_mic, num_freqs_removed_low=freqs_to_remove)

    data_vector = ir_mic_freq.T.reshape(-1)
    data_vector = gamma_r_weighted @ data_vector

    if max_cond is not None:
        if isinstance(max_cond, (list, tuple, np.ndarray)): #frequency dependent
            assert len(max_cond) == 2 
            system_matrix = regularize_matrix_frequency_dependent(system_matrix, max_cond, num_freqs, time_domain=False)
        else: #scaled identity matrix
            system_matrix = aspmat.regularize_matrix_with_condition_number(system_matrix, max_cond)

    krr_params = np.linalg.solve(system_matrix, data_vector)
    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel_func, kernel_args).T

    if verbose:
        metadata = aspmat.psd_matrix_metadata({"gamma_r" : gamma_r, "gamma_r3" : gamma_r3, "gamma_bar" : gamma_bar, "system_matrix" : system_matrix})
        return estimate, krr_params, gamma_r, gamma_r3, metadata
    return estimate

def _real_dft_weighting(num_pos, num_freqs, ir_len, freqs_to_remove):
    """Defined as C_M in the notes
    """
    c_diag = np.ones(num_freqs) * 2 / ir_len
    c_diag[-1] = 1 / ir_len
    if freqs_to_remove == 0:
        c_diag[0] = 1 / ir_len
    c_diag = np.tile(c_diag, num_pos)
    C = np.diag(c_diag)
    return C

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

def _time_to_freq_data_weighting(data_weighting, num_freqs, freqs_to_remove):
    """
    
    Parameters
    ----------
    data_weighting : np.ndarray of shape (num_pos, ir_len)
        weighting for each microphone and time sample. The weighting is assumed to be in the time domain
        Run _parse_data_weighting to get the correct format
    
    Returns
    -------
    Q : np.ndarray of shape (num_channels, num_freqs, num_freqs)
        The weighting matrix in the frequency domain
        Should be considered as a block matrix with num_channels blocks
    """
    pass






# def krr_stationary_mics_regularized_restricted(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, kernel_func, kernel_args, reg_kernel_func, reg_kernel_args, verbose=False, max_cond=1e10):
#     """Estimates the impulse responses at the evaluation points using kernel ridge regression.

#     Uses a regularization defined by a linear operator.

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
#         The regularization parameter
#     kernel_func : function
#         The kernel function defined by the function space and the regularization specifically (R* R) Gamma(r, r') 
#         The function should have the signature kernel_func(pos1, pos2, wave_num, *args) and return a kernel matrix
#         which is a np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs). See documentation
#         in kernel.py for more information.
#     kernel_args : list
#         Additional arguments to the kernel function.
#     reg_kernel_func : function
#         The kernel function defined by the function space and the regularization, specifically (R* R)^3 Gamma(r, r'). Format
#         is the same as kernel_func.
#     reg_kernel_args : list
#         Additional arguments to the reg_kernel_func.
#     verbose : bool
#         If True, returns additional metadata and intermediate results.
    
#     Returns
#     -------
#     ir_eval : np.ndarray of shape (num_eval, ir_len)
#         The estimated impulse responses at the evaluation points.
#     """
#     num_pos = pos_mic.shape[0]
#     num_eval = pos_eval.shape[0]
#     ir_len = ir_mic.shape[-1]
#     wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
#     gamma = kernel_func(pos_mic, pos_mic, wave_num, *kernel_args)
#     gamma_r3 = reg_kernel_func(pos_mic, pos_mic, wave_num, *reg_kernel_args)

#     #gamma_bar = _calc_gamma_bar(gamma)
#     gamma_bar = kernel._matmul_param(gamma, gamma)
#     system_matrix_reg = gamma_bar + reg_param * gamma_r3

#     num_remove = 10
#     reduced_ir_len = ir_len - num_remove
#     F = splin.dft(ir_len)
#     F_red = F[num_remove:,:]
#     F_inv = F.conj().T
#     F_inv_red = F_inv[:,num_remove:]

#     system_matrix_reg = F_red[None,None,:,:] @ system_matrix_reg @ F_inv_red[None,None,:,:]
#     gamma_red = F_red[None,None,:,:] @ gamma

#     gamma = aspmat._param2blockmat(gamma)
#     gamma_red = aspmat._param2blockmat(gamma_red)
#     gamma_r3 = aspmat._param2blockmat(gamma_r3)
#     gamma_bar = aspmat._param2blockmat(gamma_bar)
#     system_matrix_reg = aspmat._param2blockmat(system_matrix_reg)


#     data_vector = ir_mic.reshape(-1)
#     weighted_data_vector = gamma_red @ data_vector
#     krr_params = np.linalg.solve(system_matrix_reg, weighted_data_vector)
#     #krr_params = np.linalg.lstsq(system_matrix_reg, weighted_data_vector, rcond=1e-10)[0]

    
#     krr_params = krr_params.reshape(num_pos, reduced_ir_len)
#     krr_params = np.squeeze(F_inv_red @ krr_params[:,:,None])
#     print(np.abs(np.imag(krr_params)).max())
#     krr_params = np.real(krr_params)

#     estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel_func, kernel_args)

#     if verbose:
#         metadata = {}

#         for mat_name, mat in {"gamma":gamma, "gamma_r3" : gamma_r3, "gamma_bar" : gamma_bar, "system_matrix_reg" : system_matrix_reg}.items():
#             metadata[f"{mat_name} max eigenvalue"] = splin.eigvalsh(mat, subset_by_index = (mat.shape[-1]-2, mat.shape[-1]-1)).tolist()
#             metadata[f"{mat_name} min eigenvalue"] = splin.eigvalsh(mat, subset_by_index = (0, 1)).tolist()
#             metadata[f"{mat_name} condition"] = np.linalg.cond(mat).tolist()
#         return estimate, krr_params, gamma, gamma_r3, metadata
#     return estimate







def krr_stationary_mics_regularized_simplified(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, kernel_func, kernel_args, reg_kernel_func, reg_kernel_args, verbose=False, max_cond = 1e10):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Uses a regularization defined by a linear operator.

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
    gamma_r3 = reg_kernel_func(pos_mic, pos_mic, wave_num, *reg_kernel_args)


    gamma = aspmat.param2blockmat(gamma)
    gamma_r3 = aspmat.param2blockmat(gamma_r3)
    #gamma_bar = _param2blockmat(gamma_bar)

    
    gamma_to_inv = aspmat.regularize_matrix_with_condition_number(gamma, max_cond)

    reg_matrix = np.linalg.solve(gamma_to_inv, gamma_r3)
    system_matrix = gamma + reg_param * reg_matrix #+ 1e-5 * np.eye(gamma_r3.shape[-1])
    system_matrix = aspmat.regularize_matrix_with_condition_number(system_matrix, max_cond)

    data_vector = ir_mic.reshape(-1)
    krr_params = np.linalg.solve(system_matrix, data_vector)
    krr_params = krr_params.reshape(num_pos, ir_len)

    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel_func, kernel_args)

    if verbose:
        metadata = aspmat.psd_matrix_metadata({"gamma": gamma, "gamma_r3" : gamma_r3, "system_matrix" : system_matrix})
        return estimate, krr_params, gamma, gamma_r3, metadata
    return estimate


def krr_stationary_mics_direction_regularized(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, direction, beta):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Adds a directional weighting to regularize the sound field. This is done by using a linear regularization operator. 

    Assumes a diagonal directional weighting, formulated such that \\lvert w(d) \\rvert is a von Mises-Fisher distribution, 
    which gives a closed-form solution for the kernel function. 

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
    direction : np.ndarray of shape (1, 3)
        a unit vector describing the direction of the weighting. The direction should be from (0,0,0) towards
        the source. 
    beta : float
        The strength of the weighting. A larger value will give more regularization.

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    gamma = kernel.time_domain_directional_kernel_vonmises(pos_mic, pos_mic, wave_num, direction, beta)
    gamma_r3 = kernel.time_domain_directional_kernel_vonmises(pos_mic, pos_mic, wave_num, direction, 3*beta)

    gamma_bar = aspmat.matmul_param(gamma, gamma)

    gamma = aspmat.param2blockmat(gamma)
    gamma_r3 = aspmat.param2blockmat(gamma_r3)
    gamma_bar = aspmat.param2blockmat(gamma_bar)

    #system_matrix = np.linalg.solve(gamma_stacked, gamma_bar_stacked)
    system_matrix = gamma_bar + reg_param * gamma_r3 #+ 1e-6 * np.eye(mat_size)

    data_vector = ir_mic.reshape(-1)
    weighted_data_vector = gamma @ data_vector
    krr_params = np.linalg.solve(system_matrix, weighted_data_vector)

    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel.time_domain_directional_kernel_vonmises, [direction, beta])

    # out_gamma = kernel.time_domain_directional_kernel_vonmises(pos_eval, pos_mic, wave_num, direction, beta)
    # estimate = np.zeros((num_eval, ir_len))
    # for m in range(num_pos):
    #     estimate += out_gamma[:,m,...] @ krr_params[m*ir_len:(m+1)*ir_len]
    return estimate

def krr_stationary_mics_direction_regularized_approx(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, direction, beta):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Adds a directional weighting to regularize the sound field. This is done by using a linear regularization operator. 

    Assumes a diagonal directional weighting, formulated such that \\lvert w(d) \\rvert is a von Mises-Fisher distribution, 
    which gives a closed-form solution for the kernel function. 

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
    direction : np.ndarray of shape (1, 3)
        a unit vector describing the direction of the weighting. The direction should be from (0,0,0) towards
        the source. 
    beta : float
        The strength of the weighting. A larger value will give more regularization.

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    gamma = kernel.time_domain_directional_kernel_vonmises_approx(pos_mic, pos_mic, wave_num, direction, beta)
    gamma_r3 = kernel.time_domain_directional_kernel_vonmises_approx(pos_mic, pos_mic, wave_num, direction, 3*beta)

    gamma_bar = aspmat.matmul_param(gamma, gamma)

    gamma = aspmat.param2blockmat(gamma)
    gamma_r3 = aspmat.param2blockmat(gamma_r3)
    gamma_bar = aspmat.param2blockmat(gamma_bar)

    #system_matrix = np.linalg.solve(gamma_stacked, gamma_bar_stacked)
    system_matrix = gamma_bar + reg_param * gamma_r3 #+ 1e-6 * np.eye(mat_size)

    data_vector = ir_mic.reshape(-1)
    weighted_data_vector = gamma @ data_vector
    krr_params = np.linalg.solve(system_matrix, weighted_data_vector)

    # out_gamma = kernel.time_domain_directional_kernel_vonmises_approx(pos_eval, pos_mic, wave_num, direction, beta)
    # estimate = np.zeros((num_eval, ir_len))
    # for m in range(num_pos):
    #     estimate += out_gamma[:,m,...] @ krr_params[m*ir_len:(m+1)*ir_len]

    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel.time_domain_directional_kernel_vonmises_approx, [direction, beta])
    return estimate


def krr_stationary_mics_direction_regularized_changedip(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, direction, beta):
    """Estimates the impulse responses at the evaluation points using kernel ridge regression.

    Adds a directional weighting to regularize the sound field. This is done by just changing out the inner product
    in the RKHS to a weighted inner product, the same as was done in [koyamaSpatial2021]. The result is that the solution
    has the same form as the unregularized solution, but with a different kernel function.

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
        The regularization parameter. Scales the regularization term in the optimization.
    direction : np.ndarray of shape (1, 3)
        a unit vector describing the direction of the weighting. The direction should be from (0,0,0) towards
        the source. 
    beta : float
        The strength of the weighting. A larger value will give more regularization.

    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    gamma = kernel.time_domain_directional_kernel_vonmises(pos_mic, pos_mic, wave_num, direction, beta)

    #gamma_bar = aspmat.matmul_param(gamma, gamma)
    #gamma_bar = aspmat.param2blockmat(gamma_bar)
    gamma = aspmat.param2blockmat(gamma)

    #system_matrix = np.linalg.solve(gamma, gamma_bar)
    system_matrix = gamma + reg_param * np.eye(gamma.shape[-1])

    data_vector = ir_mic.reshape(-1)
    krr_params = np.linalg.solve(system_matrix, data_vector)

    estimate = reconstruct(pos_eval, pos_mic, wave_num, krr_params, kernel.time_domain_directional_kernel_vonmises, [direction, beta])
    return estimate








def _calc_cost_regularization_term(data_vector, krr_params, kernel_reconstruct, kernel_reg):
    """

    Parameters
    ----------
    data_vector : np.ndarray of shape (num_mics, ir_len)
        The impulse responses measured at the microphones.
    estimates : np.ndarray of shape (num_mics, ir_len)
        The estimated impulse responses at the microphones.
    krr_params : np.ndarray of shape (num_mics, ir_len)
        The kernel ridge regression parameters.
    kernel_reconstruct : np.ndarray of shape (num_mics, num_mics, ir_len, ir_len)
        The kernel used for reconsturction. Should be Gamma if the krr_params were computed using diffuse kernel, but Gamma_r if the krr_params
        were computed using a regularized kernel.
    kernel_reg : np.ndarray of shape (num_mics, num_mics, ir_len, ir_len)
        The kernel used for regularization. Should be Gamma_r if the krr_params, were computed using the diffuse kernel, and Gamma_r^3 if the 
        krr_params were computed using a regularized kernel.
    reg_param : float
        The regularization parameter.

    Kernel_mat should be Gamma_r if the krr_params were computed using diffuse kernel, but Gamma_r^3 if the krr_params
    were computed using a regularized kernel.
    """
    krr_params = krr_params.reshape(-1)
    data_vector = data_vector.reshape(-1)

    reg_term = np.squeeze(krr_params[None,:] @ kernel_reg @ krr_params[:, None])
    return reg_term

def _calc_cost_data_term(data_vector, krr_params, kernel_reconstruct, kernel_reg):
    """

    Parameters
    ----------
    data_vector : np.ndarray of shape (num_mics, ir_len)
        The impulse responses measured at the microphones.
    estimates : np.ndarray of shape (num_mics, ir_len)
        The estimated impulse responses at the microphones.
    krr_params : np.ndarray of shape (num_mics, ir_len)
        The kernel ridge regression parameters.
    kernel_reconstruct : np.ndarray of shape (num_mics, num_mics, ir_len, ir_len)
        The kernel used for reconsturction. Should be Gamma if the krr_params were computed using diffuse kernel, but Gamma_r if the krr_params
        were computed using a regularized kernel.
    kernel_reg : np.ndarray of shape (num_mics, num_mics, ir_len, ir_len)
        The kernel used for regularization. Should be Gamma_r if the krr_params, were computed using the diffuse kernel, and Gamma_r^3 if the 
        krr_params were computed using a regularized kernel.
    reg_param : float
        The regularization parameter.

    Kernel_mat should be Gamma_r if the krr_params were computed using diffuse kernel, but Gamma_r^3 if the krr_params
    were computed using a regularized kernel.
    """
    krr_params = krr_params.reshape(-1)
    data_vector = data_vector.reshape(-1)

    data_term = np.linalg.norm(data_vector[:,None] - kernel_reconstruct @ krr_params[:,None])**2

    # data_term = 0
    # for m in range(data_vector.shape[0]):
    #     est = np.zeros_like(data_vector[m,:])
    #     for m2 in range(data_vector.shape[0]):
    #         est += kernel_reconstruct[m,m2,...] @ krr_params[m2,:,None]
    #     data_term += np.linalg.norm(data_vector[m,:] - est)**2

    # reg_term = 0
    # for m in range(data_vector.shape[0]):
    #     for m2 in range(data_vector.shape[0]):
    #         reg_term += np.squeeze(krr_params[m,None,:] @ kernel_reg[m,m2,...] @ krr_params[m2,:,None])
    # tot = data_term + reg_param * reg_term
    return data_term
