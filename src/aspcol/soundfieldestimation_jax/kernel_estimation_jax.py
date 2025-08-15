
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
        gamma_eval = kernel.time_domain_diffuse_kernel(pos_eval_batch[None,:], pos_mic, wave_num, real_nyquist=even_ir_length)
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


def _blockmat2param(R, num_mic, ir_len):
    """
    takes blockmat of form (num_mic * ir_len, num_mic * ir_len) and output parametrized
    matrices of form (num_mic, num_mic, ir_len, ir_len)
    
    """
    new_mat = np.zeros((num_mic, num_mic, ir_len, ir_len), dtype=R.dtype)
    for i in range(num_mic):
        for j in range(num_mic):
            new_mat[i,j,:,:] = R[i*ir_len:(i+1)*ir_len, j*ir_len:(j+1)*ir_len]
    return new_mat



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
    gamma = kernel.time_domain_diffuse_kernel(pos_mic, pos_mic, wave_num, real_nyquist=even_dft_length)
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
    krr_params = jnp.linalg.solve(system_matrix_reg, data_vector)

    estimate = reconstruct_diffuse(pos_eval, pos_mic, wave_num, krr_params)

    if verbose:
        return estimate, krr_params, gamma
    return estimate


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

    gamma = kernel.time_domain_directional_kernel_vonmises(pos_mic, pos_mic, wave_num, direction, beta)
    krr_params = _calc_krr_parameters(gamma, ir_mic, reg_param, data_weighting, freq_weighting)
    estimate = reconstruct_directional_vonmises(pos_eval, pos_mic, wave_num, krr_params, direction, beta)

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
    krr_params = jnp.linalg.solve(system_matrix_reg, data_vector)
    return krr_params



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
        gamma_eval = kernel.time_domain_directional_kernel_vonmises(pos_eval_batch[None,:], pos_mic, wave_num, direction, beta)
        estimate = jnp.squeeze(aspmat.matmul_param(gamma_eval, krr_params[:,None,:,None]), axis=(0,1,3))
        return estimate

    estimate = jax.lax.map(_reconstruct_inner_loop, pos_eval, batch_size=batch_size)
    return estimate




@partial(jax.jit, static_argnames=["verbose", "max_cond"])
def krr_stationary_mics_envelope_regularized(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, envelope_reg, reg_points, verbose=False, data_weighting=None, max_cond=1e10):
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
    
    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    gamma = kernel.time_domain_envelope_kernel(pos_mic, pos_mic, wave_num, envelope_reg, reg_points)
    gamma_r3 = kernel.time_domain_envelope_kernel_r3(pos_mic, pos_mic, wave_num, envelope_reg, reg_points)
        
    if data_weighting is not None:
        if data_weighting.ndim == 1:
            data_weighting = data_weighting[None,:]
        gamma_weighted = gamma * data_weighting[None,:,None,:] #matrix multiplication from the right
    else:
        gamma_weighted = gamma
    gamma_bar = aspmat.matmul_param(gamma_weighted, gamma)

    gamma = aspmat.param2blockmat(gamma)
    gamma_weighted = aspmat.param2blockmat(gamma_weighted)
    gamma_r3 = aspmat.param2blockmat(gamma_r3)
    gamma_bar = aspmat.param2blockmat(gamma_bar)

    
        #data_weighting = jnp.diag(1 / data_weighting)
        #reg_matrix = data_weighting * reg_param
    #else:
        #reg_matrix = reg_param * jnp.eye(mat_size)

    
    system_matrix_reg_pre_reg = gamma_bar + reg_param * gamma_r3 #+ 1e-5 * np.eye(gamma_r3.shape[-1])

    # Regularize the inverse problem
    system_matrix_reg = aspmat.regularize_matrix_with_condition_number(system_matrix_reg_pre_reg, max_cond=max_cond)
    #max_ev = jnp.linalg.eigvalsh(system_matrix_reg_pre_reg)[-1]#, subset_by_index=(system_matrix_reg.shape[-1]-1, system_matrix_reg.shape[-1]-1))
    #identity_scaling = max_ev / max_cond
    #system_matrix_reg = system_matrix_reg_pre_reg + identity_scaling * jnp.eye(system_matrix_reg_pre_reg.shape[-1])


    data_vector = ir_mic.reshape(-1)
    weighted_data_vector = gamma_weighted @ data_vector
    krr_params = jnp.linalg.solve(system_matrix_reg, weighted_data_vector)
    krr_params = krr_params.reshape(num_pos, ir_len)

    gamma_eval = kernel.time_domain_envelope_kernel(pos_eval, pos_mic, wave_num, envelope_reg, reg_points)
    estimate = reconstruct_from_kernel(gamma_eval, krr_params)

    if verbose:
        # metadata = {}
        # for mat_name, mat in {"gamma":gamma, 
        #                     "gamma_r3" : gamma_r3, 
        #                     "gamma_bar" : gamma_bar, 
        #                     "system_matrix_pre_reg" : system_matrix_reg_pre_reg,
        #                     "system_matrix_reg" : system_matrix_reg}.items():
        #     metadata[f"{mat_name} max eigenvalue"] = jnp.linalg.eigvalsh(mat)[-1]
        #     metadata[f"{mat_name} min eigenvalue"] = jnp.linalg.eigvalsh(mat)[0]
        #     metadata[f"{mat_name} condition"] = jnp.linalg.cond(mat)
        return estimate, krr_params, gamma, gamma_r3#, metadata
    return estimate


@partial(jax.jit, static_argnames=["verbose", "max_cond"])
def krr_stationary_mics_envelope_regularized_changedip(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, envelope_reg, reg_points, verbose=False, data_weighting=None, max_cond=None):
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
    gamma = kernel.time_domain_envelope_kernel(pos_mic, pos_mic, wave_num, envelope_reg, reg_points)
    gamma = aspmat.param2blockmat(gamma)

    if data_weighting is not None:
        data_weighting = _data_weighting_argument_parsing(data_weighting, num_pos, ir_len)
        data_weighting = jnp.diag(1 / data_weighting)
        reg_matrix = data_weighting * reg_param
    else:
        #data_weighting = _data_weighting_argument_parsing(envelope_reg, num_pos, ir_len)
        #data_weighting = jnp.diag(1 / data_weighting)
        #reg_matrix = data_weighting * reg_param
        reg_matrix = reg_param * jnp.eye(mat_size)
        
    #gamma = aspmat.param2blockmat(gamma)
    #gamma_weighted = aspmat.param2blockmat(gamma_weighted)

    system_matrix_reg = gamma + reg_matrix

    # Regularize the inverse problem
    if max_cond is not None:
        system_matrix_reg = aspmat.regularize_matrix_with_condition_number(system_matrix_reg, max_cond=max_cond)
    #max_ev = jnp.linalg.eigvalsh(system_matrix_reg_pre_reg)[-1]#, subset_by_index=(system_matrix_reg.shape[-1]-1, system_matrix_reg.shape[-1]-1))
    #identity_scaling = max_ev / max_cond
    #system_matrix_reg = system_matrix_reg_pre_reg + identity_scaling * jnp.eye(system_matrix_reg_pre_reg.shape[-1])

    data_vector = ir_mic.reshape(-1)
    krr_params = jnp.linalg.solve(system_matrix_reg, data_vector)
    krr_params = krr_params.reshape(num_pos, ir_len)

    # data_vector = ir_mic.reshape(-1)
    # weighted_data_vector = gamma_weighted @ data_vector
    # krr_params = jnp.linalg.solve(system_matrix_reg, weighted_data_vector)
    # krr_params = krr_params.reshape(num_pos, ir_len)

    gamma_eval = kernel.time_domain_envelope_kernel(pos_eval, pos_mic, wave_num, envelope_reg, reg_points)
    estimate = reconstruct_from_kernel(gamma_eval, krr_params)

    if verbose:
        # metadata = {}
        # for mat_name, mat in {"gamma":gamma, 
        #                     "gamma_r3" : gamma_r3, 
        #                     "gamma_bar" : gamma_bar, 
        #                     "system_matrix_pre_reg" : system_matrix_reg_pre_reg,
        #                     "system_matrix_reg" : system_matrix_reg}.items():
        #     metadata[f"{mat_name} max eigenvalue"] = jnp.linalg.eigvalsh(mat)[-1]
        #     metadata[f"{mat_name} min eigenvalue"] = jnp.linalg.eigvalsh(mat)[0]
        #     metadata[f"{mat_name} condition"] = jnp.linalg.cond(mat)
        return estimate, krr_params, gamma#, metadata
    return estimate

@partial(jax.jit, static_argnames=["verbose", "max_cond"])
def krr_stationary_mics_frequency_weighted(ir_mic, pos_mic, pos_eval, samplerate, c, reg_param, weighting_mat, verbose=False, max_cond=1e10):
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
    
    Returns
    -------
    ir_eval : np.ndarray of shape (num_eval, ir_len)
        The estimated impulse responses at the evaluation points.
    """
    num_pos = pos_mic.shape[0]
    num_eval = pos_eval.shape[0]
    ir_len = ir_mic.shape[-1]
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    gamma = kernel.time_domain_frequency_weighted_kernel(pos_mic, pos_mic, wave_num, weighting_mat)
    weighting_mat_r3 = weighting_mat @ weighting_mat @ weighting_mat
    gamma_r3 = kernel.time_domain_frequency_weighted_kernel(pos_mic, pos_mic, wave_num, weighting_mat_r3)

    gamma_bar = aspmat.matmul_param(gamma, gamma)

    gamma = aspmat.param2blockmat(gamma)
    gamma_r3 = aspmat.param2blockmat(gamma_r3)
    gamma_bar = aspmat.param2blockmat(gamma_bar)

    system_matrix_reg = gamma_bar + reg_param * gamma_r3

    # Regularize the inverse problem
    system_matrix_reg = aspmat.regularize_matrix_with_condition_number(system_matrix_reg, max_cond=max_cond)
    # max_ev = jnp.linalg.eigvalsh(system_matrix_reg)[-1]#, subset_by_index=(system_matrix_reg.shape[-1]-1, system_matrix_reg.shape[-1]-1))
    # desired_inv_condition_number = 1e10
    # identity_scaling = max_ev / desired_inv_condition_number
    # system_matrix_reg += identity_scaling * jnp.eye(system_matrix_reg.shape[-1])


    data_vector = ir_mic.reshape(-1)
    weighted_data_vector = gamma @ data_vector
    krr_params = jnp.linalg.solve(system_matrix_reg, weighted_data_vector)
    krr_params = krr_params.reshape(num_pos, ir_len)

    gamma_eval = kernel.time_domain_frequency_weighted_kernel(pos_eval, pos_mic, wave_num, weighting_mat)
    estimate = reconstruct_from_kernel(gamma_eval, krr_params)

    if verbose:
        metadata = {}
        for mat_name, mat in {"gamma":gamma, "gamma_r3" : gamma_r3, "gamma_bar" : gamma_bar, "system_matrix_reg" : system_matrix_reg}.items():
            metadata[f"{mat_name} max eigenvalue"] = splin.eigvalsh(mat, subset_by_index = (num_pos*ir_len-2, num_pos*ir_len-1)).tolist()
            metadata[f"{mat_name} min eigenvalue"] = splin.eigvalsh(mat, subset_by_index = (0, 1)).tolist()
            metadata[f"{mat_name} condition"] = np.linalg.cond(mat).tolist()
        return estimate, krr_params, gamma, gamma_r3, metadata
    return estimate