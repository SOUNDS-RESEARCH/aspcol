"""Spatial covariance estimation using kernel ridge regression

IMPORTANT: The spatial covariance matrices in this module is defined as the complex conjugate of the spatial covariance matrices in soundfieldcontrol.py. This module is consistent with the definition in the paper [brunnstromSpatial2025], which is the complex conjugate of the definition in many sound zone control papers. 

References
----------
[brunnstromSpatial2025] Jesper Brunnstrom, Martin Bo Møller, Jan Østergaard, Toon van Waterschoot, Marc Moonen, and Filip Elvander, Spatial covariance estimation for sound field reproduction using kernel ridge regression, European Signal Processing Conference (EUSIPCO), 2025.

"""
import numpy as np
import copy 

import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)

import aspcore.matrices_jax as jmat

import aspcol.kernelinterpolation as ki
import aspcol.kernelinterpolation_jax as jki
import aspcore.montecarlo as mc

import riecovest.distance as covdist


def _basic_krr_cost(krr_params, ir_data, data_weighting, reg_param, pos_mic, wave_num):
    C = _standard_krr_cost(krr_params, ir_data, data_weighting, reg_param, pos_mic, wave_num)
    return jnp.mean(C)

def _cov_informed_krr_cost_frobenius(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat, integral_weighting, cov_data, cov_reg_param):
    C = _standard_krr_cost(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat)
    cov_est = spatial_cov_from_integral_weighting(krr_params, integral_weighting)

    cov_error = jnp.mean(jnp.abs(cov_est - cov_data)**2, axis=(1,2))
    C = C + cov_reg_param * cov_error
    total_cost = jnp.mean(C)
    return total_cost

def _cov_informed_krr_cost_gevd(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat, integral_weighting, cov_data, cov_reg_param):
    C = _standard_krr_cost(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat)
    cov_est = spatial_cov_from_integral_weighting(krr_params, integral_weighting)

    cov_data = jax.vmap(jmat.regularize_matrix_with_condition_number, in_axes=(0, None))(cov_data, 1e6)
    cov_error = jax.vmap(covdist.frob_gevd_weighted, in_axes=(0, 0, None))(cov_est, cov_data, 4)**2#cov_est.shape[-1])**2
    C = C + cov_reg_param * jnp.log(cov_error)

    total_cost = jnp.mean(C)
    return total_cost

def _cov_informed_krr_cost_wishart(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat, integral_weighting, cov_data, cov_reg_param):
    C = _standard_krr_cost(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat)
    cov_est = spatial_cov_from_integral_weighting(krr_params, integral_weighting)

    N = 5
    cov_error = -jax.vmap(covdist.wishart_log_likelihood, in_axes=(0,0,None,None))(N * cov_data, cov_est, N, 1e8)
    C = C + cov_reg_param * cov_error

    #cost_per_freq = jnp.mean(C, axis=-1)
    total_cost = jnp.mean(C)
    return total_cost

def _cov_informed_krr_cost_airm(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat, integral_weighting, cov_data, cov_reg_param):
    C = _standard_krr_cost(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat)

    cov_est = spatial_cov_from_integral_weighting(krr_params, integral_weighting)
    #cov_data = jax.vmap(jmat.regularize_matrix_with_condition_number, in_axes=(0, None))(cov_data, 1e8)
    cov_est = jax.vmap(jmat.regularize_matrix_with_condition_number, in_axes=(0, None))(cov_est, 1e6)
    cov_error = jax.vmap(covdist.airm)(cov_est, cov_data)
    C = C + cov_reg_param * cov_error
    total_cost = jnp.mean(C)
    return total_cost

def _cov_informed_krr_cost_wasserstein(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat, integral_weighting, cov_data, cov_reg_param):
    C = _standard_krr_cost(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat)

    cov_est = spatial_cov_from_integral_weighting(krr_params, integral_weighting)
    #cov_data = jax.vmap(jmat.regularize_matrix_with_condition_number, in_axes=(0, None))(cov_data, 1e8)
    cov_est = jax.vmap(jmat.regularize_matrix_with_condition_number, in_axes=(0, None))(cov_est, 1e9)
    cov_error = jax.vmap(covdist.wasserstein_distance)(cov_est, cov_data)
    C = C + cov_reg_param * cov_error
    total_cost = jnp.mean(C)
    return total_cost



def krr_estimation_cov_informed(ir_data, pos_mic, wave_num, reg_param, integral_pos_func, integral_volume, num_mc_samples, cov_data, cov_reg_param, cost_func=_cov_informed_krr_cost_frobenius, kernel_func=None, kernel_args=None, num_steps=10000, learning_rate=1e-2):
    """The proposed spatial covariance estimation  method of [brunnstromSpatial2025].

    Parameters
    ----------
    ir_data : ndarray of shape (num_freq, num_source, num_data)
        the measured impulse responses to be used in the spatial covariance estimation
    pos_mic : ndarray of shape (num_mic, 3)
        positions of the microphones
    wave_num : ndarray of shape (num_freqs)
        the wavenumbers of all considered frequencies, defined as 2 pi f / c, where c is the speed of sound
    reg_param : float
        regularization parameter for the kernel ridge regression
    integral_pos_func : function or np.ndarray of shape (num_positions, 3)
        function that generates random positions with uniform distrbution for the integral. 
        Should take an integer as input and return a ndarray of shape (num_positions, 3)
        alternatively a precomputed array of positions can be provided, in which case num_mc_samples is ignored
    integral_volume : float
        volume of the integral region. Is needed to obtain the correct scaling of the spatial covariance
    num_mc_samples : int
        number of Monte Carlo samples used in the integral approximation
    cov_data : ndarray of shape (num_mic, num_mic, num_freqs)
        the spatial covariance matrices to be used in the cost function. This represents the current best estimate of the spatial covariance. 
    cov_reg_param : float
        The scaling parameter of the spatial covariance fitting term in the cost function. If this is set higher, the spatial covariance will be fitted more closely, 
        at the expense of the ir_data fitting term.
    cost_func : function, optional
        The cost function to be used in the optimization. The default is _cov_informed_krr_cost_frobenius. 
        Other options are _cov_informed_krr_cost_gevd, _cov_informed_krr_cost_wishart, _cov_informed_krr_cost_airm, _cov_informed_krr_cost_wasserstein. 
        They are all similar in implementation, and so it should be easy to add new cost functions.
    kernel_func : function, optional
        kernel function used in the spatial covariance, by default the diffuse kernel. Any single-frequency kernel from kernelinterpolation.py can be used, such as the directional kernel.
    kernel_args : list, optional
        additional arguments to the kernel function, by default None. 
    num_steps : int, optional
        number of optimization steps, by default 10000

    Returns
    -------
    krr_params : ndarray of shape (num_freqs, num_source, num_mic)
        The estimated kernel ridge regression parameters. 
        They can be used to compute the spatial covariance using spatial_cov_from_integral_weighting.
        These could also be used to reconstruct the sound field at any position using the kernel interpolation method.
        
    References
    ----------
    [brunnstromSpatial2025] Jesper Brunnstrom, Martin Bo Møller, Jan Østergaard, Toon van Waterschoot, Marc Moonen, and Filip Elvander, Spatial covariance estimation for sound field reproduction using kernel ridge regression, European Signal Processing Conference (EUSIPCO), 2025.
    """
    rng = np.random.default_rng(1234567)
    if kernel_func is None:
        kernel_func = ki.kernel_helmholtz_3d
    if kernel_args is None:
        kernel_args = []

    num_mic = pos_mic.shape[0]
    num_freqs = wave_num.shape[-1]
    num_source = ir_data.shape[1]

    if np.array(cov_reg_param).ndim == 0:
        cov_reg_param = np.ones((num_freqs)) * cov_reg_param

    integral_weighting = spatial_cov_weighting(pos_mic, wave_num, copy.deepcopy(integral_pos_func), integral_volume, num_mc_samples, kernel_func, kernel_args)
    data_weighting = kernel_func(pos_mic, pos_mic, wave_num, *kernel_args)
    data_weighting = jnp.array(data_weighting, dtype=complex)

    num_data = ir_data.shape[-1]
    pos_data = pos_mic[:num_data, :]
    reconstruct_mat = kernel_func(pos_data, pos_mic, wave_num, *kernel_args)

    krr_param_init = rng.normal(size=(num_freqs, num_source, num_mic))+ 1j*rng.normal(size=(num_freqs, num_source, num_mic))
    krr_params = jnp.array(krr_param_init, dtype=complex)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(krr_params)

    @jax.jit
    def opt_step(krr_params_arg, opt_state_arg):
        grads = jax.grad(cost_func)(krr_params_arg, ir_data, data_weighting, reg_param, reconstruct_mat, integral_weighting, cov_data, cov_reg_param)
        grads = jnp.conj(grads) # Jax returns conjugate of correct gradient
        #krr_params_arg = krr_params_arg - learning_rate * jnp.conj(grads)
        updates, opt_state_arg = optimizer.update(grads, opt_state_arg)
        krr_params_arg = optax.apply_updates(krr_params_arg, updates)
        return krr_params_arg, opt_state_arg

    mat_history = []
    for i in range(num_steps):
        if i % 1000 == 0:
            print(f"iter: {i}")
            mat_history.append(krr_params)
            if len(mat_history) > 1:
                diff = jnp.mean(jnp.abs(mat_history[-1] - mat_history[-2]))
                cost = cost_func(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat, integral_weighting, cov_data, cov_reg_param)
                print(f"Mean reg difference: {diff}")
                print(f"Cost: {cost}")
                if diff < 1e-6:
                    break
        krr_params, opt_state = opt_step(krr_params, opt_state)
    return krr_params




def spatial_cov_weighting(pos_mic, wave_num, integral_pos_func, integral_volume, num_mc_samples, kernel_func=None, kernel_args=None):
    """Calculates the weighting matrix used in the interpolated spatial covariance

    Corresponds to K in [brunnstromSpatial2025]
    
    Parameters
    ----------
    pos_mic : ndarray of shape (num_mic, 3)
        positions of the microphones
    wave_num : ndarray of shape (num_freqs)
        the wavenumbers of all considered frequencies, defined as 2 pi f / c, where c is the speed of sound
    integral_pos_func : function or np.ndarray of shape (num_positions, 3)
        function that generates positions for the integral. Should take an integer as input and return a ndarray of shape (num_positions, 3)
        alternatively a precomputed array of positions can be provided, in which case num_mc_samples is ignored
    integral_volume : float
        volume of the integral region
    num_mc_samples : int
        number of Monte Carlo samples used in the integral approximation
    kernel_func : function, optional
        kernel function used in the spatial covariance, by default the diffuse kernel
    kernel_args : list, optional
        additional arguments to the kernel function, by default None

    Returns
    -------
    spatial_cov
    """
    if kernel_func is None:
        kernel_func = ki.kernel_helmholtz_3d
    if kernel_args is None:
        kernel_args = []

    def integrand(r):
        kappa = kernel_func(r, pos_mic, wave_num, *kernel_args)
        if kappa.ndim == 3:
            kappa = kappa[:,None,:,:]
        kappa = np.moveaxis(kappa, -2, -1)
        cov_mat = kappa[:,:,:,None,None,:] * kappa[:,None,None,:,:,:].conj()
        return cov_mat

    if callable(integral_pos_func):
        integral_val = mc.integrate(integrand, integral_pos_func, num_mc_samples, integral_volume)
    else:
        integral_val = np.mean(integrand(integral_pos_func), axis=-1) * integral_volume
    
    integral_val /= integral_volume # Normalize so that is corresponds to the space-discrete spatial covariance
    return integral_val

def spatial_cov_from_integral_weighting(krr_params, integral_weighting):
    return np.sum(np.sum(krr_params[:,:,:,None,None] * integral_weighting, axis=2) * krr_params[:,None,:,:].conj(), axis=-1)

def spatial_cov_from_integral_weighting_diffuse(krr_params, integral_weighting):
    if integral_weighting.ndim == 5:
        integral_weighting = np.squeeze(integral_weighting, axis=(1, 3))
    assert integral_weighting.ndim == 3
    return krr_params @ integral_weighting @ jnp.moveaxis(krr_params, 1,2).conj()

def _spatial_cov_weighting_diffuse(pos_mic, wave_num, integral_pos_func, integral_volume, num_mc_samples, kernel_func=None, kernel_args=None):
    if kernel_func is None:
        kernel_func = ki.kernel_helmholtz_3d
    if kernel_args is None:
        kernel_args = []

    #currently assumes we can use the same kernel for all sources, i.e. both kernel and mic positions are the same
    def integrand(r):
        kappa = kernel_func(r, pos_mic, wave_num, *kernel_args)
        kappa = np.moveaxis(kappa, 1, 2)
        return kappa[:,:,None,:] * kappa[:,None,:,:].conj()

    num_mic = pos_mic.shape[0]
    weighting_mat = mc.integrate(integrand, integral_pos_func, num_mc_samples, integral_volume)
    weighting_mat /= integral_volume # Normalize so that is corresponds to the space-discrete spatial covariance
    return weighting_mat



@jax.jit
def reconstruct_freq(krr_params, pos_output, pos_data, wave_num):
    kernel_val = jki.diffuse_kernel(pos_output, pos_data, wave_num)

    reconstructed = kernel_val @ krr_params[:,:,None]
    return jnp.squeeze(reconstructed, axis=-1)

@jax.jit
def reconstruct_from_mat(krr_params, kernel_mat):
    """
    The kernel mat is jki.diffuse_kernel(pos_output, pos_data, wave_num), or
    the corresponding kernel for the desired kernel function
    """
    reconstructed = kernel_mat @ krr_params[:,:,None]
    return jnp.squeeze(reconstructed, axis=-1)

def krr_estimation_sgd(ir_data, pos_mic, wave_num, reg_param, kernel_func=None, kernel_args=None, num_steps=10000):
    rng = np.random.default_rng(1234567)
    if kernel_func is None:
        kernel_func = ki.kernel_helmholtz_3d
    if kernel_args is None:
        kernel_args = []

    num_mic = pos_mic.shape[0]
    num_freqs = wave_num.shape[-1]
    num_source = ir_data.shape[1]

    #integral_weighting = spatial_cov_weighting(pos_mic, wave_num, integral_pos_func, integral_volume, num_mc_samples, kernel_func, kernel_args)
    data_weighting = kernel_func(pos_mic, pos_mic, wave_num, *kernel_args)
    data_weighting = jnp.array(data_weighting, dtype=complex)
    
    krr_param_init = rng.normal(size=(num_freqs, num_source, num_mic))+ 1j*rng.normal(size=(num_freqs, num_source, num_mic))
    krr_params = jnp.array(krr_param_init, dtype=complex)

    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(krr_params)

    @jax.jit
    def opt_step(krr_params_arg, opt_state_arg):
        grads = jax.grad(_basic_krr_cost)(krr_params_arg, ir_data, data_weighting, reg_param, pos_mic, wave_num)
        grads = jnp.conj(grads) # Jax returns conjugate of correct gradient
        #krr_params_arg = krr_params_arg - learning_rate * jnp.conj(grads)
        updates, opt_state_arg = optimizer.update(grads, opt_state_arg)
        krr_params_arg = optax.apply_updates(krr_params_arg, updates)
        return krr_params_arg, opt_state_arg

    mat_history = []
    for i in range(num_steps):
        if i % 1000 == 0:
            print(f"iter: {i}")
            mat_history.append(krr_params)
            if len(mat_history) > 1:
                diff = jnp.mean(jnp.abs(mat_history[-1] - mat_history[-2]))
                cost = _basic_krr_cost(krr_params, ir_data, data_weighting, reg_param, pos_mic, wave_num)
                print(f"Mean reg difference: {diff}")
                print(f"Cost: {cost}")
                if diff < 1e-6:
                    break
        krr_params, opt_state = opt_step(krr_params, opt_state)
    return krr_params

def _standard_krr_cost(krr_params, ir_data, data_weighting, reg_param, reconstruct_mat):
    C_s = []
    num_sources = krr_params.shape[1]
    for i in range(num_sources):
        if reconstruct_mat.ndim == 4:
            rmat = reconstruct_mat[:,i,:,:]
        else:
            rmat = reconstruct_mat

        sound_field_est = reconstruct_from_mat(krr_params[:,i,:], rmat)
        data_error = jnp.mean(jnp.abs(sound_field_est - ir_data[:,i,:])**2, axis=-1)

        if data_weighting.ndim == 4:
            dw = data_weighting[:,i,:,:]
        else:
            dw = data_weighting
        reg_error = reg_param * jnp.squeeze(krr_params[:,i,None,:].conj() @ dw @ krr_params[:,i,:,None], axis=(-1, -2))
        C_s.append(jnp.real(data_error) + jnp.real(reg_error))
    C = jnp.mean(jnp.stack(C_s, axis=-1), axis=-1)
    return C

