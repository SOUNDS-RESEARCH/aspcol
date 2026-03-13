"""Module for recording sound fields using a moving microphone

The estimation of a sound field from moving microphones is very computationally costly. Especially for directional microphones, the computational cost of the estimation can be prohibitive. Therefore a lot of the code in this module is implemented in jax, such that the resulting functions can be compiled, leading to considerable improvements in running time. 

Due to the need of re-implementing functions such as the spherical Bessel function in jax, the current compilable implementations in this module are somewhat restricted. There is jax implementations of e.g. the translation operator that can also be found in the module sphericalharmonics.py, but in this module it assumes order 0 and 1 harmonic coefficients only. 

The sound field estimation function inf_dimensional_shd_dynamic cannot deal with directionalities above order 1.

References
----------
[brunnstromBayesianSubmitted] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted. \n
"""

import numpy as np
import scipy.linalg as splin
import scipy.special as spspec
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jax_splin


from functools import partial
#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_debug_nans", True)


import aspcore.fouriertransform as ft_numpy
import aspcore.fouriertransform_jax as ft
import aspcore.matrices_jax as aspmat
import aspcore.montecarlo_jax as mc
import aspcore.linear_systems_jax as ls
import aspcore.quadrature_jax as quad

import aspcol.sphericalharmonics as shd_numpy
import aspcol.sphericalharmonics_jax as shd
import aspcol.planewaves_jax as pw

import aspcol.kernelinterpolation_jax as kernel



def _parse_moving_mic_args(p, pos, pos_eval, sequence):
    if p.ndim >= 2:
        p = jnp.squeeze(p)
    N = p.shape[0]

    if sequence.ndim == 2:
        sequence = jnp.squeeze(sequence, axis=0)
    assert sequence.ndim == 1
    seq_len = sequence.shape[0]
    assert seq_len % 2 == 0 #Calculations later assume seq_len is even to get the Nyquist frequency
    num_periods = N // seq_len
    assert N % seq_len == 0

    assert pos.shape == (N, 3)
    assert pos_eval.ndim == 2 and pos_eval.shape[1] == 3
    return p, pos, pos_eval, sequence, N, seq_len, num_periods


def inf_dimensional_shd_dynamic(p, pos, pos_eval, sequence, samplerate, c, reg_param, dir_coeffs, verbose=False):
    """
    Estimates the RIR at evaluation positions using data from a moving microphone
    using Bayesian inference of an infinite sequence of spherical harmonics

    Implements the method in J. Brunnström, M.B. Moeller, M. Moonen, 
    "Bayesian sound field estimation using moving microphones" 

    Assumptions:
    The noise covariance is a scaled identity matrix
    The data is measured over an integer number of periods of the sequence
    N = seq_len * M, where M is the number of periods that was measured
    The length of sequence is the length of the estimated RIR

    Parameters
    ----------
    p : ndarray of shape (N)
        sound pressure for each sample of the moving microphone
    pos : ndarray of shape (N, 3)
        position of the trajectory for each sample
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    sequence : ndarray of shape (seq_len) or (1, seq_len)
        the training signal used for the measurements
    samplerate : int
    c : float
        speed of sound
    reg_param : float
        regularization parameter
    dir_coeffs : ndarray of shape (N, num_coeffs)
        harmonic coefficients of microphone directivity, 
        Note that a higher number of coefficients will drastically increase the computational cost
    verbose : bool, optional
        if True, returns diagnostics, by default False

    Returns
    -------
    shd_coeffs : ndarray of shape (num_real_freqs, num_eval)
        time-domain harmonic coefficients of the estimated sound field
    """
    # ======= Argument parsing and constants =======
    p, pos, pos_eval, sequence, N, seq_len, num_periods = _parse_moving_mic_args(p, pos, pos_eval, sequence)
    # if p.ndim >= 2:
    #     p = np.squeeze(p)
    # N = p.shape[0]

    # if sequence.ndim == 2:
    #     sequence = np.squeeze(sequence, axis=0)
    # assert sequence.ndim == 1
    # seq_len = sequence.shape[0]
    # assert seq_len % 2 == 0 #Calculations later assume seq_len is even to get the Nyquist frequency
    # num_periods = N // seq_len
    # assert N % seq_len == 0

    wave_num = ft_numpy.get_real_wavenum(seq_len, samplerate, c)
    num_real_freqs = wave_num.shape[-1]
    #len(ft.get_real_freqs(seq_len, samplerate))

    #assert pos.shape == (N, 3)
    #assert pos_eval.ndim == 2 and pos_eval.shape[1] == 3

    if dir_coeffs.ndim == 2:
        dir_coeffs = dir_coeffs[None,:,:] #add a dimension for the number of frequencies
    assert dir_coeffs.ndim == 3
    assert dir_coeffs.shape[1] == N or dir_coeffs.shape[1] == 1
    assert dir_coeffs.shape[0] == num_real_freqs or dir_coeffs.shape[0] == 1
    
    # ======= Estimation of spherical harmonic coefficients =======
    Phi = _sequence_stft_bayesian_multiperiod_numpy(sequence, num_periods)
    Phi = Phi[:num_real_freqs,:]
    
    psi = calculate_psi(pos, dir_coeffs, wave_num, Phi, num_real_freqs)
    psi = np.asarray(psi) # convert to numpy array, as jax is no longer necessary
    noise_cov = reg_param * np.eye(N)
    psi_plus_noise_cov = psi + noise_cov
    try:
        regressor = splin.solve(psi_plus_noise_cov, p, assume_a = "pos")
    except np.linalg.LinAlgError:
        print(f"LinAlgError at reg_param = {reg_param}. Singular psi (plus noise cov) matrix")
        print(f"Calculating least-squares solution instead")
        regressor = splin.lstsq(psi_plus_noise_cov, p)[0]
    regressor = Phi.conj() * regressor[None,:]

    if verbose:
        print(f"Computing eval estimates from regressor")
    est_sound_pressure = estimate_from_regressor(regressor, pos, pos_eval, wave_num, dir_coeffs)

    if verbose:
        return est_sound_pressure, regressor, psi
    return est_sound_pressure


def estimate_from_regressor(regressor, pos, pos_eval, wave_num, dir_coeffs = None):
    """Takes the regressor from inf_dimensional_shd_dynamic, and gives back a sound field estimate. 

    IMPORTANT: This function is not JIT compatible. But it manages to compile the most costly 
    parts of the computation. 

    Gives the same result as inf_dimensional_shd_dynamic, but is much faster since computing the regressor is the primary 
    computational cost. 
    Implements the method in J. Brunnström, M.B. Moeller, M. Moonen, "Bayesian sound field estimation using moving microphones" 


    Parameters
    ----------
    regressor : ndarray of shape (num_real_freqs, N)
        The regressor calculated by inf_dimensional_shd_dynamic. 
        Represents Phi* v in eq (31) from [brunnstromBayesian2024]
    pos : ndarray of shape (N, 3)
        position of the trajectory for each sample
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    samplerate : int
    c : float
        speed of sound
    dir_coeffs : ndarray of shape (N, num_coeffs), optional
        harmonic coefficients of microphone directivity, 
        Note that a higher number of coefficients will drastically increase the computational cost
        If not provided, the microphones are assumed to be omnidirectional.

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        time-domain harmonic coefficients of the estimated sound field

    Notes
    -----
    Assumptions:
    The data is measured over an integer number of periods of the sequence
    N = seq_len * M, where M is the number of periods that was measured
    The length of sequence is the length of the estimated RIR
    """
    # ======= Argument parsing and constants =======
    N = pos.shape[0]
    assert pos.shape == (N, 3)
    assert pos_eval.ndim == 2 and pos_eval.shape[1] == 3

    num_real_freqs = wave_num.shape[-1]
    assert wave_num.ndim == 1
    assert regressor.shape == (num_real_freqs, N)

    if dir_coeffs is None:
        dir_coeffs = shd_numpy.directivity_omni() * np.ones((N, 1))
    if dir_coeffs.ndim == 2:
        dir_coeffs = dir_coeffs[None,:,:] #add a dimension for the number of frequencies
    assert dir_coeffs.ndim == 3
    assert dir_coeffs.shape[1] == N or dir_coeffs.shape[1] == 1
    assert dir_coeffs.shape[0] == num_real_freqs or dir_coeffs.shape[0] == 1

    if dir_coeffs.shape[0] != 1:
        raise NotImplementedError("Only implemented for frequency-independent directivity coefficients")
    #dir_coeffs = dir_coeffs[0,:,:]

    # ======= Reconstruction of RIR =======
    num_eval = pos_eval.shape[0]
    dir_omni = shd_numpy.directivity_omni() #* np.ones((num_eval, 1))
    dir_omni = dir_omni[None,:,:] # add a dimension for the number of frequencies

    num_eval = pos_eval.shape[0]
    est_sound_pressure = np.zeros(((num_real_freqs, num_eval)), dtype=complex)
    for i in range(num_eval):
        #if i % 10 == 0:
        print(f"Estimating sound pressure at position {i} of {num_eval}")
        kernel_val = shd_numpy.translated_inner_product(pos_eval[i:i+1,:], pos, dir_omni, dir_coeffs, wave_num)
        est_sound_pressure[:,i:i+1] = np.squeeze(kernel_val @ regressor[:,:,None], axis=-1)

    # MAKE A TRANSLATION OPERATOR THAT WORKS FOR MIXED 0TH AND 1ST ORDERS. THEN IMPLEMENT FOLLOWING
    #max_order = shd.shd_max_order(dir_coeffs.shape[-1])
    #gaunt_set = _calculate_gaunt_set(0, max_order)
    #return _estimate_from_regressor_compiled(regressor, pos, pos_eval, wave_num, dir_omni, dir_coeffs, gaunt_set)
    return est_sound_pressure


def calculate_psi(pos, dir_coeffs, wave_num, Phi, num_real_freqs):
    if dir_coeffs.ndim == 2:
        dir_coeffs = dir_coeffs[:,None,:]
    assert pos.ndim == 2
    assert pos.shape[1] == 3
    num_pos = pos.shape[0]

    assert wave_num.ndim == 1
    assert wave_num.shape[0] == num_real_freqs
    
    assert dir_coeffs.ndim == 3
    if dir_coeffs.shape[0] == wave_num.shape[0]:
        raise NotImplementedError("Only implemented for frequency-independent directivity coefficients")
    assert dir_coeffs.shape[0] == 1
    assert dir_coeffs.shape[1] == pos.shape[0] or dir_coeffs.shape[1] == 1

    assert Phi.ndim == 2
    assert Phi.shape == (num_real_freqs, num_pos)

    max_order = 1
    gaunt_set = shd._calculate_gaunt_set(max_order, max_order)

    psi = _calculate_psi_compiled(pos, dir_coeffs, wave_num, Phi, gaunt_set)
    return psi


@jax.jit
def _calculate_psi_compiled(pos, dir_coeffs, wave_num, Phi, gaunt_set):
    num_pos = pos.shape[0]

    Phi = Phi.T # we need to scan over leading axis
    dir_coeffs = dir_coeffs[0,:,:] # assume frequency independent directivity

    def psi_scan_loop(carry, arg_slice):
        (pos_i, phi_i, dir_coeffs_i) = arg_slice

        def psi_scan_inner_loop(carry_inner, arg_slice_inner):
            (pos_j, phi_j, dir_coeffs_j) = arg_slice_inner

            pos_diff = pos_i[None,:] - pos_j[None,:]
            phi_factor = phi_i * jnp.conj(phi_j)

            T = shd.translation_operator(pos_diff, wave_num, gaunt_set)
            inner_product = jnp.moveaxis(jnp.conj(dir_coeffs_i)[None,None,:,None], -1, -2) @ T @ dir_coeffs_j[None,None,:,None]
            inner_product = jnp.squeeze(inner_product) * phi_factor

            psi_ij = 2 * jnp.sum(jnp.real(inner_product[1:-1,...]))
            psi_ij = psi_ij + jnp.real(inner_product[0,...]) + jnp.real(inner_product[-1,...])

            return carry_inner, psi_ij
        _, psi_i = jax.lax.scan(psi_scan_inner_loop, 0, (pos, Phi, dir_coeffs))        
        return (carry, psi_i)
    _, psi = jax.lax.scan(psi_scan_loop, 0, (pos, Phi, dir_coeffs))
    return psi








def _sequence_stft_bayesian_multiperiod_numpy(sequence, num_periods):
    """
    Assumes that the sequence is periodic.
    Assumes that sequence argument only contains one period
    
    Parameters
    ----------
    sequence : ndarray of shape (seq_len,)
    num_periods : int

    Returns
    -------
    Phi : ndarray of shape (seq_len, num_periods*seq_len)
    """
    Phi = _seq_stft_bayesian_numpy(sequence)
    return np.tile(Phi, (1, num_periods))

def _seq_stft_bayesian_numpy(sequence):
    """

    Parameters
    ----------
    sequence : ndarray of shape (seq_len,)

    Assume the sequence is periodic with period B

    Returns
    -------
    Phi : ndarray of shape (seq_len, seq_len)
        first axis contains frequency bins
        second axis contains time indices
    
    """
    if sequence.ndim == 2:
        sequence = np.squeeze(sequence, axis=0)
    assert sequence.ndim == 1
    B = sequence.shape[0]

    Phi = np.zeros((B, B), dtype=complex)
    for n in range(B):
        Phi[:,n] = ft_numpy.fft(np.roll(sequence, -n)) / B
    return Phi

@partial(jax.jit, static_argnames=["num_periods"])
def _seq_stft_bayesian_multiperiod(sequence, num_periods):
    """
    Assumes that the sequence is periodic.
    Assumes that sequence argument only contains one period
    
    Parameters
    ----------
    sequence : ndarray of shape (seq_len,)
    num_periods : int

    Returns
    -------
    Phi : ndarray of shape (seq_len, num_periods*seq_len)
    """
    Phi = _seq_stft_bayesian(sequence)
    return jnp.tile(Phi, (1, num_periods))

def _seq_stft_bayesian(sequence):
    """
    Assumes the sequence is periodic with period B

    Parameters
    ----------
    sequence : ndarray of shape (seq_len,)

    Returns
    -------
    Phi : ndarray of shape (num_real_freqs, seq_len)
        first axis contains frequency bins
        second axis contains time indices
    
    """
    if sequence.ndim == 2:
        sequence = jnp.squeeze(sequence, axis=0)
    #assert sequence.ndim == 1
    B = sequence.shape[0]

    def inner_func(n):
        return ft.rfft(jnp.roll(sequence, -n)) / B

    phis = jax.vmap(inner_func, out_axes=1)(jnp.arange(B))
    return phis


@partial(jax.jit, static_argnames=["num_periods"])
def _seq_stft_krr_multiperiod(sequence, num_periods):
    """
    Assumes that the sequence is periodic.
    Assumes that sequence argument only contains one period
    
    Parameters
    ----------
    sequence : ndarray of shape (seq_len,)
    num_periods : int

    Returns
    -------
    Phi : ndarray of shape (seq_len, num_periods*seq_len)
    """
    Phi = _seq_stft_krr(sequence)
    return jnp.tile(Phi, (1, num_periods))

def _seq_stft_krr(sequence):
    """
    Assumes the sequence is periodic with period B

    Parameters
    ----------
    sequence : ndarray of shape (seq_len,)

    Returns
    -------
    Phi : ndarray of shape (num_real_freqs, seq_len)
        first axis contains frequency bins
        second axis contains time indices
    
    """
    if sequence.ndim == 2:
        sequence = jnp.squeeze(sequence, axis=0)
    B = sequence.shape[0]

    def inner_func(n):
        phi_n = jnp.roll(sequence, -n) #so that n is the first element
        phi_n = jnp.roll(phi_n, -1) # so that n ends up last
        phi_n = jnp.flip(phi_n) # so that we get n first and then n-i as we move later in the vector
        return ft.rfft(phi_n)

    phis = jax.vmap(inner_func, out_axes=1)(jnp.arange(B))
    return phis

























@partial(jax.jit, static_argnames=["return_params", "batch_size"])
def krr_moving_mic_directional(p, pos, pos_eval, sequence, samplerate, c, reg_param, direction, beta, return_params=False, batch_size=8):
    """Estimates the RIR at evaluation positions using data from a moving omnidirectional microphone

    Parameters
    ----------
    p : ndarray of shape (N)
        sound pressure for each sample of the moving microphone
    pos : ndarray of shape (N, 3)
        position of the trajectory for each sample
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    sequence : ndarray of shape (seq_len) or (1, seq_len)
        the training signal used for the measurements
    samplerate : int
    c : float
        speed of sound
    reg_param : float
        regularization parameter

    Returns
    -------
    
    """
    # ======= Argument parsing and constants =======
    p, pos, pos_eval, sequence, N, seq_len, num_periods = _parse_moving_mic_args(p, pos, pos_eval, sequence)

    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)

    phi_f = _seq_stft_krr_multiperiod(sequence, num_periods)
    K = _calc_directional_kernel_mat(pos, wave_num, phi_f, direction, beta, seq_len, batch_size=batch_size)

    # reg_param is scaled so that it is equivalent to the Bayesian regularization parameter
    reg_matrix = seq_len * reg_param * jnp.eye(N)
    krr_params = jax_splin.solve(K + reg_matrix, p, assume_a="pos")
    krr_params = phi_f * krr_params[None,:]

    est_sound_pressure = reconstruct_krr_moving_mic_directional(krr_params, pos_eval, pos, wave_num, direction, beta, batch_size=batch_size)
    if return_params:
        return est_sound_pressure, krr_params, K
    return est_sound_pressure

@partial(jax.jit, static_argnames=["seq_len", "batch_size"])
def _calc_directional_kernel_mat(pos, wave_num, phi_f, direction, beta, seq_len, batch_size=8):
    dft_weighting = ft.rdft_weighting(seq_len)

    def _kernel_inner_loop(system_mat, scanned_args):
        (wave_num_single, phi_single, dft_weight) = scanned_args
        phi_rank1_matrix = phi_single[:,None].conj() * phi_single[None,:]
        system_mat_incr = dft_weight * jnp.real(jnp.squeeze(kernel.kernel_directional_vonmises(pos, pos, wave_num_single, direction, beta)) * phi_rank1_matrix)
        system_mat = system_mat + system_mat_incr
        return system_mat, system_mat

    kernel_system_mat = jnp.zeros((pos.shape[0], pos.shape[0]), dtype=float)
    kernel_system_mat, _ = jax.lax.scan(_kernel_inner_loop, kernel_system_mat, (wave_num, phi_f, dft_weighting), unroll=batch_size)
    return kernel_system_mat


def reconstruct_krr_moving_mic_directional(krr_params, pos_eval, pos_mic, wave_num, direction, beta, batch_size=8):
    """Takes the KRR parameters, and gives back a sound field estimate. 

    Parameters
    ----------
    krr_params : ndarray of shape (num_real_freqs, N)
        krr parameters from krr_moving_mic_directional
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    pos : ndarray of shape (N, 3)
        positions of the trajectory for each sample
    k : ndarray of shape (num_freq)
        wavenumbers

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points"""
    def _reconstruct_inner_loop(pos_eval_single):
        kernel_val = kernel.kernel_directional_vonmises(pos_eval_single[None,:], pos_mic, wave_num, direction, beta).astype(complex)
        kernel_val = jnp.squeeze(kernel_val, axis=1) # remove axis corresponding to single_eval
        p_est = jnp.sum(kernel_val * krr_params, axis=-1)
        return p_est
    
    estimate = jnp.moveaxis(jax.lax.map(_reconstruct_inner_loop, pos_eval, batch_size=batch_size), 0, 1)
    return estimate


@partial(jax.jit, static_argnames=["return_params"])
def krr_moving_mic_diffuse(p, pos, pos_eval, sequence, samplerate, c, reg_param, return_params=False):
    """Estimates the RIR at evaluation positions using data from a moving omnidirectional microphone

    reg_param is scaled by seq_len to have the same effect as the Bayesian method

    Parameters
    ----------
    p : ndarray of shape (N)
        sound pressure for each sample of the moving microphone
    pos : ndarray of shape (N, 3)
        position of the trajectory for each sample
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    sequence : ndarray of shape (seq_len) or (1, seq_len)
        the training signal used for the measurements
    samplerate : int
    c : float
        speed of sound
    reg_param : float
        regularization parameter

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    p, pos, pos_eval, sequence, N, seq_len, num_periods = _parse_moving_mic_args(p, pos, pos_eval, sequence)
    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)

    phi_f = _seq_stft_krr_multiperiod(sequence, num_periods)
    K = _calc_diffuse_kernel_mat(pos, wave_num, phi_f, seq_len)

    reg_matrix = seq_len * reg_param * jnp.eye(N)
    krr_params = jax_splin.solve(K + reg_matrix, p, assume_a="pos")
    krr_params = phi_f * krr_params[None,:]

    est_sound_pressure = reconstruct_krr_moving_mic_diffuse(krr_params, pos_eval, pos, wave_num)
    if return_params:
        return est_sound_pressure, krr_params, K
    return est_sound_pressure

@partial(jax.jit, static_argnames=["seq_len", "batch_size"])
def _calc_diffuse_kernel_mat(pos, wave_num, Phi, seq_len, batch_size=8):
    dft_weighting = ft.rdft_weighting(seq_len)

    def _kernel_inner_loop(system_mat, scanned_args):
        (wave_num_single, phi_single, dft_weight) = scanned_args
        phi_rank1_matrix = phi_single[:,None].conj() * phi_single[None,:] #Phi[f,:,None] * Phi[f,None,:].conj()
        system_mat_incr = dft_weight * jnp.real(jnp.squeeze(kernel.kernel_diffuse(pos, pos, wave_num_single)) * phi_rank1_matrix)
        system_mat = system_mat + system_mat_incr
        return system_mat, system_mat

    kernel_system_mat = jnp.zeros((pos.shape[0], pos.shape[0]), dtype=float)
    kernel_system_mat, _ = jax.lax.scan(_kernel_inner_loop, kernel_system_mat, (wave_num, Phi, dft_weighting), unroll=batch_size)
    return kernel_system_mat

@partial(jax.jit, static_argnames=["batch_size"])
def reconstruct_krr_moving_mic_diffuse(krr_params, pos_eval, pos_mic, wave_num, batch_size=8):
    """Takes the regressor from inf_dimensional_shd_dynamic, and gives back a sound field estimate. 
    Reconstructs the sound field at the evaluation points using the regressor matrix
    from est_inf_dimensional_shd_dynamic

    Parameters
    ----------
    krr_params : ndarray of shape (num_real_freqs, N)
        regressor matrix from est_inf_dimensional_shd_dynamic
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    pos : ndarray of shape (N, 3)
        positions of the trajectory for each sample
    k : ndarray of shape (num_freq)
        wavenumbers

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    def _reconstruct_inner_loop(pos_eval_single):
        kernel_val = kernel.kernel_diffuse(pos_eval_single[None,:], pos_mic, wave_num).astype(complex)
        p_est = jnp.sum(kernel_val * krr_params[:,None,:], axis=-1)
        return jnp.squeeze(p_est, axis=-1)
    
    estimate = jnp.moveaxis(jax.lax.map(_reconstruct_inner_loop, pos_eval, batch_size=batch_size), 0, 1)
    return estimate


@partial(jax.jit, static_argnames="score")
def cross_validation_krr_moving_mic_diffuse(data, pos, sequence, samplerate, c, reg_params, score="gcv", **kwargs):
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
    best_reg_param : float
        regularization parameter that minimizes the chosen score
    score_values : ndarray of shape (len(reg_params),)
        score values for each regularization parameter

    References
    ----------
    Time-domain sound field estimation using kernel ridge regression, J Brunnström, M. B. Møller, J Østergaard, S Koyama, T van Waterschoot, M Moonen, 2025
    """
    data, pos, _, sequence, N, seq_len, num_periods = _parse_moving_mic_args(data, pos, jnp.zeros((1,3)), sequence)
    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)

    phi_f = _seq_stft_krr_multiperiod(sequence, num_periods)
    K = _calc_diffuse_kernel_mat(pos, wave_num, phi_f, seq_len)

    reg_matrix_base = jnp.eye(N)

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








def _random_rotation_matrix(key):
    """
    Generate a random 3x3 rotation matrix using axis–angle.
    """
    key_axis, key_angle = jax.random.split(key)

    # Random unit axis
    axis = jax.random.normal(key_axis, (3,))
    axis = axis / jnp.linalg.norm(axis)

    theta = jax.random.uniform(key_angle, (), minval=0.0, maxval=2 * jnp.pi)

    x, y, z = axis
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    C = 1.0 - c

    R = jnp.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])

    return R


@jax.jit
def _apply_random_rotation(A, key):
    """
    A: (num_points, 3) array of unit vectors
    """
    R = _random_rotation_matrix(key)
    return A @ R.T



@partial(jax.jit, static_argnames=["same_direction_for_all_freqs"])
def reconstruct_moving_mic_rff(params, pos_eval, wave_num, basis_directions, same_direction_for_all_freqs=True):
    z_eval = _get_rff_basis_vec(pos_eval, basis_directions, wave_num, same_direction_for_all_freqs)
    p_est = jnp.sum(z_eval * params[:,None,:], axis=-1) 
    return p_est

@partial(jax.jit, static_argnames=["num_basis", "return_params", "same_direction_for_all_freqs", "deterministic_directions"])
def krr_moving_mic_rff(p, pos, pos_eval, sequence, samplerate, c, reg_param, num_basis=64, key=None, return_params=False, direction = None, beta = None, same_direction_for_all_freqs=True, deterministic_directions = False):
    """Sound field estimation with moving microphone using KRR with random Fourier features
    
    Parameters
    ----------
    p : ndarray of shape (N)
        sound pressure for each sample of the moving microphone
    pos : ndarray of shape (N, 3)
        position of the trajectory for each sample
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    sequence : ndarray of shape (seq_len)
        N is a multiple of seq_len. The loudspeaker sequence used for the measurements, 
        which is assumed to be periodic with period seq_len
    samplerate : int
    c : float
        speed of sound
    reg_param : float
        regularization parameter
    num_basis : int, optional
        number of random basis directions to use, by default 64
    deterministic_directions : bool, optional
        if True, uses deterministic basis directions from a t-design instead of random ones. This will interpret num_basis as
        the order of the t-design, so the actual number of basis functions will be larger, usually not too far from num_basis**2. 
        Requires same_direction_for_all_freqs to be True.
        By default False.

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    Notes
    -----
    Uses the diffuse kernel implicitly
    the reg_param is not identical to the standard KRR methods, and so might have to be chosen differently
    The params returns by return_params can be interpreted as plane wave coefficients, but are
    not the same as the standard krr parameters

    """
    p, pos, pos_eval, sequence, N, seq_len, num_periods = _parse_moving_mic_args(p, pos, pos_eval, sequence)

    if key is None:
        key = jax.random.key(23456743)

    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)
    num_real_freqs = wave_num.shape[-1]
    phi_f = _seq_stft_krr_multiperiod(sequence, num_periods)

    if deterministic_directions:
        basis_directions = quad.t_design(num_basis) #num_basis interpreted as t-design order
        basis_directions = _apply_random_rotation(basis_directions, key) # to avoid always having the same directions for the same num_basis
        
        assert same_direction_for_all_freqs, "deterministic_directions is only implemented for same_direction_for_all_freqs=True"
        num_basis = basis_directions.shape[0]
        tot_num_basis = num_basis
    else:
        if same_direction_for_all_freqs:
            tot_num_basis = num_basis
        else:
            tot_num_basis = num_basis * num_real_freqs

        if direction is None:
            assert beta is None, "beta should not be provided if direction is not provided"
            basis_directions = mc.uniform_random_on_sphere(tot_num_basis, key)
        else:
            assert beta is not None, "beta should be provided if direction is provided"
            basis_directions = mc.vonmises_fisher_on_sphere(tot_num_basis, -direction, beta, key)
        if not same_direction_for_all_freqs:
            basis_directions = basis_directions.reshape((num_real_freqs, num_basis, 3))

    Z = _rff_z_matrix(-pos, wave_num, phi_f, basis_directions, num_basis, seq_len, N, same_direction_for_all_freqs)

    system_mat = Z.T @ Z
    system_mat = system_mat + seq_len * reg_param * jnp.eye(seq_len * num_basis, dtype=Z.dtype)
    projected_data = Z.T @ p

    params = jax_splin.solve(system_mat, projected_data, assume_a="pos")
    params = params.reshape(seq_len, num_basis)
    params = ft.real_vec_to_dft_domain(params, scale=True) # (num_real_freqs, num_basis)

    p_est = reconstruct_moving_mic_rff(params, pos_eval, wave_num, basis_directions, same_direction_for_all_freqs)

    if return_params:
        return p_est, params, basis_directions, Z
    return p_est # (num_real_freqs, num_eval)


def _get_rff_basis_vec(pos, basis_directions, wave_num, same_direction_for_all_freqs):
    """

    Parameters
    ----------
    basis_directions : array of shape (num_basis, 3) or (num_real_freqs, num_basis, 3)
        the former must be used if same_direction_for_all_freqs is True.
        the latter must be used if same_direction_for_all_freqs is False

    returns z of shape (num_real_freqs, num_pos, num_basis)
    """
    num_real_freqs = wave_num.shape[-1]
    if same_direction_for_all_freqs:
        num_basis = basis_directions.shape[0]
        z = pw.plane_wave(pos, basis_directions, wave_num) / jnp.sqrt(num_basis)
    else:
        num_basis = basis_directions.shape[1]
        z = jnp.stack([pw.plane_wave(pos, basis_directions[f,:,:], wave_num[f]) for f in range(num_real_freqs)], axis=0) / jnp.sqrt(num_basis)
    return z



@partial(jax.jit, static_argnames=["num_basis", "seq_len", "N", "same_direction_for_all_freqs"])
def _rff_z_matrix(pos, wave_num, phi_f, basis_directions, num_basis, seq_len, N, same_direction_for_all_freqs):

    Z = _get_rff_basis_vec(pos, basis_directions, wave_num, same_direction_for_all_freqs)

    Z = Z * phi_f[:,:,None]
    Z = ft.dft_domain_to_real_vec(Z, even=True, scale=True)  # (seq_len, N, num_basis)
    Z = jnp.moveaxis(Z, 0, 1) # (N, seq_len, num_basis)
    Z = jnp.reshape(Z, (N, seq_len* num_basis)) # (N, seq_len * num_basis)
    return Z




@partial(jax.jit, static_argnames=["num_basis", "same_direction_for_all_freqs", "score"])
def cross_validation_krr_moving_mic_rff(data, pos, sequence, samplerate, c, reg_params, num_basis=64, key=None, direction = None, beta = None, same_direction_for_all_freqs=True, score="gcv"):
    """Computes the GCV score for sound field estimation with random Fourier features

    IMPORTANT: The same key must be used as in the later estimation step for the cross validation results to be valid

    Parameters
    ----------
    data : np.ndarray of shape (num_samples)
        the time-domain sound pressure measurements samples
    pos : np.ndarray of shape (num_samples, 3)
        Positions of the measurement points.
    reg_param : float
        regularization parameter to evaluate the GCV score for.

    score : str in {'gcv'}
        decides which type of score to use to determine the best regularization parameter.
        gcv is generalized cross-validation

    Returns
    -------
    best_reg_param : float
        regularization parameter that minimizes the chosen score
    score_values : ndarray of shape (len(reg_params),)
        score values for each regularization parameter

    References
    ----------
    Time-domain sound field estimation using kernel ridge regression, J Brunnström, M. B. Møller, J Østergaard, S Koyama, T van Waterschoot, M Moonen, 2025
    """
    data, pos, _, sequence, N, seq_len, num_periods = _parse_moving_mic_args(data, pos, jnp.zeros((1,3)), sequence)

    if key is None:
        key = jax.random.key(23456743)

    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)
    num_real_freqs = wave_num.shape[-1]
    phi_f = _seq_stft_krr_multiperiod(sequence, num_periods)

    if same_direction_for_all_freqs:
        tot_num_basis = num_basis
    else:
        tot_num_basis = num_basis * num_real_freqs

    if direction is None:
        assert beta is None, "beta should not be provided if direction is not provided"
        basis_directions = mc.uniform_random_on_sphere(tot_num_basis, key)
    else:
        assert beta is not None, "beta should be provided if direction is provided"
        basis_directions = mc.vonmises_fisher_on_sphere(tot_num_basis, -direction, beta, key)
    if not same_direction_for_all_freqs:
        basis_directions = basis_directions.reshape((num_real_freqs, num_basis, 3))

    Z = _rff_z_matrix(-pos, wave_num, phi_f, basis_directions, num_basis, seq_len, N, same_direction_for_all_freqs)

    def compute_score(reg):
        if score == "gcv":
            score_value = ls.gcv_score(Z, data, reg)
        else:
            raise ValueError("unrecognized score parameter")
        return score_value
    
    score_values = jax.lax.map(compute_score, reg_params, batch_size=1)  
    best_index = jnp.argmin(score_values)
    best_reg_param = reg_params[best_index]

    return best_reg_param, score_values