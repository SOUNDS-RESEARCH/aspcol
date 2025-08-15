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
import aspcol.sphericalharmonics as shd_numpy

import aspcore.fouriertransform_jax as ft
import aspcore.matrices_jax as aspmat
import aspcore.montecarlo_jax as mc
import aspcol.sphericalharmonics_jax as shd
import aspcol.planewaves_jax as pw

import aspcol.kernelinterpolation_jax.kernel_jax as kernel


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
    
    psi = calculate_psi(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs)
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

def _estimate_from_regressor_compiled(regressor, pos, pos_eval, wave_num, dir_omni, dir_coeffs, gaunt_set):

    def psi_scan_loop(carry, arg_slice):
        (pos_i, dir_coeffs_i) = arg_slice

        def psi_scan_inner_loop(carry_inner, arg_slice_inner):
            (pos_j, dir_coeffs_j) = arg_slice_inner

            pos_diff = pos_i[None,:] - pos_j[None,:]

            T = shd.translation_operator(pos_diff, wave_num, gaunt_set)
            inner_product = jnp.moveaxis(jnp.conj(dir_coeffs_i)[None,None,:,None], -1, -2) @ T @ dir_coeffs_j[None,None,:,None]
            inner_product = jnp.squeeze(inner_product)

            psi_ij = 2 * jnp.sum(jnp.real(inner_product[1:-1,...]))
            psi_ij = psi_ij + jnp.real(inner_product[0,...]) + jnp.real(inner_product[-1,...])

            return carry_inner, psi_ij
        
        _, psi_i = jax.lax.scan(psi_scan_inner_loop, 0, (pos, dir_coeffs))        
        return (carry, psi_i)
    
    _, psi = jax.lax.scan(psi_scan_loop, 0, (pos_eval, dir_omni))


    kernel_val = shd_numpy.translated_inner_product(pos_eval, pos, dir_omni, dir_coeffs, wave_num)
    est_sound_pressure = np.squeeze(kernel_val @ regressor[:,:,None], axis=-1)
    return est_sound_pressure

def _calculate_psi_compiled_each_freq(pos, dir_coeffs, k, Phi, seq_len, num_real_freqs, gaunt_set):
    N = pos.shape[0]
    psi = np.zeros((N, N), dtype = float)

    for f in range(1, num_real_freqs-1):
        print(f"Frequency {f}, going from 1 to {num_real_freqs-2} (inclusive)")
        phi_rank1_matrix = Phi[f,:,None] * Phi[f,None,:].conj()

        psi_f = np.squeeze(shd.translated_inner_product(pos, dir_coeffs, k[f:f+1], gaunt_set), axis=0) * phi_rank1_matrix
        psi += 2 * np.real(psi_f)

    # no conjugation required for zeroth frequency and the Nyquist frequency, 
    # since they will be real already for a real input sequence
    psi += np.squeeze(np.real_if_close(shd.translated_inner_product(pos, dir_coeffs, k[0:1], gaunt_set)), axis=0) * np.real_if_close(Phi[0,:,None] * Phi[0,None,:])
    psi += np.squeeze(np.real(shd.translated_inner_product(pos, dir_coeffs, k[seq_len//2:seq_len//2+1], gaunt_set)), axis=0) * np.real_if_close(Phi[seq_len//2,:,None] * Phi[seq_len//2,None,:])
    return psi


def calculate_psi(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs):
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

    print("Running compiled Psi calculation")
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

@jax.jit
def _calculate_psi_compiled_single_loop_in_scan(pos, dir_coeffs, wave_num, Phi, gaunt_set):
    """
    is slow to compile, intermediate version between _calculate_psi_compiled_with_python_loops and _calculate_psi_compiled
    """
    num_pos = pos.shape[0]

    Phi = Phi.T # we need to scan over leading axis
    dir_coeffs = dir_coeffs[0,:,:] # assume frequency independent directivity

    def psi_scan_loop(carry, arg_slice):
        (pos_i, phi_i, dir_coeffs_i) = arg_slice

        psi_vals = []
        for j in range(num_pos):
            pos_diff = pos_i[None,:] - pos[j:j+1,:]
            phi_factor = phi_i * jnp.conj(Phi[j,:])

            T = shd.translation_operator(pos_diff, wave_num, gaunt_set)
            inner_product = jnp.moveaxis(jnp.conj(dir_coeffs_i)[None,None,:,None], -1, -2) @ T @ dir_coeffs[None,j:j+1,:,None]
            inner_product = jnp.squeeze(inner_product) * phi_factor

            psi_val = 2 * jnp.sum(jnp.real(inner_product[1:-1,...]))
            psi_val = psi_val + jnp.real(inner_product[0,...]) + jnp.real(inner_product[-1,...])
            psi_vals.append(psi_val)
        psi_i = jnp.stack(psi_vals, axis=0)
            
        carry = carry + 1
        return (carry, psi_i)
    
    carry, psi = jax.lax.scan(psi_scan_loop, 0, (pos, Phi, dir_coeffs))
    return psi


@jax.jit
def _calculate_psi_compiled_with_python_loops(pos, dir_coeffs, wave_num, Phi, gaunt_set):
    """
    is really slow to compile, so should not be used 
    """
    num_pos = pos.shape[0]

    psi_vals = []
    for i in range(num_pos):
        for j in range(num_pos):
            pos_diff = pos[i:i+1,:] - pos[j:j+1,:]
            phi_factor = Phi[:,i] * jnp.conj(Phi[:,j])

            T = shd.translation_operator(pos_diff, wave_num, gaunt_set)
            inner_product = jnp.moveaxis(jnp.conj(dir_coeffs)[:,i:i+1,:,None], -1, -2) @ T @ dir_coeffs[:,j:j+1,:,None]
            inner_product = jnp.squeeze(inner_product) * phi_factor

            psi_val = 2 * jnp.sum(jnp.real(inner_product[1:-1,...]))
            psi_val = psi_val + jnp.real(inner_product[0,...]) + jnp.real(inner_product[-1,...])
            psi_vals.append(psi_val)

    psi = jnp.stack(psi_vals, axis=0)
    psi = jnp.reshape(psi, (num_pos, num_pos))
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

    est_sound_pressure = reconstruct_directional(krr_params, pos_eval, pos, wave_num, direction, beta, batch_size=batch_size)
    if return_params:
        return est_sound_pressure, krr_params, K
    return est_sound_pressure

#@partial(jax.jit, static_argnames=["seq_len", "batch_size"])
def _calc_directional_kernel_mat_attempted_optimization(pos, wave_num, phi_f, direction, beta, seq_len, batch_size=8):
    dft_weighting = ft.rdft_weighting(seq_len)

    assert direction.ndim == 1 or direction.ndim == 2
    if direction.ndim == 2:
        assert direction.shape[0] == 1
        direction = direction[0,:]

    pos_diff = pos[:,None,:] - pos[None,:,:] # shape (N, N, 3)
    pos_diff = pos_diff.reshape((pos.shape[0]**2, 3)) #such that we can chunk it

    phi_f_all = phi_f[:,:,None].conj() * phi_f[:,None,:] # shape (num_real_freqs, N, N)
    phi_f_all = phi_f_all.reshape((phi_f_all.shape[0], -1)).T # shape (N*N, num_real_freqs, )

    angle_term = -1j * beta * direction #(3,) #inverts direction to match kernel definition

    CHUNK_SIZE = 1024

    def loop1(xs):
        pos_diff_single, phi_f_single = xs
        pos_term = wave_num[:,None] * pos_diff_single[None,:] # shape (num_real_freqs, 3)
        kernel_val = jnp.sinc(jnp.sqrt(jnp.sum((angle_term - pos_term)**2, axis=-1)) / jnp.pi)

        K_val = jnp.sum(dft_weighting * jnp.real(phi_f_single * kernel_val))
        return K_val

    K = jax.lax.map(loop1, (pos_diff, phi_f_all), batch_size=batch_size)
    K = K.reshape((pos.shape[0], pos.shape[0])) # shape (N, N)

    # def _kernel_inner_loop(system_mat, scanned_args):   
    #     (wave_num_single, phi_single, dft_weight) = scanned_args
    #     phi_rank1_matrix = phi_single[:,None].conj() * phi_single[None,:]

    #     pos_term = wave_num_single * pos_diff
    #     kernel_val = jnp.sinc(jnp.sqrt(jnp.sum((angle_term - pos_term)**2, axis=-1)) / jnp.pi)
    #     system_mat_incr = dft_weight * jnp.real(kernel_val * phi_rank1_matrix)

    #     system_mat = system_mat + system_mat_incr
    #     #system_mat_incr = dft_weight * jnp.real(jnp.squeeze(kernel.directional_kernel_vonmises(pos, pos, wave_num_single, -direction, beta)) * phi_rank1_matrix)
    #     #system_mat = system_mat + system_mat_incr
    #     return system_mat, system_mat
    

    # kernel_system_mat = jnp.zeros((pos.shape[0], pos.shape[0]), dtype=float)
    # kernel_system_mat, _ = jax.lax.scan(_kernel_inner_loop, kernel_system_mat, (wave_num, phi_f, dft_weighting), unroll=batch_size)
    return K


    # angle_term = 1j * beta * direction.reshape((1,-1,1,1,direction.shape[-1]))
    # pos_term = wave_num.reshape((-1,1,1,1,1)) * (pos1.reshape((1,1,-1,1,pos1.shape[-1])) - pos2.reshape((1,1,1,-1,pos2.shape[-1])))
    # return jnp.sinc(jnp.sqrt(jnp.sum((angle_term - pos_term)**2, axis=-1)) / jnp.pi)

@partial(jax.jit, static_argnames=["seq_len", "batch_size"])
def _calc_directional_kernel_mat(pos, wave_num, phi_f, direction, beta, seq_len, batch_size=8):
    dft_weighting = ft.rdft_weighting(seq_len)

    def _kernel_inner_loop(system_mat, scanned_args):
        (wave_num_single, phi_single, dft_weight) = scanned_args
        phi_rank1_matrix = phi_single[:,None].conj() * phi_single[None,:]
        system_mat_incr = dft_weight * jnp.real(jnp.squeeze(kernel.directional_kernel_vonmises(pos, pos, wave_num_single, -direction, beta)) * phi_rank1_matrix)
        system_mat = system_mat + system_mat_incr
        return system_mat, system_mat

    kernel_system_mat = jnp.zeros((pos.shape[0], pos.shape[0]), dtype=float)
    kernel_system_mat, _ = jax.lax.scan(_kernel_inner_loop, kernel_system_mat, (wave_num, phi_f, dft_weighting), unroll=batch_size)
    return kernel_system_mat


def reconstruct_directional(krr_params, pos_eval, pos_mic, wave_num, direction, beta, batch_size=8):
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
        estimated RIR per frequency at the evaluation points"""
    def _reconstruct_inner_loop(pos_eval_single):
        kernel_val = kernel.directional_kernel_vonmises(pos_eval_single[None,:], pos_mic, wave_num, -direction, beta).astype(complex)
        kernel_val = jnp.squeeze(kernel_val, axis=1) # remove extra axis corresponding to the number of directions
        p_est = jnp.sum(kernel_val * krr_params[:,None,:], axis=-1)
        return jnp.squeeze(p_est, axis=-1)
    
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
    
    """
    p, pos, pos_eval, sequence, N, seq_len, num_periods = _parse_moving_mic_args(p, pos, pos_eval, sequence)
    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)

    phi_f = _seq_stft_krr_multiperiod(sequence, num_periods)
    K = _calc_diffuse_kernel_mat(pos, wave_num, phi_f, seq_len)

    reg_matrix = seq_len * reg_param * jnp.eye(N)
    krr_params = jax_splin.solve(K + reg_matrix, p, assume_a="pos")
    krr_params = phi_f * krr_params[None,:]

    est_sound_pressure = reconstruct_diffuse(krr_params, pos_eval, pos, wave_num)
    if return_params:
        return est_sound_pressure, krr_params, K
    return est_sound_pressure


@partial(jax.jit, static_argnames=["seq_len", "batch_size"])
def _calc_diffuse_kernel_mat(pos, wave_num, Phi, seq_len, batch_size=8):
    num_real_freqs = wave_num.shape[-1]
    dft_weighting = ft.rdft_weighting(seq_len)

    def _kernel_inner_loop(system_mat, scanned_args):
        (wave_num_single, phi_single, dft_weight) = scanned_args
        phi_rank1_matrix = phi_single[:,None].conj() * phi_single[None,:] #Phi[f,:,None] * Phi[f,None,:].conj()
        system_mat_incr = dft_weight * jnp.real(jnp.squeeze(kernel.diffuse_kernel(pos, pos, wave_num_single)) * phi_rank1_matrix)
        system_mat = system_mat + system_mat_incr
        return system_mat, system_mat

    kernel_system_mat = jnp.zeros((pos.shape[0], pos.shape[0]), dtype=float)
    kernel_system_mat, _ = jax.lax.scan(_kernel_inner_loop, kernel_system_mat, (wave_num, Phi, dft_weighting), unroll=batch_size)
    return kernel_system_mat

@partial(jax.jit, static_argnames=["batch_size"])
def reconstruct_diffuse(krr_params, pos_eval, pos_mic, wave_num, batch_size=8):
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
        kernel_val = kernel.diffuse_kernel(pos_eval_single[None,:], pos_mic, wave_num).astype(complex)
        p_est = jnp.sum(kernel_val * krr_params[:,None,:], axis=-1)
        return jnp.squeeze(p_est, axis=-1)
    
    estimate = jnp.moveaxis(jax.lax.map(_reconstruct_inner_loop, pos_eval, batch_size=batch_size), 0, 1)
    return estimate




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


@partial(jax.jit, static_argnames=["num_periods"])
def _seq_stft_krr_multiperiod_OLD(sequence, num_periods):
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
    Phi = _seq_stft_krr_OLD(sequence)
    return jnp.tile(Phi, (1, num_periods))

def _seq_stft_krr_OLD(sequence):
    """
    Assumes the sequence is periodic with period B
    Implements by calculating the Bayesian version and then scaling and conjugating it. 

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
        return ft.rfft(jnp.roll(sequence, -n)) / B
    phis = jax.vmap(inner_func, out_axes=1)(jnp.arange(B))
    return phis.conj() * B


@partial(jax.jit, static_argnames=["num_basis", "return_params"])
def krr_moving_mic_rff(p, pos, pos_eval, sequence, samplerate, c, reg_param, num_basis=64, seed=None, return_params=False, direction = None, beta = None):
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

    Notes
    -----
    Uses the diffuse kernel implicitly
    the reg_param is not identical to the standard KRR methods, and so might have to be chosen differently
    The params returns by return_params can be interpreted as plane wave coefficients, but are
    not the same as the standard krr parameters

    """
    p, pos, pos_eval, sequence, N, seq_len, num_periods = _parse_moving_mic_args(p, pos, pos_eval, sequence)

    if seed is None:
        seed = 123456
    key = jax.random.PRNGKey(seed)

    #p_est, params, Z = _rff_compilable_subproblem(
    #    p, pos, pos_eval, sequence, samplerate, c, reg_param, num_basis, seq_len, N, num_periods, key)

    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)
    num_real_freqs = wave_num.shape[-1]
    phi_f = _seq_stft_krr_multiperiod(sequence, num_periods)

    if direction is None:
        assert beta is None, "beta should not be provided if direction is not provided"
        basis_directions = mc.uniform_random_on_sphere(num_basis*num_real_freqs, key)
    else:
        assert beta is not None, "beta should be provided if direction is provided"
        basis_directions = mc.vonmises_fisher_on_sphere(num_basis*num_real_freqs, direction, beta, key)
    basis_directions = basis_directions.reshape((num_real_freqs, num_basis, 3))

    Z = _rff_z_matrix(-pos, wave_num, phi_f, basis_directions, num_basis, seq_len, N)

    system_mat = Z.T @ Z
    system_mat = system_mat + seq_len * reg_param * jnp.eye(seq_len * num_basis, dtype=Z.dtype)
    projected_data = Z.T @ p

    params = jax_splin.solve(system_mat, projected_data, assume_a="pos")
    params = params.reshape(seq_len, num_basis)
    params = ft.real_vec_to_dft_domain(params, scale=True) # (num_real_freqs, num_basis)

    z_eval = jnp.stack([pw.plane_wave(pos_eval, basis_directions[f,:,:], wave_num[f]) for f in range(num_real_freqs)], axis=0) / jnp.sqrt(num_basis)
    z_eval = jnp.moveaxis(z_eval, 0, 1) # (num_eval, num_real_freqs, num_basis)

    p_est = jnp.sum(z_eval * params[None,:,:], axis=-1).T #(num_eval, seq_len)

    if return_params:
        return p_est, params, Z
    return p_est # (num_real_freqs, num_eval)

# @partial(jax.jit, static_argnames=["num_basis", "seq_len", "num_periods", "N"])
# def _rff_compilable_subproblem(p, pos, pos_eval, sequence, samplerate, c, reg_param, num_basis, seq_len, N, num_periods, key):
#     wave_num = ft.get_real_wavenum(seq_len, samplerate, c)
#     num_real_freqs = wave_num.shape[-1]
#     phi_f = _seq_stft_krr_multiperiod(sequence, num_periods)

#     basis_directions = mc.uniform_random_on_sphere(num_basis*num_real_freqs, key).reshape((num_real_freqs, num_basis, 3))

#     Z = _rff_z_matrix(-pos, wave_num, phi_f, basis_directions, num_basis, seq_len, N)

#     system_mat = Z.T @ Z
#     system_mat = system_mat + seq_len * reg_param * jnp.eye(seq_len * num_basis, dtype=Z.dtype)
#     projected_data = Z.T @ p

#     params = jax_splin.solve(system_mat, projected_data, assume_a="pos")
#     #params = jnp.linalg.solve(system_mat, projected_data)
#     params = params.reshape(seq_len, num_basis)
#     params = ft.real_vec_to_dft_domain(params, scale=True) # (num_real_freqs, num_basis)

#     z_eval = jnp.stack([pw.plane_wave(pos_eval, basis_directions[f,:,:], wave_num[f]) for f in range(num_real_freqs)], axis=0) / jnp.sqrt(num_basis)
#     z_eval = jnp.moveaxis(z_eval, 0, 1) # (num_eval, num_real_freqs, num_basis)

#     p_est = jnp.sum(z_eval * params[None,:,:], axis=-1).T #(num_eval, seq_len)
#     return p_est, params, Z

@partial(jax.jit, static_argnames=["num_basis", "seq_len", "N"])
def _rff_z_matrix(pos, wave_num, phi_f, basis_directions, num_basis, seq_len, N):
    num_real_freqs = wave_num.shape[-1]
    Z = jnp.stack([pw.plane_wave(pos, basis_directions[f,:,:], wave_num[f]) for f in range(num_real_freqs)], axis=0) / jnp.sqrt(num_basis)

    Z = Z * phi_f[:,:,None]
    Z = ft.dft_domain_to_real_vec(Z, even=True, scale=True)  # (seq_len, N, num_basis)
    Z = jnp.moveaxis(Z, 0, 1) # (N, seq_len, num_basis)
    Z = jnp.reshape(Z, (N, seq_len* num_basis)) # (N, seq_len * num_basis)
    return Z