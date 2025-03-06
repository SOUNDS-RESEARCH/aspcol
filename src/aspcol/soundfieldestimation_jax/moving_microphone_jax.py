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
#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_debug_nans", True)

import aspcore.fouriertransform as ft
import aspcol.sphericalharmonics as shd_numpy

import aspcol.sphericalharmonics_jax as shd


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
    if p.ndim >= 2:
        p = np.squeeze(p)
    N = p.shape[0]

    if sequence.ndim == 2:
        sequence = np.squeeze(sequence, axis=0)
    assert sequence.ndim == 1
    seq_len = sequence.shape[0]
    assert seq_len % 2 == 0 #Calculations later assume seq_len is even to get the Nyquist frequency
    num_periods = N // seq_len
    assert N % seq_len == 0

    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)
    num_real_freqs = wave_num.shape[-1]
    #len(ft.get_real_freqs(seq_len, samplerate))

    assert pos.shape == (N, 3)
    assert pos_eval.ndim == 2 and pos_eval.shape[1] == 3

    if dir_coeffs.ndim == 2:
        dir_coeffs = dir_coeffs[None,:,:] #add a dimension for the number of frequencies
    assert dir_coeffs.ndim == 3
    assert dir_coeffs.shape[1] == N or dir_coeffs.shape[1] == 1
    assert dir_coeffs.shape[0] == num_real_freqs or dir_coeffs.shape[0] == 1
    
    # ======= Estimation of spherical harmonic coefficients =======
    Phi = _sequence_stft_multiperiod(sequence, num_periods)
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



def _sequence_stft_multiperiod(sequence, num_periods):
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
    Phi = _sequence_stft(sequence)
    return np.tile(Phi, (1, num_periods))

def _sequence_stft(sequence):
    """Might not correspond to the definition in the paper

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
        seq_vec = np.roll(sequence, -n) #so that n is the first element
        seq_vec = np.roll(seq_vec, -1) # so that n ends up last
        seq_vec = np.flip(seq_vec) # so that we get n first and then n-i as we move later in the vector
        for f in range(B):
            exp_vec = ft.idft_vector(f, B)
            Phi[f,n] = np.sum(exp_vec * seq_vec) 
    # Fast version, not sure if correct
    # for n in range(B):
    #     Phi[:,n] = np.fft.fft(np.roll(sequence, -n), axis=-1)
    return Phi
