import aspcol.filterdesign as fd
import numpy as np
import scipy.linalg as splin
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_debug_nans", True)
from functools import partial

import aspcol.soundfieldestimation as sfe
import aspcol.sphericalharmonics as shd
import aspcol.fouriertransform as ft



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

    wave_num = ft.get_wavenum(seq_len, samplerate, c)
    num_real_freqs = len(ft.get_real_freqs(seq_len, samplerate))

    assert pos.shape == (N, 3)
    assert pos_eval.ndim == 2 and pos_eval.shape[1] == 3

    if dir_coeffs.ndim == 2:
        dir_coeffs = dir_coeffs[None,:,:] #add a dimension for the number of frequencies
    assert dir_coeffs.ndim == 3
    assert dir_coeffs.shape[1] == N or dir_coeffs.shape[1] == 1
    assert dir_coeffs.shape[0] == num_real_freqs or dir_coeffs.shape[0] == 1
    

    # ======= Estimation of spherical harmonic coefficients =======
    Phi = sfe._sequence_stft_multiperiod(sequence, num_periods)

    #dist_mat = np.sqrt(np.sum((np.expand_dims(pos,1) - np.expand_dims(pos,0))**2, axis=-1))
    
    psi = calculate_psi(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs)
    noise_cov = reg_param * np.eye(N)
    regressor = splin.solve(psi + noise_cov, p, assume_a = "pos")
    regressor = Phi.conj()[:num_real_freqs,:] * regressor[None,:]

    # ======= Reconstruction of RIR =======
    est_sound_pressure = estimate_from_regressor(regressor, pos, pos_eval, wave_num[:num_real_freqs], dir_coeffs)
    # num_eval = pos_eval.shape[0]
    # dir_omni = shd.directivity_omni() * np.ones((num_eval, 1))
    # dir_omni = dir_omni[None,:,:] # add a dimension for the number of frequencies

    # kernel_val = shd.translated_inner_product(pos_eval, pos, dir_omni, dir_coeffs, wave_num[:num_real_freqs]) #ki.kernel_helmholtz_3d(pos_eval, pos, k[f:f+1]).astype(complex)[0,:,:]
    # est_sound_pressure = np.squeeze(kernel_val @ regressor[:,:,None], axis=-1) #np.sum(kernel_val * regressor[:,None,:], axis=-1)
    
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
    regressor : ndarray of shape (num_freqs, N)
        The regressor calculated by inf_dimensional_shd_dynamic. 
        Represents Phi* v in eq (31) from [brunnstromBayesian2024]
    pos : ndarray of shape (N, 3)
        position of the trajectory for each sample
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    wave_num : ndarray of shape (num_real_freq, )
        wavenumber, defined as w / c where w is the angular frequency
    dir_coeffs : ndarray of shape (N, num_coeffs), optional
        harmonic coefficients of microphone directivity, 
        Note that a higher number of coefficients will drastically increase the computational cost
        If not provided, the microphones are assumed to be omnidirectional.

    Returns
    -------
    shd_coeffs : ndarray of shape (num_real_freqs, num_eval)
        time-domain harmonic coefficients of the estimated sound field

    Notes
    -----
    Assumptions:
    The data is measured over an integer number of periods of the sequence
    N = seq_len * M, where M is the number of periods that was measured
    The length of sequence is the length of the estimated RIR
    """
    # ======= Argument parsing and constants =======
    #if sequence.ndim == 2:
    #    sequence = np.squeeze(sequence, axis=0)
    #assert sequence.ndim == 1
    #seq_len = sequence.shape[0]
    #assert seq_len % 2 == 0 #Calculations later assume seq_len is even to get the Nyquist frequency

    N = pos.shape[0]
    #assert N % seq_len == 0
    assert pos.shape == (N, 3)
    assert pos_eval.ndim == 2 and pos_eval.shape[1] == 3

    num_real_freqs = wave_num.shape[-1]

    if dir_coeffs is None:
        dir_coeffs = shd.directivity_omni() * np.ones((N, 1))
    if dir_coeffs.ndim == 2:
        dir_coeffs = dir_coeffs[None,:,:] #add a dimension for the number of frequencies
    assert dir_coeffs.ndim == 3
    assert dir_coeffs.shape[1] == N or dir_coeffs.shape[1] == 1
    assert dir_coeffs.shape[0] == num_real_freqs or dir_coeffs.shape[0] == 1

    # ======= Reconstruction of RIR =======
    num_eval = pos_eval.shape[0]
    dir_omni = shd.directivity_omni() #* np.ones((num_eval, 1))
    dir_omni = dir_omni[None,:,:] # add a dimension for the number of frequencies

    # kernel_val = shd.translated_inner_product(pos_eval, pos, dir_omni, dir_coeffs, wave_num)
    # est_sound_pressure = np.squeeze(kernel_val @ regressor[:,:,None], axis=-1)

    num_eval = pos_eval.shape[0]
    est_sound_pressure = np.zeros(((num_real_freqs, num_eval)), dtype=complex)
    for i in range(num_eval):
        kernel_val = shd.translated_inner_product(pos_eval[i:i+1,:], pos, dir_omni, dir_coeffs, wave_num)
        est_sound_pressure[:,i:i+1] = np.squeeze(kernel_val @ regressor[:,:,None], axis=-1)
    return est_sound_pressure



def calculate_psi(pos, dir_coeffs, k, Phi, seq_len, num_real_freqs):
    N = pos.shape[0]
    psi = np.zeros((N, N), dtype = float)

    print(f"starting TIP")
    # tip = shd.translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, k[1:num_real_freqs-1])
    # phi_rank1_matrix = Phi[1:num_real_freqs-1,:,None] * Phi[1:num_real_freqs-1,None,:].conj()
    # psi = 2 * np.sum(np.real(tip * phi_rank1_matrix), axis=0)

    for f in range(1, num_real_freqs-1):
        print(f"Frequency {f}, going from 1 to {num_real_freqs-2} (inclusive)")
        phi_rank1_matrix = Phi[f,:,None] * Phi[f,None,:].conj()

        psi_f = np.squeeze(shd.translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, k[f:f+1]), axis=0) * phi_rank1_matrix
        psi += 2 * np.real(psi_f)

    # no conjugation required for zeroth frequency and the Nyquist frequency, 
    # since they will be real already for a real input sequence
    psi += np.squeeze(np.real(shd.translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, k[0:1])), axis=0) * np.real_if_close(Phi[0,:,None] * Phi[0,None,:])
    psi += np.squeeze(np.real(shd.translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, k[seq_len//2:seq_len//2+1])), axis=0) * np.real_if_close(Phi[seq_len//2,:,None] * Phi[seq_len//2,None,:])
    return psi


def inf_dimensional_shd_dynamic_compiled(p, pos, pos_eval, sequence, samplerate, c, reg_param, dir_coeffs, verbose=False):
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
    Phi = sfe._sequence_stft_multiperiod(sequence, num_periods)
    Phi = Phi[:num_real_freqs,:]
    
    psi = calculate_psi_compiled(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs)
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

    # ======= Reconstruction of RIR =======
    num_eval = pos_eval.shape[0]

    if verbose:
        print(f"Computing eval estimates from regressor")
    est_sound_pressure = estimate_from_regressor_compiled(regressor, pos, pos_eval, wave_num, dir_coeffs)
    # dir_omni = shd.directivity_omni() * np.ones((num_eval, 1))
    # dir_omni = dir_omni[None,:,:] # add a dimension for the number of frequencies

    # kernel_val = shd.translated_inner_product(pos_eval, pos, dir_omni, dir_coeffs, wave_num) #ki.kernel_helmholtz_3d(pos_eval, pos, k[f:f+1]).astype(complex)[0,:,:]
    # est_sound_pressure = np.squeeze(kernel_val @ regressor[:,:,None], axis=-1) #np.sum(kernel_val * regressor[:,None,:], axis=-1)

    if verbose:
        return est_sound_pressure, regressor, psi
    return est_sound_pressure


def estimate_from_regressor_compiled(regressor, pos, pos_eval, wave_num, dir_coeffs = None):
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
        dir_coeffs = shd.directivity_omni() * np.ones((N, 1))
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
    dir_omni = shd.directivity_omni() #* np.ones((num_eval, 1))
    dir_omni = dir_omni[None,:,:] # add a dimension for the number of frequencies

    num_eval = pos_eval.shape[0]
    est_sound_pressure = np.zeros(((num_real_freqs, num_eval)), dtype=complex)
    for i in range(num_eval):
        #if i % 10 == 0:
        print(f"Estimating sound pressure at position {i} of {num_eval}")
        kernel_val = shd.translated_inner_product(pos_eval[i:i+1,:], pos, dir_omni, dir_coeffs, wave_num)
        est_sound_pressure[:,i:i+1] = np.squeeze(kernel_val @ regressor[:,:,None], axis=-1)

    # MAKE A TRANSLATION OPERATOR THAT WORKS FOR MIXED 1TH AND 2ND ORDERS. THEN IMPLEMENT FOLLOWING
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

            T = translation_operator(pos_diff, wave_num, gaunt_set)
            inner_product = jnp.moveaxis(jnp.conj(dir_coeffs_i)[None,None,:,None], -1, -2) @ T @ dir_coeffs_j[None,None,:,None]
            inner_product = jnp.squeeze(inner_product)

            psi_ij = 2 * jnp.sum(jnp.real(inner_product[1:-1,...]))
            psi_ij = psi_ij + jnp.real(inner_product[0,...]) + jnp.real(inner_product[-1,...])

            return carry_inner, psi_ij
        
        _, psi_i = jax.lax.scan(psi_scan_inner_loop, 0, (pos, dir_coeffs))        
        return (carry, psi_i)
    
    _, psi = jax.lax.scan(psi_scan_loop, 0, (pos_eval, dir_omni))


    kernel_val = shd.translated_inner_product(pos_eval, pos, dir_omni, dir_coeffs, wave_num)
    est_sound_pressure = np.squeeze(kernel_val @ regressor[:,:,None], axis=-1)
    return est_sound_pressure

def _calculate_psi_compiled_each_freq(pos, dir_coeffs, k, Phi, seq_len, num_real_freqs, gaunt_set):
    N = pos.shape[0]
    psi = np.zeros((N, N), dtype = float)

    for f in range(1, num_real_freqs-1):
        print(f"Frequency {f}, going from 1 to {num_real_freqs-2} (inclusive)")
        phi_rank1_matrix = Phi[f,:,None] * Phi[f,None,:].conj()

        psi_f = np.squeeze(translated_inner_product(pos, dir_coeffs, k[f:f+1], gaunt_set), axis=0) * phi_rank1_matrix
        psi += 2 * np.real(psi_f)

    # no conjugation required for zeroth frequency and the Nyquist frequency, 
    # since they will be real already for a real input sequence
    psi += np.squeeze(np.real_if_close(translated_inner_product(pos, dir_coeffs, k[0:1], gaunt_set)), axis=0) * np.real_if_close(Phi[0,:,None] * Phi[0,None,:])
    psi += np.squeeze(np.real(translated_inner_product(pos, dir_coeffs, k[seq_len//2:seq_len//2+1], gaunt_set)), axis=0) * np.real_if_close(Phi[seq_len//2,:,None] * Phi[seq_len//2,None,:])
    return psi


def calculate_psi_compiled(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs):
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
    gaunt_set = _calculate_gaunt_set(max_order, max_order)

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

            T = translation_operator(pos_diff, wave_num, gaunt_set)
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

            T = translation_operator(pos_diff, wave_num, gaunt_set)
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

            T = translation_operator(pos_diff, wave_num, gaunt_set)
            inner_product = jnp.moveaxis(jnp.conj(dir_coeffs)[:,i:i+1,:,None], -1, -2) @ T @ dir_coeffs[:,j:j+1,:,None]
            inner_product = jnp.squeeze(inner_product) * phi_factor

            psi_val = 2 * jnp.sum(jnp.real(inner_product[1:-1,...]))
            psi_val = psi_val + jnp.real(inner_product[0,...]) + jnp.real(inner_product[-1,...])
            psi_vals.append(psi_val)

    psi = jnp.stack(psi_vals, axis=0)
    psi = jnp.reshape(psi, (num_pos, num_pos))
    return psi


@jax.jit
def translated_inner_product(pos, dir_coeffs, wave_num, gaunt_set):
    """
    
    <T(r_1 - r_2, omega_m) gamma_2(omega_m), gamma_1(omega_m)>
    
    Parameters
    ----------
    pos1 : ndarray of shape (num_pos1, 3)
        positions of the first set of measurement points
    pos2 : ndarray of shape (num_pos2, 3)
        positions of the second set of measurement points
    dir_coeffs1 : ndarray of shape (num_freqs, num_coeffs1) or (num_freqs, num_pos1, num_coeffs1)
        coefficients of the directivity function for the first set of measurement points
    dir_coeffs2 : ndarray of shape (num_freqs, num_coeffs2) or (num_freqs, num_pos1, num_coeffs1)
        coefficients of the directivity function for the second set of mea surement points
    wave_num : ndarray of shape (num_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.

    Returns
    -------
    psi : ndarray of shape (num_freqs, num_pos1, num_pos2)
        inner product of the translated directivity functions
    """
    assert pos.ndim == 2
    assert pos.shape[-1] == 3
    num_pos = pos.shape[0]

    assert wave_num.ndim == 1
    num_freqs = wave_num.shape[0]
    
    if dir_coeffs.ndim == 2:
        dir_coeffs = dir_coeffs[:,None,:]
    assert dir_coeffs.ndim == 3
    assert dir_coeffs.shape[1] == num_pos or dir_coeffs.shape[1] == 1
    assert dir_coeffs.shape[0] == num_freqs or dir_coeffs.shape[0] == 1

    pos_diff = pos[:,None,:] - pos[None,:,:]

    translated_coeffs2 = jax.vmap(translate_shd_coeffs, in_axes=(None, 0, None, None), out_axes=1)(dir_coeffs, pos_diff, wave_num, gaunt_set)
    

    #translated_coeffs2 = jnp.stack([translate_shd_coeffs(dir_coeffs, pos_diff[m,:,:], wave_num, gaunt_set) for m in range(num_pos)], axis=1)
    inner_product_matrix = jnp.sum(translated_coeffs2 * dir_coeffs.conj()[:,:,None,:], axis=-1)
    return inner_product_matrix



@jax.jit
def translate_shd_coeffs(shd_coeffs, pos, wave_num, gaunt_set):
    """Translates the provided shd_coeffs to another expansion center
    
    shd_coeffs(pos_orig + pos) = translate(pos) shd_coeffs(pos_orig)
    
    Parameters
    ----------
    shd_coeffs : ndarray of shape (num_freqs, num_coeffs,) or (num_freqs, num_pos, num_coeffs,)
        coefficients of the sequence
    pos : ndarray of shape (num_pos, 3)
        position argument to the translation operator
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    max_order_output : int
        maximum order of the translated coefficients

    Returns
    -------
    shd_coeffs_translated : ndarray of shape (num_freqs, num_pos, num_coeffs,)
        translated coefficients
        if num_pos == 1, the returned array will have shape (num_freqs, num_coeffs,)
    """
    if shd_coeffs.ndim == 2:
        shd_coeffs = shd_coeffs[:,None,:]
    assert pos.ndim == 2
    assert pos.shape[1] == 3
    assert wave_num.ndim == 1
    assert shd_coeffs.ndim == 3
    assert shd_coeffs.shape[0] == wave_num.shape[0] or shd_coeffs.shape[0] == 1
    assert shd_coeffs.shape[1] == pos.shape[0] or shd_coeffs.shape[1] == 1
    #num_coeffs = shd_coeffs.shape[-1]

    #max_order_input = shd.shd_max_order(num_coeffs)

    T_all = translation_operator(pos, wave_num, gaunt_set)
    translated_coeffs = T_all @ shd_coeffs[...,None]

    num_pos = pos.shape[0]
    if num_pos == 1:
        translated_coeffs = jnp.squeeze(translated_coeffs, axis=1)
    return jnp.squeeze(translated_coeffs, axis=-1)

@jax.jit
def translation_operator(pos, wave_num, gaunt_set):
    """Translation operator for harmonic coefficients, such that 
    shd_coeffs(pos_orig + pos) = T(pos) @ shd_coeffs(pos_orig)

    Defined according to T = hat{S}^T, where hat{S} is the basis translation matrix defined in
    P. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles.
    This definition is consistent with the definition (6) in Ueno et al., Sound Field Recording Using
    Distributed Microphones Based on Harmonic Analysis of Infinite Order. 
    
    Parameters
    ----------
    pos : ndarray of shape (num_pos, 3)
        position argument to the translation operator
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    max_order_input : int
        maximum order of the coefficients that should be translated
    max_order_output : int
        maximum order of the translated coefficients

    Returns
    -------
    T : ndarray of shape (num_freqs, num_pos, num_coeffs_output, num_coeffs_input)
        translation operator, such that shd_coeffs(pos_orig + pos) = T(pos) @ shd_coeffs(pos_orig)
    """

    S = basis_translation_3_80(pos, wave_num, gaunt_set)
    T = jnp.moveaxis(S, -1, -2)
    return T

@jax.jit
def basis_translation_3_80_slow_compile(pos, wave_num, gaunt_set):
    """Translation operator for shd basis function, such that 
    shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)

    Implemented according to equation 3.80 in 
    P. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles.

    Hardcoded for max_order = 1 to make compilation feasible
    Currently returns nan when pos is (0,0,0)
    
    Parameters
    ----------
    pos : ndarray of shape (num_pos, 3)
        position argument to the translation operator
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    max_order_input : int
        maximum order of the coefficients that should be translated
    max_order_output : int
        maximum order of the translated coefficients

    Returns
    -------
    T : ndarray of shape (num_freqs, num_pos, num_coeffs_output, num_coeffs_input)
        translation operator, such that shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)
    """
    max_order = 1
    num_coeffs = 4
    num_pos = pos.shape[0]
    num_freq = wave_num.shape[0]

    tr_op = jnp.zeros((num_freq, num_pos, num_coeffs, num_coeffs), dtype=complex)
    orders = np.array([0, 1, 1, 1])
    degrees = np.array([0, -1, 0, 1])

    max_tot_order = 2 * max_order 
    num_coeffs_tot = 9
    orders_tot = np.array([0,  1,  1,  1,  2,  2,  2,  2,  2])
    degrees_tot = np.array([0, -1,  0,  1, -2, -1,  0,  1,  2])

    num_q = max_tot_order + 1
    all_q = np.arange(max_tot_order+1)
    q_lst = []
    for out_idx, (n, m) in enumerate(zip(orders, degrees)):
        q_lst.append([])
        for in_idx, (nu, mu) in enumerate(zip(orders, degrees)):
            q_lst[out_idx].append(np.arange(np.abs(mu-m), n+nu+1))

    harm_idx_lst = []
    for out_idx, (n, m) in enumerate(zip(orders, degrees)):
        harm_idx_lst.append([])
        for in_idx, (nu, mu) in enumerate(zip(orders, degrees)):
            harm_idx_lst[out_idx].append(shd_coeffs_order_degree_to_index(q_lst[out_idx][in_idx], mu-m))
    
    radius, azimuth, zenith = cart2spherical(pos)
    radius = radius[None, None, ...]
    azimuth = azimuth[None, ...]
    zenith = zenith[None, ...]

    orders_tot = orders_tot[:, None]
    degrees_tot = degrees_tot[:, None]

    wave_num = wave_num[:, None, None]
    all_q = all_q[None,:,None]

    bessel_shape = (num_freq, num_q, num_pos)
    bessel = spherical_jn(jnp.ravel(jnp.broadcast_to(all_q, bessel_shape)), 
                                        jnp.ravel(jnp.broadcast_to(wave_num * radius, bessel_shape))) # ordered as (num_freq, num_q, num_pos)
    bessel = jnp.reshape(bessel, bessel_shape)

    harm_shape = (num_coeffs_tot, num_pos)
    harm = jax.scipy.special.sph_harm(jnp.ravel(jnp.broadcast_to(degrees_tot, harm_shape)), 
                                        jnp.ravel(jnp.broadcast_to(orders_tot, harm_shape)), 
                                        jnp.ravel(jnp.broadcast_to(azimuth, harm_shape)), 
                                        jnp.ravel(jnp.broadcast_to(zenith, harm_shape)), 
                                        max_tot_order) # ordered as (num_coeffs_tot, num_pos)
    harm = jnp.reshape(harm, harm_shape)
    harm = jnp.conj(harm)

    for out_idx, (n, m) in enumerate(zip(orders, degrees)):
        for in_idx, (nu, mu) in enumerate(zip(orders, degrees)):
            q_array = q_lst[out_idx][in_idx]
            harm_idxs = harm_idx_lst[out_idx][in_idx]

            basis_val = bessel[:,q_array,:] * harm[None,harm_idxs, :]
            g = (1j)**q_array[None,:,None] * gaunt_set[out_idx, in_idx, q_array][None,:,None]
            factor = 4*jnp.pi * (1j)**(nu-n) * (-1.0)**(m)
            sum_val = factor * jnp.sum(basis_val * g, axis=1)
            tr_op = tr_op.at[..., out_idx, in_idx].set(sum_val)


    zero_pos = jnp.linalg.norm(pos, axis=-1) == 0
    tr_op = jnp.where(zero_pos[None,:,None,None], jnp.eye(num_coeffs, num_coeffs)[None,None,:,:], tr_op)
    return tr_op



@jax.jit
def basis_translation_3_80(pos, wave_num, gaunt_set):
    """Translation operator for shd basis function, such that 
    shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)

    Implemented according to equation 3.80 in 
    P. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles.

    Hardcoded for max_order = 1 to make compilation feasible
    Currently returns nan when pos is (0,0,0)
    
    Parameters
    ----------
    pos : ndarray of shape (num_pos, 3)
        position argument to the translation operator
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    max_order_input : int
        maximum order of the coefficients that should be translated
    max_order_output : int
        maximum order of the translated coefficients

    Returns
    -------
    T : ndarray of shape (num_freqs, num_pos, num_coeffs_output, num_coeffs_input)
        translation operator, such that shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)
    """
    max_order = 1
    num_coeffs = 4
    num_pos = pos.shape[0]
    num_freq = wave_num.shape[0]

    tr_op = jnp.zeros((num_freq, num_pos, num_coeffs, num_coeffs), dtype=complex)
    orders = np.array([0, 1, 1, 1])
    degrees = np.array([0, -1, 0, 1])

    max_tot_order = 2 * max_order 
    num_coeffs_tot = 9
    orders_tot = np.array([0,  1,  1,  1,  2,  2,  2,  2,  2])
    degrees_tot = np.array([0, -1,  0,  1, -2, -1,  0,  1,  2])

    num_q = max_tot_order + 1
    all_q = np.arange(max_tot_order+1)
    q_lst = []
    for out_idx, (n, m) in enumerate(zip(orders, degrees)):
        q_lst.append([])
        for in_idx, (nu, mu) in enumerate(zip(orders, degrees)):
            q_lst[out_idx].append(np.arange(np.abs(mu-m), n+nu+1))

    harm_idx_lst = []
    for out_idx, (n, m) in enumerate(zip(orders, degrees)):
        harm_idx_lst.append([])
        for in_idx, (nu, mu) in enumerate(zip(orders, degrees)):
            harm_idx_lst[out_idx].append(shd_coeffs_order_degree_to_index(q_lst[out_idx][in_idx], mu-m))

    radius, azimuth, zenith = cart2spherical(pos)
    radius = radius[None, None, ...]
    azimuth = azimuth[None, ...]
    zenith = zenith[None, ...]

    orders_tot = orders_tot[:, None]
    degrees_tot = degrees_tot[:, None]

    wave_num = wave_num[:, None, None]
    all_q = all_q[None,:,None]

    bessel_shape = (num_freq, num_q, num_pos)
    bessel = spherical_jn(jnp.ravel(jnp.broadcast_to(all_q, bessel_shape)), 
                                        jnp.ravel(jnp.broadcast_to(wave_num * radius, bessel_shape))) # ordered as (num_freq, num_q, num_pos)
    bessel = jnp.reshape(bessel, bessel_shape)

    harm_shape = (num_coeffs_tot, num_pos)
    harm = jax.scipy.special.sph_harm(jnp.ravel(jnp.broadcast_to(degrees_tot, harm_shape)), 
                                        jnp.ravel(jnp.broadcast_to(orders_tot, harm_shape)), 
                                        jnp.ravel(jnp.broadcast_to(azimuth, harm_shape)), 
                                        jnp.ravel(jnp.broadcast_to(zenith, harm_shape)), 
                                        max_tot_order) # ordered as (num_coeffs_tot, num_pos)
    harm = jnp.reshape(harm, harm_shape)
    harm = jnp.conj(harm)

    for out_idx, (n, m) in enumerate(zip(orders, degrees)):
        for in_idx, (nu, mu) in enumerate(zip(orders, degrees)):
            q_array = q_lst[out_idx][in_idx]
            harm_idxs = harm_idx_lst[out_idx][in_idx]

            basis_val = bessel[:,q_array,:] * harm[None,harm_idxs, :]
            g = (1j)**q_array[None,:,None] * gaunt_set[out_idx, in_idx, q_array][None,:,None]
            factor = 4*jnp.pi * (1j)**(nu-n) * (-1.0)**(m)
            sum_val = factor * jnp.sum(basis_val * g, axis=1)
            tr_op = tr_op.at[..., out_idx, in_idx].set(sum_val)


    zero_pos = jnp.linalg.norm(pos, axis=-1) == 0
    tr_op = jnp.where(zero_pos[None,:,None,None], jnp.eye(num_coeffs, num_coeffs)[None,None,:,:], tr_op)
    return tr_op





def _calculate_gaunt_set(max_order_input, max_order_output):
    """Numpy function for precaculating the Gaunt coefficients for the translation operator.

    Used to avoid having to write a jax implementation of wigner coefficients. 
    """
    num_coeffs_input = shd_num_coeffs(max_order_input)
    num_coeffs_output = shd_num_coeffs(max_order_output)
    orders_input, degrees_input = shd_num_degrees_vector(max_order_input)
    orders_output, degrees_output = shd_num_degrees_vector(max_order_output)
    gaunt = np.zeros((num_coeffs_output, num_coeffs_input, max_order_input+max_order_output+1), dtype=float)
    for out_idx, (n, m) in enumerate(zip(orders_output, degrees_output)):
        for in_idx, (nu, mu) in enumerate(zip(orders_input, degrees_input)):
            for q in range(n+nu+1):
                if np.abs(mu-m) <= q:
                    g = shd.gaunt_coefficient(n, m, nu, -mu, q)
                    gaunt[out_idx, in_idx, q] = g
    return gaunt


def shd_coeffs_order_degree_to_index(order, degree):
    return order**2 + order + degree



# def basis_translation_3_80(pos, wave_num, max_order_input, max_order_output, gaunt):
#     """Translation operator for shd basis function, such that 
#     shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)x

#     Implemented according to equation 3.80 in 
#     P. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles.
    
#     Parameters
#     ----------
#     pos : ndarray of shape (num_pos, 3)
#         position argument to the translation operator
#     wave_num : ndarray of shape (num_freq,)
#         wavenumber, defined as w / c where w is the angular frequency
#         and c is the speed of sound.
#     max_order_input : int
#         maximum order of the coefficients that should be translated
#     max_order_output : int
#         maximum order of the translated coefficients
#     gaunt : 
#         Gaunt coefficients for the translation operator. Calulate using calculate_gaunt_set

#     Returns
#     -------
#     T : ndarray of shape (num_freqs, num_pos, num_coeffs_output, num_coeffs_input)
#         translation operator, such that shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)
#     """
#     num_coeffs_input = shd_num_coeffs(max_order_input)
#     num_coeffs_output = shd_num_coeffs(max_order_output)
#     max_tot_order = max_order_input + max_order_output
#     num_pos = pos.shape[0]
#     num_freq = wave_num.shape[0]

#     tr_op = jnp.zeros((num_freq, num_pos, num_coeffs_output, num_coeffs_input), dtype=complex)

#     orders_input, degrees_input = shd_num_degrees_vector(max_order_input)
#     orders_output, degrees_output = shd_num_degrees_vector(max_order_output)

#     tr_op_f = []
#     for f in range(num_freq):
#         tr_op_fp = []
#         #for p in range(num_pos):
#             #tr_op_fpi = []
#         for out_idx, (n, m) in enumerate(zip(orders_output, degrees_output)):
#             tr_op_fpij = []
#             for in_idx, (nu, mu) in enumerate(zip(orders_input, degrees_input)):
#                 #sum_val = 0
#                 sum_val = []
#                 for q in range(n+nu+1):
#                     if jnp.abs(mu-m) <= q:
#                         basis_val = np.squeeze(shd_basis(pos, jnp.array([q]), jnp.array([mu-m]), wave_num[f:f+1], max_tot_order))
#                         sum_val.append((1j)**q * (-1.0)**(m) * jnp.conj(basis_val) * gaunt[out_idx, in_idx, q])

#                 sum_val = jnp.stack(sum_val, axis=-1)
#                 tot_sum_val = jnp.sqrt(4*jnp.pi) * (1j)**(nu-n) * np.sum(sum_val, axis=-1)

#                 tr_op_fpij.append(tot_sum_val)
#             tr_op_fpij = jnp.stack(tr_op_fpij, axis=-1)
#             #tr_op_fpi.append(tr_op_fpij)

#             #tr_op_fpi = jnp.stack(tr_op_fpi, axis=0)
#             tr_op_fp.append(tr_op_fpij)
#         tr_op_fp = jnp.stack(tr_op_fp, axis=1)
#         tr_op_f.append(tr_op_fp)    
#     tr_op = jnp.stack(tr_op_f, axis=0)

#     #for f in range(num_freq):
#     #    for p in range(num_pos):
#     #        if jnp.sum(np.abs(pos[p,:])) == 0:
#     #            tr_op[f, p, :, :] = jnp.eye(num_coeffs_output, num_coeffs_input)
#     return tr_op

# def calculate_gaunt_set(max_order_input, max_order_output):
#     """Numpy function for precaculating the Gaunt coefficients for the translation operator.

#     Used to avoid having to write a jax implementation of wigner coefficients. 
#     """
#     num_coeffs_input = shd.shd_num_coeffs(max_order_input)
#     num_coeffs_output = shd.shd_num_coeffs(max_order_output)
#     orders_input, degrees_input = shd.shd_num_degrees_vector(max_order_input)
#     orders_output, degrees_output = shd.shd_num_degrees_vector(max_order_output)
#     gaunt = np.zeros((num_coeffs_output, num_coeffs_input, max_order_input+max_order_output+1), dtype=float)
#     for out_idx, (n, m) in enumerate(zip(orders_output, degrees_output)):
#         for in_idx, (nu, mu) in enumerate(zip(orders_input, degrees_input)):
#             for q in range(n+nu+1):
#                 if np.abs(mu-m) <= q:
#                     g = shd.gaunt_coefficient(n, m, nu, -mu, q)
#                     gaunt[out_idx, in_idx, q] = g
#     return gaunt


def factorial(n):
    return jax.scipy.special.factorial(n, exact=False)

@jax.jit
def spherical_jn(v, z):
    """
    """
    return spherical_jn_max_order_2(v, z)


@jax.jit
def spherical_jn_max_order_2(v, z):
    """Calculates j_v(z), the spherical Bessel function of the first kind of order v

    Hard codes the function for order 0, 1 and 2, which is the ones required for sound field
    estimation with first order directional microphones. Uses a small-argument approximation to maintain
    acceptable numerical stability for small arguments.

    Parameters
    ----------
    v : ndarray of shape (num_points,)
        order of the spherical Bessel function
    z : ndarray of shape (num_points,)
        argument of the spherical Bessel function

    Returns
    -------
    order_v : ndarray of shape (num_points,)
        spherical Bessel function of the first kind of order v
        
    Notes
    -----
    The definitions are written explicitly at https://mathworld.wolfram.com/SphericalBesselFunctionoftheFirstKind.html
    Alternatively, the second-order function can be derived from the recursion defined in (14) in https://www.fresco.org.uk/functions/barnett/APP23.pdf

    The small-argument approximation is written explicitly at the end of page 2 in
    https://www.damtp.cam.ac.uk/user/tong/aqm/bessel.pdf

    """
    #assert np.all(v <= 2)
    # order_0 = jnp.sin(z) / z
    # order_1 = jnp.sin(z) / z**2 - jnp.cos(z) / z
    # order_2 = (3 / z**3 - 1 / z) * jnp.sin(z)  - (3 / z**2) * jnp.cos(z)

    order_0 = jnp.sinc(z / jnp.pi)
    order_1 = jnp.sinc(z / jnp.pi) / z - jnp.cos(z) / z

    div_term = 3 / z**2 
    order_2 = (div_term-1) * jnp.sinc(z / jnp.pi) - div_term * jnp.cos(z)

    order_1_small = z / 3
    order_2_small = z**2 / 15

    ORDER1_SWITCH_VALUE = 2.5e-4
    ORDER2_SWITCH_VALUE = 6e-3

    order_1 = jnp.where(z < ORDER1_SWITCH_VALUE, order_1_small, order_1)
    order_2 = jnp.where(z < ORDER2_SWITCH_VALUE, order_2_small, order_2)
    #order_2 = (div_term - 1) * jnp.sinc(z / jnp.pi)  - div_term * jnp.cos(z)

    #order_1 = jnp.where(z == 0, 0, order_1)
    #order_2 = jnp.where(z == 0, 0, order_2)
    all_orders = jnp.stack([order_0, order_1, order_2], axis=0)

    #selection = [all_orders[v_i] for v_i in v]
    #selection_idxs = [jnp.arange(z_shape) for z_shape in z.shape]

    return all_orders[v, jnp.arange(z.shape[-1])]


def spherical_jn_recursion(v, z):
    """Uses a simple recursion to calculate the spherical Bessel function of the first kind

    Is numerically unstable for small arguments for orders over 2. 
    
    Parameters
    ----------
    v : array of shape (num_points,)
        order of the spherical Bessel function
    z : array of shape (num_points,)
        argument of the spherical Bessel function
    
    Returns
    -------
    order_v : array_like
        spherical Bessel function of the first kind of order v

    Notes
    -----
    Uses (14) from:
    https://www.fresco.org.uk/functions/barnett/APP23.pdf
    """
    max_v = jnp.max(v)
    all_jn = []

    order_v_minus2 = jnp.sin(z) / z
    order_v_minus1 = jnp.sin(z) / z**2 - jnp.cos(z) / z
    all_jn.append(order_v_minus2)
    all_jn.append(order_v_minus1)

    for i in range(2, max_v+1):
        order_v = (2*i + 1) / z * order_v_minus1 - order_v_minus2
        all_jn.append(order_v)
        order_v_minus2 = order_v_minus1
        order_v_minus1 = order_v
    all_jn = jnp.stack(all_jn, axis=0)

    return all_jn[v,np.arange(z.shape[-1])]

def spherical_jn_recursion_scalar_order(v, z):
    """
    
    Parameters
    ----------
    v : int
        order of the spherical Bessel function
    z : array_like
        argument of the spherical Bessel function
    
    Returns
    -------
    order_v : array_like
        spherical Bessel function of the first kind of order v
    """
    order_v_minus2 = jnp.sin(z) / z
    if v == 0:
        return order_v_minus2
    order_v_minus1 = jnp.sin(z) / z**2 - jnp.cos(z) / z
    if v == 1:
        return order_v_minus1

    for i in range(2, v+1):
        order_v = (2*i + 1) / z * order_v_minus1 - order_v_minus2
        order_v_minus2 = order_v_minus1
        order_v_minus1 = order_v
    return order_v

def spherical_jn_log_series(v, z, max_idx=10):
    """Calculates j_v(z), the spherical Bessel function of the first kind of order v

    only accepts integer order v and real arguments z

    Parameters
    ----------
    v : ndarray of shape (num_points,)
        order of the spherical Bessel function
    z : ndarray of shape (num_points,)
        argument of the spherical Bessel function
    max_idx : int
        maximum index of the series expansion

    uses series expansion
    
    Notes
    -----
    max_ind does not increase the accuracy much after 10 or 15. 
    """
    #max_idx = 35
    #output = jnp.zeros_like(z)
    v = v[:,None]
    z = z[:,None]
    k = jnp.arange(max_idx)[None,:]

    log_terms = v * jnp.log(2) + (v + 2*k) * jnp.log(z) + jax.scipy.special.gammaln(k + v + 1) - jax.scipy.special.gammaln(k + 1) - jax.scipy.special.gammaln(2*k + 2*v + 2)
    all_series_terms = (-1)**(k) * jnp.exp(log_terms)
    output = jnp.sum(all_series_terms, axis=-1)
    return output


def spherical_jn_series(v, z, max_idx=10):
    """Calculates j_v(z), the spherical Bessel function of the first kind of order v

    only accepts integer order v and real arguments z

    Parameters
    ----------
    v : ndarray of shape (num_points,)
        order of the spherical Bessel function
    z : ndarray of shape (num_points,)
        argument of the spherical Bessel function
    max_idx : int
        maximum index of the series expansion

    uses series expansion
    
    Notes
    -----
    max_ind = 35 allows for high accuracy for z < 20
    requires 64bit precision, otherwise it will return NaNs. 
    up to 15 or 20 max_idx is generally fine if 32bit precision is used
    """
    #max_idx = 35
    #output = jnp.zeros_like(z)
    v = v[:,None]
    z = z[:,None]
    k = jnp.arange(max_idx)[None,:]

    order_factor = 2**(v) * (-1)**(k) * factorial(k + v) / (factorial(k) * factorial(2*k + 2*v + 1))
    argument = z**(v) * z**(2*k)

    return jnp.sum(order_factor * argument, axis=-1)

    #return np.sum(2**(v) * z**(v) * (-1)**(k) * z**(2*k) * factorial(k + v) / (factorial(k) * factorial(2*k + 2*v + 1)))
    # output = jnp.zeros_like(z)
    # for k in range(max_idx):
    #     order_factor = 2**(v) * ((-1)**(k) * factorial(k + v)) / (factorial(k) * factorial(2*k + 2*v + 1))
    #     output += order_factor * z**(v) * z**(2*k)
    # return output

def cart2spherical(cart_coord):
    """Transforms the provided cartesian coordinates to spherical coordinates

    Parameters
    ----------
    cart_coord : ndarray of shape (num_points, 3)

    Returns
    -------
    r : ndarray of shape (num_points, 1)
        radius of each point
    angle : ndarray of shape (num_points, 2)
        angle[:,0] is theta, the angle in the xy plane, where 0 is x direction, pi/2 is y direction
        angle[:,1] is phi, the zenith angle, where 0 is z direction, pi is negative z direction
    """
    r = jnp.linalg.norm(cart_coord, axis=1)
    r_xy = jnp.linalg.norm(cart_coord[:,:2], axis=1)

    azimuth = jnp.arctan2(cart_coord[:,1], cart_coord[:,0])
    zenith = jnp.arctan2(r_xy, cart_coord[:,2])
    #angle = jnp.concatenate((theta[:,None], phi[:,None]), axis=1)
    return (r, azimuth, zenith)



def shd_basis(pos, order, degree, wave_num, max_order):
    """Spherical harmonic basis function for sound field in 3D

    Implements: sqrt(4pi) j_order(kr) Y_order^degree(polar angle, zenith angle)
    degree and order of the spherical harmonics might be swapped according to some
    definitions.
    
    Parameters
    ----------
    pos : Tensor of shape (num_pos, 3)
        positions of the evaluation points
    order : ndarray of shape (num_coeffs,)
        order of the spherical harmonics. Can be any non-negative integer
    degree : ndarray of shape (num_coeffs,)
        degree of the spherical harmonics. Must satisfy |degree[i]| <= order[i] for all i
    wave_num : float or ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular 
        frequency and c is the speed of sound
    max_order : int
        required for jax to be able to compile the function

    Returns
    -------
    f : ndarray of shape (num_freq, num_coeffs, num_pos)
        values of the spherical harmonic basis function at the evaluation points

    Notes
    -----
    See more detailed derivations in any of:
    [1] Natsuki Ueno, Shoichi Koyama, Hiroshi Saruwatai, 'Sound Field Recording Using 
    Distributed Microphones Based on Harmonic Analysis of Infinite Order'
    [2] J. Brunnström, M.B. Moeller, M. Moonen, 'Bayesian sound field estimation using
    moving microphones'
    """
    #Parse arguments to get correct shapes
    num_pos = pos.shape[0]
    num_coeffs = order.shape[0]
    num_freq = wave_num.shape[0]

    radius, theta, phi = cart2spherical(pos)

    size = (num_freq, num_coeffs, num_pos)
    radius = jnp.broadcast_to(radius[None,None,:], size)
    theta = jnp.broadcast_to(theta[None,None,:], size)
    phi = jnp.broadcast_to(phi[None,None,:], size)
    order = jnp.broadcast_to(order[None,:,None], size)
    degree = jnp.broadcast_to(degree[None,:,None], size)
    wave_num = jnp.broadcast_to(wave_num[:,None,None], size)

    radius = jnp.reshape(radius, -1)
    theta = jnp.reshape(theta, -1)
    phi = jnp.reshape(phi, -1)
    order = jnp.reshape(order, -1)
    degree = jnp.reshape(degree, -1)
    wave_num = jnp.reshape(wave_num, -1)

    harm = jax.scipy.special.sph_harm(degree, order, theta, phi, n_max = max_order)

    # Calculate the function values

    bessel = spherical_jn(order, wave_num * radius)
    f = jnp.sqrt(4*jnp.pi) * bessel * harm
    f = jnp.reshape(f, size)
    return f

def shd_num_coeffs(max_order):
    """Number of coefficients required to represent a spherical harmonic function of a given order
    
    Parameters
    ----------
    max_order : int
        maximum order of the spherical harmonics
    
    Returns
    -------
    num_coeffs : int
        number of coefficients required to represent a spherical harmonic function of the given order
    """
    return (max_order+1)**2

def shd_num_degrees_vector(max_order : int):
    """
    Constructs a vector with the index of each order and degree
    when order = n, the degrees are only non-zero for -n <= degree <= n

    Parameters
    ----------
    max_order : int
        is the maximum order that is included

    Returns
    -------
    order : ndarray of shape ()
    degree : ndarray of shape ()
    """
    order = []
    degree = []

    for n in range(max_order+1):
        pos_degrees = jnp.arange(n+1)
        degree_n = jnp.concatenate((-jnp.flip(pos_degrees[1:]), pos_degrees))
        degree.append(degree_n)

        order.append(n*jnp.ones_like(degree_n))
    degree = jnp.concatenate(degree)
    order = jnp.concatenate(order)
    return order, degree























RP1 = jnp.array([
-8.99971225705559398224E8, 4.52228297998194034323E11,
-7.27494245221818276015E13, 3.68295732863852883286E15,])
RQ1 = jnp.array([
 1.0, 6.20836478118054335476E2, 2.56987256757748830383E5, 8.35146791431949253037E7, 
 2.21511595479792499675E10, 4.74914122079991414898E12, 7.84369607876235854894E14, 
 8.95222336184627338078E16, 5.32278620332680085395E18,])

PP1 = jnp.array([
 7.62125616208173112003E-4, 7.31397056940917570436E-2, 1.12719608129684925192E0, 
 5.11207951146807644818E0, 8.42404590141772420927E0, 5.21451598682361504063E0, 1.00000000000000000254E0,])
PQ1 = jnp.array([
 5.71323128072548699714E-4, 6.88455908754495404082E-2, 1.10514232634061696926E0, 
 5.07386386128601488557E0, 8.39985554327604159757E0, 5.20982848682361821619E0, 9.99999999999999997461E-1,])

QP1 = jnp.array([
 5.10862594750176621635E-2, 4.98213872951233449420E0, 7.58238284132545283818E1, 
 3.66779609360150777800E2, 7.10856304998926107277E2, 5.97489612400613639965E2, 2.11688757100572135698E2, 2.52070205858023719784E1,])
QQ1  = jnp.array([
 1.0, 7.42373277035675149943E1, 1.05644886038262816351E3, 4.98641058337653607651E3, 
 9.56231892404756170795E3, 7.99704160447350683650E3, 2.82619278517639096600E3, 3.36093607810698293419E2,])

YP1 = jnp.array([
 1.26320474790178026440E9,-6.47355876379160291031E11, 1.14509511541823727583E14,
 -8.12770255501325109621E15, 2.02439475713594898196E17,-7.78877196265950026825E17,])
YQ1 = jnp.array([
 5.94301592346128195359E2, 2.35564092943068577943E5, 7.34811944459721705660E7, 
 1.87601316108706159478E10, 3.88231277496238566008E12, 6.20557727146953693363E14, 
 6.87141087355300489866E16, 3.97270608116560655612E18,])

Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1
PIO4 = .78539816339744830962 # pi/4
THPIO4 = 2.35619449019234492885 # 3*pi/4
SQ2OPI = .79788456080286535588 # sqrt(2/pi)

PP0 = jnp.array([
  7.96936729297347051624E-4,
  8.28352392107440799803E-2,
  1.23953371646414299388E0,
  5.44725003058768775090E0,
  8.74716500199817011941E0,
  5.30324038235394892183E0,
  9.99999999999999997821E-1,
])
PQ0 = jnp.array([
  9.24408810558863637013E-4,
  8.56288474354474431428E-2,
  1.25352743901058953537E0,
  5.47097740330417105182E0,
  8.76190883237069594232E0,
  5.30605288235394617618E0,
  1.00000000000000000218E0,
])

QP0 = jnp.array([
-1.13663838898469149931E-2,
-1.28252718670509318512E0,
-1.95539544257735972385E1,
-9.32060152123768231369E1,
-1.77681167980488050595E2,
-1.47077505154951170175E2,
-5.14105326766599330220E1,
-6.05014350600728481186E0,
])
QQ0 = jnp.array([1.0,
  6.43178256118178023184E1,
  8.56430025976980587198E2,
  3.88240183605401609683E3,
  7.24046774195652478189E3,
  5.93072701187316984827E3,
  2.06209331660327847417E3,
  2.42005740240291393179E2,
])

YP0 = jnp.array([
 1.55924367855235737965E4,
-1.46639295903971606143E7,
 5.43526477051876500413E9,
-9.82136065717911466409E11,
 8.75906394395366999549E13,
-3.46628303384729719441E15,
 4.42733268572569800351E16,
-1.84950800436986690637E16,
])
YQ0 = jnp.array([
 1.04128353664259848412E3,
 6.26107330137134956842E5,
 2.68919633393814121987E8,
 8.64002487103935000337E10,
 2.02979612750105546709E13,
 3.17157752842975028269E15,
 2.50596256172653059228E17,
])

DR10 = 5.78318596294678452118E0
DR20 = 3.04712623436620863991E1

RP0 = jnp.array([
-4.79443220978201773821E9,
 1.95617491946556577543E12,
-2.49248344360967716204E14,
 9.70862251047306323952E15,
])
RQ0 = jnp.array([ 1.0,
 4.99563147152651017219E2,
 1.73785401676374683123E5,
 4.84409658339962045305E7,
 1.11855537045356834862E10,
 2.11277520115489217587E12,
 3.10518229857422583814E14,
 3.18121955943204943306E16,
 1.71086294081043136091E18,
])

def j0_small(x):
    '''
    Implementation of J0 for x < 5 
    '''
    z = x * x
    # if x < 1.0e-5:
    #     return 1.0 - z/4.0

    p = (z - DR10) * (z - DR20)
    p = p * jnp.polyval(RP0,z)/jnp.polyval(RQ0, z)
    return jnp.where(x<1e-5,1-z/4.0,p)
    

def j0_large(x):
    '''
    Implementation of J0 for x >= 5
    '''

    w = 5.0/x
    q = 25.0/(x*x)
    p = jnp.polyval(PP0, q)/jnp.polyval(PQ0, q)
    q = jnp.polyval(QP0, q)/jnp.polyval(QQ0, q)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)

def j0(x):
    '''
    Implementation of J0 for all x in Jax


    References
    ----------
    Given in https://github.com/benjaminpope/sibylla/blob/main/notebooks/bessel_test.ipynb
    under MIT License
    '''

    return jnp.where(jnp.abs(x) < 5.0, j0_small(jnp.abs(x)),j0_large(jnp.abs(x)))

def j1_small(x):
    z = x * x
    w = jnp.polyval(RP1, z) / jnp.polyval(RQ1, z)
    w = w * x * (z - Z1) * (z - Z2)
    return w

def j1_large_c(x):    
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(PP1, z) / jnp.polyval(PQ1, z)
    q = jnp.polyval(QP1, z) / jnp.polyval(QQ1, z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)

def j1(x):
    """
    Bessel function of order one - using the implementation from CEPHES, translated to Jax.
    """
    return jnp.sign(x)*jnp.where(jnp.abs(x) < 5.0, j1_small(jnp.abs(x)),j1_large_c(jnp.abs(x)))

def jv(n, x):
    """Compute the Bessel function J_n(x), for n >= 0"""
    # use recurrence relations
    def body(carry, i):
        jnm1, jn = carry
        jnplus = (2*i)/x * jn - jnm1
        return (jn, jnplus), jnplus
    _, jn = jax.lax.scan(body, (j0(x), j1(x)), jnp.arange(1,n))
    return jn[-1]


def jneghalf(x):
    """Compute the Bessel function J_{-1/2}(x)"""
    return jnp.sqrt(2/(jnp.pi*x)) * jnp.cos(x)

def jposhalf(x):
    """Compute the Bessel function J_{1/2}(x)"""
    return jnp.sqrt(2/(jnp.pi*x)) * jnp.sin(x)


def jvplushalf(n, x):
    """Compute the Bessel function J_n+1/2(x), for n >= 0"""
    # use recurrence relations
    def body(carry, i):
        jnm1, jn = carry
        jnplus = (2*i)/x * jn - jnm1
        return (jn, jnplus), jnplus
    _, jn = jax.lax.scan(body, (jneghalf(x), jposhalf(x)), jnp.arange(0,n)+0.5)
    return jn[-1]