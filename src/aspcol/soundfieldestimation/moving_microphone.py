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

import aspcol.sphericalharmonics as shd
import aspcol.utilities as util
import aspcol.kernelinterpolation as ki


def inf_dimensional_shd_dynamic(p, pos, pos_eval, sequence, samplerate, c, reg_param, dir_coeffs=None, verbose=False):
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
    dir_coeffs : ndarray of shape (N, num_coeffs), optional
        harmonic coefficients of microphone directivity, 
        if not supplied, the directivity is assumed to be omnidirectional. 
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

    if dir_coeffs is None:
        return _est_inf_dimensional_shd_omni(p, pos, pos_eval, sequence, samplerate, c, reg_param, verbose)
    if dir_coeffs.ndim == 2:
        dir_coeffs = dir_coeffs[None,:,:] #add a dimension for the number of frequencies
    assert dir_coeffs.ndim == 3
    assert dir_coeffs.shape[1] == N or dir_coeffs.shape[1] == 1
    assert dir_coeffs.shape[0] == num_real_freqs or dir_coeffs.shape[0] == 1
    

    # ======= Estimation of spherical harmonic coefficients =======
    Phi = _sequence_stft_multiperiod(sequence, num_periods)
    
    psi = calculate_psi(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs)
    noise_cov = reg_param * np.eye(N)
    regressor = splin.solve(psi + noise_cov, p, assume_a = "pos")
    regressor = Phi.conj()[:num_real_freqs,:] * regressor[None,:]

    est_sound_pressure = estimate_from_regressor(regressor, pos, pos_eval, wave_num[:num_real_freqs], dir_coeffs)
    
    if verbose:
        return est_sound_pressure, regressor, psi
    return est_sound_pressure


def _est_inf_dimensional_shd_omni(p, pos, pos_eval, sequence, samplerate, c, reg_param, verbose=False):
    """Estimates the RIR at evaluation positions using data from a moving microphone
    using Bayesian inference of an infinite sequence of spherical harmonics

    Assumptions:
    The microphones are omnidirectional
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
    verbose : bool, optional
        if True, returns diagnostics, by default False

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    References
    ----------
    [brunnstromBayesianSubmitted]
    """
    # ======= Argument parsing =======
    #if p.ndim >= 2:
    #    p = np.squeeze(p)

    #if sequence.ndim == 2:
    #    sequence = np.squeeze(sequence, axis=0)
    #assert sequence.ndim == 1

    #assert pos.ndim == 2

    N = p.shape[0]
    seq_len = sequence.shape[0]
    num_periods = N // seq_len
    assert N % seq_len == 0

    k = ft.get_wavenum(seq_len, samplerate, c)
    num_real_freqs = len(ft.get_real_freqs(seq_len, samplerate))

    # ======= Estimation of spherical harmonic coefficients =======
    Phi = _sequence_stft_multiperiod(sequence, num_periods)

    #division by pi is a correction for the sinc function used later
    dist_mat = np.sqrt(np.sum((np.expand_dims(pos,1) - np.expand_dims(pos,0))**2, axis=-1))  / np.pi 
    
    psi = np.zeros((N, N), dtype = float)

    # no conjugation required for zeroth frequency and the Nyquist frequency, 
    # since they will be real already for a real input sequence
    psi += np.sinc(dist_mat * k[0]) * np.real_if_close(Phi[0,:,None] * Phi[0,None,:])
    assert seq_len % 2 == 0 #following line is only correct if B is even
    psi += np.sinc(dist_mat * k[seq_len//2]) * np.real_if_close(Phi[seq_len//2,:,None] * Phi[seq_len//2,None,:])

    for f in range(1, num_real_freqs-1):
        phi_rank1_matrix = Phi[f,:,None] * Phi[f,None,:].conj()
        psi += 2*np.real(np.sinc(dist_mat * k[f]) * phi_rank1_matrix)

    noise_cov = reg_param * np.eye(N)
    regressor = splin.solve(psi + noise_cov, p, assume_a = "pos")

    regressor = Phi.conj() * regressor[None,:]
    regressor = regressor[:num_real_freqs,:] # should be replaced by fixing the function calculating Phi instead

    # ======= Reconstruction of RIR =======
    est_sound_pressure = _estimate_from_regressor_omni(regressor, pos_eval, pos, k[:num_real_freqs])

    #est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    #for f in range(num_real_freqs):
    #    kernel_val = ki.kernel_helmholtz_3d(pos_eval, pos, k[f:f+1]).astype(complex)[0,:,:]
    #    est_sound_pressure[f, :] = np.sum(kernel_val * regressor[f,None,:], axis=-1)

    if verbose:
        #diagnostics = {}
        #diagnostics["regularization parameter"] = reg_param
        #diagnostics["condition number"] = np.linalg.cond(psi).tolist()
        #diagnostics["smallest eigenvalue"] = splin.eigh(psi, subset_by_index=(0,0), eigvals_only=True).tolist()
        #diagnostics["largest eigenvalue"] = splin.eigh(psi, subset_by_index=(N-1, N-1), eigvals_only=True).tolist()
        return est_sound_pressure, regressor, psi
    else:
        return est_sound_pressure



def est_spatial_spectrum_dynamic(p, pos, pos_eval, sequence, samplerate, c, reg_param, r_max=None, verbose=False):
    """Estimates the RIR at evaluation positions using data from a moving microphone

    Implements the method from Katzberg et al. "Spherical harmonic 
    representation for dynamic sound-field measurements"
    
    Assumptions:
    The spherical harmonics are expanded around the origin (0,0,0)
    The sequence is periodic, and the pressure is measured for an integer number of periods
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
    r_max : float, optional
        radius of the sphere onto which the spatial spectrum is computed. If not provided, it is 
        set to the maximum distance from the origin to any of the microphone positions. 
    verbose : bool, optional
        if True, returns diagnostics, by default False

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    # ======= Argument checking =======
    if p.ndim >= 2:
        p = np.squeeze(p)

    if sequence.ndim == 2:
        sequence = np.squeeze(sequence, axis=0)
    assert sequence.ndim == 1

    N = p.shape[0]
    seq_len = sequence.shape[0]
    num_periods = N // seq_len
    assert N % seq_len == 0
    num_eval = pos_eval.shape[0]

    k = ft.get_wavenum(seq_len, samplerate, c)
    num_freqs = len(k)
    num_real_freqs = len(ft.get_real_freqs(seq_len, samplerate))

    r, angles = util.cart2spherical(pos)

    # ======= Estimation of spatial spectrum coefficients =======
    phi = _sequence_stft_multiperiod(sequence, num_periods)

    if r_max is None:
        r_max = np.max(r)
    max_orders = shd.shd_min_order(k[:num_real_freqs], r_max)
    max_orders = np.concatenate((max_orders, np.flip(max_orders[1:])))
    
    Sigma = []

    freq_idx_list = np.arange(num_real_freqs)
    for f in freq_idx_list:#range(num_real_freqs):
        order, modes = shd.shd_num_degrees_vector(max_orders[f])
        Y_f = spspec.sph_harm(modes[None,:], order[None,:], angles[:,0:1], angles[:,1:2])
        B_f = spspec.spherical_jn(order[None,:], k[f]*r[:,None])

       # D_f = spspec.spherical_jn(order, k[f]*r_max)
        S_f = phi[f,:]

        Sigma_f = S_f[:,None] * Y_f * B_f #/ D_f[None,:]
        Sigma.append(Sigma_f)

    # for f in np.flip(np.arange(1, num_real_freqs)):
    #     order, modes = shd.shd_num_degrees_vector(max_orders[f])
    #     Y_f = spspec.sph_harm(modes[None,:], order[None,:], angles[:,0:1], angles[:,1:2])
    #     B_f = spspec.spherical_jn(order[None,:], k[f]*r[:,None])
    #     S_f = phi[f,:]
    #     Sigma_f = S_f[:,None] * Y_f * B_f
    #     Sigma.append(np.conj(Sigma_f))

    Sigma = np.concatenate(Sigma, axis=-1)
    system_mat = Sigma.conj().T @ Sigma + reg_param * np.eye(Sigma.shape[-1])
    print(f"Size of spatial spectrum system matrix: {system_mat.shape}")
    a = splin.solve(system_mat, Sigma.conj().T @ p, assume_a="pos")
    #a, residue, rank, singular_values = np.linalg.lstsq(Sigma, p, rcond=None)
    #a = lsqL2(Sigma, p, 1e-6)


    # ======= Reconstruction of RIR =======
    rir_est = np.zeros((num_real_freqs, num_eval), dtype=complex)
    r_eval, angles_eval = util.cart2spherical(pos_eval)
    ord_idx = 0
    for f in range(num_real_freqs):
        order, modes = shd.shd_num_degrees_vector(max_orders[f])
        num_ord = len(order)
        
        #j_denom = spspec.spherical_jn(order, k[f]*r_max)
        j_num = spspec.spherical_jn(order[None,:], k[f]*r_eval[:,None])

        Y = spspec.sph_harm(modes[None,:], order[None,:], angles_eval[:,0:1], angles_eval[:,1:2])
        rir_est[f, :] = np.sum(a[None,ord_idx:ord_idx+num_ord] * Y * j_num  , axis=-1) # / j_denom[None,:]
        ord_idx += num_ord

    if verbose:
        diagnostics = {}
        diagnostics["condition number"] = np.linalg.cond(Sigma).tolist()
        diagnostics["smallest singular value"] = splin.svdvals(Sigma)[0].tolist()
        diagnostics["largest singular"] = splin.svdvals(Sigma)[-1].tolist()
        diagnostics["r_max"] = r_max
        return rir_est, diagnostics
    return rir_est


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
    N = pos.shape[0]
    assert pos.shape == (N, 3)
    assert pos_eval.ndim == 2 and pos_eval.shape[1] == 3

    num_real_freqs = wave_num.shape[-1]

    if dir_coeffs is None:
        return _estimate_from_regressor_omni(regressor, pos_eval, pos, wave_num)
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

def _estimate_from_regressor_omni(regressor, pos_eval, pos, k):
    """Takes the regressor from inf_dimensional_shd_dynamic, and gives back a sound field estimate. 
    Reconstructs the sound field at the evaluation points using the regressor matrix
    from est_inf_dimensional_shd_dynamic

    Parameters
    ----------
    regressor : ndarray of shape (num_real_freqs, N)
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
    num_real_freqs = regressor.shape[0]
    assert k.shape[-1] == num_real_freqs
    assert pos.shape[0] == regressor.shape[-1]

    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for f in range(num_real_freqs):
        kernel_val = ki.kernel_helmholtz_3d(pos_eval, pos, k[f:f+1]).astype(complex)[0,:,:]
        est_sound_pressure[f, :] = np.sum(kernel_val * regressor[f,None,:], axis=-1)
    return est_sound_pressure



def calculate_psi(pos, dir_coeffs, k, Phi, seq_len, num_real_freqs):
    N = pos.shape[0]
    psi = np.zeros((N, N), dtype = float)

    #print(f"starting TIP")
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

