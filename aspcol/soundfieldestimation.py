"""Collection of algorithms for sound field estimation

* Kernel interpolation [uenoKernel2018]
* Infinite dimensional spherical harmonic analysis for moving microphones [brunnstromBayesianSubmitted]
* Spatial spectrum estimation for moving microphones [katzbergSpherical2021]

References
----------
[uenoKernel2018] N. Ueno, S. Koyama, and H. Saruwatari, “Kernel ridge regression with constraint of Helmholtz equation for sound field interpolation,” in 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan: IEEE, Sep. 2018, pp. 436–440. doi: 10.1109/IWAENC.2018.8521334. `[link] <https://doi.org/10.1109/IWAENC.2018.8521334>`__ \n
[brunnstromBayesianSubmitted] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted. \n
[katzbergSpherical2021] F. Katzberg, M. Maass, and A. Mertins, “Spherical harmonic representation for dynamic sound-field measurements,” in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Jun. 2021, pp. 426–430. doi: 10.1109/ICASSP39728.2021.9413708. `[link] <https://doi.org/10.1109/ICASSP39728.2021.9413708>`__ \n
"""
import numpy as np
import scipy.linalg as splin
import scipy.spatial.distance as spdist
import scipy.special as spspec

import aspcol.kernelinterpolation as ki
import aspcol.utilities as util
import aspcol.filterdesign as fd
import aspcol.pseq as pseq
import aspcol.sphericalharmonics as sph
import aspcol.fouriertransform as ft



#============= FREQUENCY DOMAIN METHODS - STATIONARY MICROPHONES =============
def est_ki_diffuse_freq(p_freq, pos, pos_eval, k, reg_param):
    """Estimates the RIR in the frequency domain using kernel interpolation
    
    Uses the frequency domain sound pressure as input

    Parameters
    ----------
    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    k : ndarray of shape (num_freq)
        wavenumbers
    reg_param : float
        regularization parameter for kernel interpolation

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    References
    ----------
    [uenoKernel2018]
    """
    est_filt = ki.get_krr_parameters(ki.kernel_helmholtz_3d, reg_param, pos_eval, pos, k)
    p_ki = est_filt @ p_freq[:,:,None]
    return np.squeeze(p_ki, axis=-1)


def nearest_neighbour_freq(p_freq, pos, pos_eval):
    """
    Estimates the sound field at the evaluation points by simply selecting the value
    associated with the nearest microphone position. 

    Parameters
    ----------
    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    dist = spdist.cdist(pos, pos_eval)
    min_idx = np.argmin(dist, axis=0)

    num_real_freqs = p_freq.shape[0]
    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for pos_idx in range(pos_eval.shape[0]):
        est_sound_pressure[:,pos_idx] = p_freq[:,min_idx[pos_idx]]
    return est_sound_pressure


def inf_dim_shd_analysis(p_freq, pos, pos_eval, wave_num, dir_coeffs, reg_param):
    """Estimates the sound field using directional microphones 

    Since we reconstruct the sound pressure immediately, without explicitly computing
    the spherical harmonic coefficients, no truncation or expansion center must be set.
    
    Parameters
    ----------
    p_freq : ndarray of shape (num_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    wave_num : ndarray of shape (num_freq)
        wavenumbers
    dir_coeffs : ndarray of shape (num_freq, num_mic, num_coeffs)
        harmonic coefficients of the directionality of the microphones
    reg_param : float
        regularization parameter. Must be non-negative
    
    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    num_mic = pos.shape[0]

    psi = sph.translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, wave_num)
    psi_plus_noise_cov = psi + np.eye(num_mic) * reg_param
    regression_vec = np.linalg.solve(psi_plus_noise_cov, p_freq)[...,None]

    omni_dir = sph.directivity_omni()
    estimator_matrix = sph.translated_inner_product(pos_eval, pos, omni_dir, dir_coeffs, wave_num)
    est_sound_pressure = np.squeeze(estimator_matrix @ regression_vec, axis=-1)
    return est_sound_pressure
# ============= TIME DOMAIN METHODS - STATIONARY MICROPHONES =============


def pseq_nearest_neighbour(p, seq, pos, pos_eval):
    """
    Estimates the sound field at the evaluation points by simply selecting the value
    associated with the nearest microphone position. 
    Assumes that the sequence is a perfect periodic sequence.

    Parameters
    ----------
    p : ndarray of shape (num_mic, seq_len)
        sound pressure in time domain at num_mic microphone positions
    seq : ndarray of shape (seq_len) or (1, seq_len)
        training signal used for the measurements
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points
    """
    rir = pseq.decorrelate(p, seq)
    rir_freq = ft.rfft(rir)

    dist = spdist.cdist(pos, pos_eval)
    min_idx = np.argmin(dist, axis=0)

    num_real_freqs = rir_freq.shape[0]
    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for pos_idx in range(pos_eval.shape[0]):
        est_sound_pressure[:,pos_idx] = rir_freq[:,min_idx[pos_idx]]
    return est_sound_pressure


def est_ki_diffuse(p, seq, pos, pos_eval, samplerate, c, reg_param):
    """
    Estimates the RIR in the frequency domain using kernel interpolation
    Assumes seq is a perfect periodic sequence

    Parameters
    ----------
    p : ndarray of shape (M, seq_len)
        sound pressure in time domain at M microphone positions
    seq : ndarray of shape (seq_len)
        the training signal used for the measurements
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    samplerate : int
    c : float
        speed of sound
    reg_param : float
        regularization parameter for kernel interpolation

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    References
    ----------
    [uenoKernel2018]
    """
    rir = pseq.decorrelate(p, seq)

    fft_len = rir.shape[-1]
    rir_freq = ft.rfft(rir)
    k = ft.get_real_wavenum(fft_len, samplerate, c)

    return est_ki_diffuse_freq(rir_freq, pos, pos_eval, k, reg_param)











#============= MOVING MICROPHONE ESTIMATION =============


def est_inf_dimensional_shd_dynamic(p, pos, pos_eval, sequence, samplerate, c, reg_param, verbose=False):
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
    if p.ndim >= 2:
        p = np.squeeze(p)

    if sequence.ndim == 2:
        sequence = np.squeeze(sequence, axis=0)
    assert sequence.ndim == 1

    assert pos.ndim == 2

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
    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for f in range(num_real_freqs):
        kernel_val = ki.kernel_helmholtz_3d(pos_eval, pos, k[f:f+1]).astype(complex)[0,:,:]
        est_sound_pressure[f, :] = np.sum(kernel_val * regressor[f,None,:], axis=-1)

    if verbose:
        #diagnostics = {}
        #diagnostics["regularization parameter"] = reg_param
        #diagnostics["condition number"] = np.linalg.cond(psi).tolist()
        #diagnostics["smallest eigenvalue"] = splin.eigh(psi, subset_by_index=(0,0), eigvals_only=True).tolist()
        #diagnostics["largest eigenvalue"] = splin.eigh(psi, subset_by_index=(N-1, N-1), eigvals_only=True).tolist()
        return est_sound_pressure, regressor, psi#, diagnostics
    else:
        return est_sound_pressure

def reconstruct_inf_dimensional_shd_dynamic(regressor, pos_eval, pos, k):
    """
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



def est_spatial_spectrum_dynamic(p, pos, pos_eval, sequence, samplerate, c, reg_param, r_max=None, verbose=False):
    """
    Estimates the RIR at evaluation positions using data from a moving microphone

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
    max_orders = sph.shd_min_order(k[:num_real_freqs], r_max)
    max_orders = np.concatenate((max_orders, np.flip(max_orders[1:])))
    
    Sigma = []

    freq_idx_list = np.arange(num_real_freqs)
    for f in freq_idx_list:#range(num_real_freqs):
        order, modes = sph.shd_num_degrees_vector(max_orders[f])
        Y_f = spspec.sph_harm(modes[None,:], order[None,:], angles[:,0:1], angles[:,1:2])
        B_f = spspec.spherical_jn(order[None,:], k[f]*r[:,None])

       # D_f = spspec.spherical_jn(order, k[f]*r_max)
        S_f = phi[f,:]

        Sigma_f = S_f[:,None] * Y_f * B_f #/ D_f[None,:]
        Sigma.append(Sigma_f)

    # for f in np.flip(np.arange(1, num_real_freqs)):
    #     order, modes = sph.shd_num_degrees_vector(max_orders[f])
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
        order, modes = sph.shd_num_degrees_vector(max_orders[f])
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


def lsqL2(A, y, lamb=1e-10):
    U,S, Vh = np.linalg.svd(A, full_matrices=False)
    return np.conj(Vh).T @ ((np.conj(U).T @ y) * (S/(S**2+lamb)))


# def est_spatial_spectrum_dynamic(p, pos, pos_eval, sequence, samplerate, c, r_max, verbose=False):
#     """
#     Estimates the RIR at evaluation positions using data from a moving microphone

#     Implements the method from Katzberg et al. "Spherical harmonic 
#     representation for dynamic sound-field measurements"
    
#     Assumptions:
#     The spherical harmonics are expanded around the origin (0,0,0)
#     The sequence is periodic, and the pressure is measured for an integer number of periods
#     The length of sequence is the length of the estimated RIR

#     Parameters
#     ----------
#     p : ndarray of shape (N)
#         sound pressure for each sample of the moving microphone
#     pos : ndarray of shape (N, 3)
#         position of the trajectory for each sample
#     pos_eval : ndarray of shape (num_eval, 3)
#         positions of the evaluation points
#     sequence : ndarray of shape (seq_len) or (1, seq_len)
#         the training signal used for the measurements
#     samplerate : int
#     c : float
#         speed of sound
#     r_max : float
#         radius of the sphere onto which the spatial spectrum is computed
#     verbose : bool, optional
#         if True, returns diagnostics, by default False

#     Returns
#     -------
#     est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
#         estimated RIR per frequency at the evaluation points
#     """
#     # ======= Argument checking =======
#     if p.ndim >= 2:
#         p = np.squeeze(p)

#     if sequence.ndim == 2:
#         sequence = np.squeeze(sequence, axis=0)
#     assert sequence.ndim == 1

#     N = p.shape[0]
#     seq_len = sequence.shape[0]
#     num_periods = N // seq_len
#     assert N % seq_len == 0
#     num_eval = pos_eval.shape[0]

#     k = ft.get_wavenum(seq_len, samplerate, c)
#     num_freqs = len(k)
#     num_real_freqs = len(ft.get_real_freqs(seq_len, samplerate))

#     # ======= Estimation of spatial spectrum coefficients =======
#     phi = _sequence_stft_multiperiod(sequence, num_periods)
#     max_orders = sph.shd_min_order(k, r_max)
#     r, angles = util.cart2spherical(pos)

#     Sigma = []

#     for f in range(num_freqs):
#         order, modes = sph.shd_num_degrees_vector(max_orders[f])
#         Y_f = spspec.sph_harm(modes[None,:], order[None,:], angles[:,0:1], angles[:,1:2])
#         B_f = spspec.spherical_jn(order[None,:], k[f]*r[:,None])

#         D_f = spspec.spherical_jn(order, k[f]*r_max)
#         S_f = phi[f,:]

#         Sigma_f = S_f[:,None] * Y_f * B_f / D_f[None,:]
#         Sigma.append(Sigma_f)

#     Sigma = np.concatenate(Sigma, axis=-1)

#     a, residue, rank, singular_values = np.linalg.lstsq(Sigma, p, rcond=None)


#     # ======= Reconstruction of RIR =======
#     rir_est = np.zeros((num_real_freqs, num_eval), dtype=complex)
#     r_eval, angles_eval = util.cart2spherical(pos_eval)
#     ord_idx = 0
#     for f in range(num_real_freqs):
#         order, modes = sph.shd_num_degrees_vector(max_orders[f])
#         num_ord = len(order)
        
#         j_denom = spspec.spherical_jn(order, k[f]*r_max)
#         j_num = spspec.spherical_jn(order[None,:], k[f]*r_eval[:,None])

#         Y = spspec.sph_harm(modes[None,:], order[None,:], angles_eval[:,0:1], angles_eval[:,1:2])
#         rir_est[f, :] = np.sum(a[None,ord_idx:ord_idx+num_ord] * Y * j_num / j_denom[None,:], axis=-1)
#         ord_idx += num_ord

#     if verbose:
#         diagnostics = {}
#         diagnostics["residue"] = residue.tolist()
#         diagnostics["condition number"] = np.linalg.cond(Sigma).tolist()
#         diagnostics["smallest singular value"] = splin.svdvals(Sigma)[0].tolist()
#         diagnostics["largest singular"] = splin.svdvals(Sigma)[-1].tolist()
#         diagnostics["r_max"] = r_max
#         return rir_est, diagnostics
#     return rir_est



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

    Currently is likely using the wrong time convention. 

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



