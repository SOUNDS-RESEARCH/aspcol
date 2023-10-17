"""Collection of algorithms for sound field estimation

* Kernel interpolation [1]
* Infinite dimensional spherical harmonic analysis for moving microphones [2]
* Spatial spectrum estimation for moving microphones [3]

References
----------
`[1] <doi.org/10.1109/IWAENC.2018.8521334>`_ N. Ueno, S. Koyama, and H. Saruwatari, “Kernel ridge regression with constraint of Helmholtz equation for sound field interpolation,” in 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan: IEEE, Sep. 2018, pp. 436–440. doi: 10.1109/IWAENC.2018.8521334.
[2] J. Brunnström, M. B. Mo/ller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted.
`[3] <doi.org/10.1109/ICASSP39728.2021.9413708>`_ F. Katzberg, M. Maass, and A. Mertins, “Spherical harmonic representation for dynamic sound-field measurements,” in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Jun. 2021, pp. 426–430. doi: 10.1109/ICASSP39728.2021.9413708.


"""
import numpy as np
import scipy.linalg as splin
import scipy.spatial.distance as spdist
import scipy.special as spspec

import aspcol.kernelinterpolation as ki
import aspcol.utilities as util
import aspcol.filterdesign as fd
import aspcol.pseq as pseq


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
    rir_freq = np.fft.rfft(rir, axis=-1).T

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
    """
    rir = pseq.decorrelate(p, seq)

    fft_len = rir.shape[-1]
    rir_freq = np.fft.rfft(rir, axis=-1).T
    k = fd.get_real_wavenum(fft_len, samplerate, c)

    return est_ki_diffuse_freq(rir_freq, pos, pos_eval, k, reg_param)

def est_ki_diffuse_freq(p_freq, pos, pos_eval, k, reg_param):
    """
    Estimates the RIR in the frequency domain using kernel interpolation
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
    """
    est_filt = ki.get_krr_parameters(ki.kernel_helmholtz_3d, reg_param, pos_eval, pos, k)
    p_ki = est_filt @ p_freq[:,:,None]
    return np.squeeze(p_ki, axis=-1)





def est_inf_dimensional_shd_dynamic(p, pos, pos_eval, sequence, samplerate, c, reg_param, verbose=False):
    """
    Estimates the RIR at evaluation positions using data from a moving microphone
    using Bayesian inference of an infinite sequence of spherical harmonics

    Implements the method in J. Brunnström, M.B. Moeller, M. Moonen, 
    "Bayesian sound field estimation using moving microphones" 

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
    """
    # ======= Argument parsing =======
    if p.ndim >= 2:
        p = np.squeeze(p)

    if sequence.ndim == 2:
        sequence = np.squeeze(sequence, axis=0)
    assert sequence.ndim == 1

    N = p.shape[0]
    seq_len = sequence.shape[0]
    num_periods = N // seq_len
    assert N % seq_len == 0

    k = fd.get_wavenum(seq_len, samplerate, c)
    num_real_freqs = len(fd.get_real_freqs(seq_len, samplerate))

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
    right_side = splin.solve(psi + noise_cov, p, assume_a = "pos")

    right_side = Phi.conj() * right_side[None,:]

    # ======= Reconstruction of RIR =======
    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for f in range(num_real_freqs):
        kernel_val = ki.kernel_helmholtz_3d(pos_eval, pos, k[f:f+1]).astype(complex)[0,:,:]
        est_sound_pressure[f, :] = np.sum(kernel_val * right_side[f,None,:], axis=-1)

    if verbose:
        diagnostics = {}
        diagnostics["regularization parameter"] = reg_param
        diagnostics["condition number"] = np.linalg.cond(psi).tolist()
        diagnostics["smallest eigenvalue"] = splin.eigh(psi, subset_by_index=(0,0), eigvals_only=True).tolist()
        diagnostics["largest eigenvalue"] = splin.eigh(psi, subset_by_index=(N-1, N-1), eigvals_only=True).tolist()
        return est_sound_pressure, diagnostics
    else:
        return est_sound_pressure



def est_spatial_spectrum_dynamic(p, pos, pos_eval, sequence, samplerate, c, r_max, verbose=False):
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
    r_max : float
        radius of the sphere onto which the spatial spectrum is computed
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

    k = fd.get_wavenum(seq_len, samplerate, c)
    num_freqs = len(k)
    num_real_freqs = len(fd.get_real_freqs(seq_len, samplerate))

    # ======= Estimation of spatial spectrum coefficients =======
    phi = _sequence_stft_multiperiod(sequence, num_periods)
    max_orders = shd_min_order(k, r_max)
    r, angles = util.cart2spherical(pos)

    Sigma = []

    for f in range(num_freqs):
        order, modes = shd_num_degrees_vector(max_orders[f])
        Y_f = spspec.sph_harm(modes[None,:], order[None,:], angles[:,0:1], angles[:,1:2])
        B_f = spspec.spherical_jn(order[None,:], k[f]*r[:,None])

        D_f = spspec.spherical_jn(order, k[f]*r_max)
        S_f = phi[f,:]

        Sigma_f = S_f[:,None] * Y_f * B_f / D_f[None,:]
        Sigma.append(Sigma_f)

    Sigma = np.concatenate(Sigma, axis=-1)

    a, residue, rank, singular_values = np.linalg.lstsq(Sigma, p, rcond=None)


    # ======= Reconstruction of RIR =======
    rir_est = np.zeros((num_real_freqs, num_eval), dtype=complex)
    r_eval, angles_eval = util.cart2spherical(pos_eval)
    ord_idx = 0
    for f in range(num_real_freqs):
        order, modes = shd_num_degrees_vector(max_orders[f])
        num_ord = len(order)
        
        j_denom = spspec.spherical_jn(order, k[f]*r_max)
        j_num = spspec.spherical_jn(order[None,:], k[f]*r_eval[:,None])

        Y = spspec.sph_harm(modes[None,:], order[None,:], angles_eval[:,0:1], angles_eval[:,1:2])
        rir_est[f, :] = np.sum(a[None,ord_idx:ord_idx+num_ord] * Y * j_num / j_denom[None,:], axis=-1)
        ord_idx += num_ord

    if verbose:
        diagnostics = {}
        diagnostics["residue"] = residue.tolist()
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
    """
    Might not correspond to the definition in the paper

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
            exp_vec = fd.idft_vector(f, B)
            Phi[f,n] = np.sum(exp_vec * seq_vec) 
    # Fast version, not sure if correct
    # for n in range(B):
    #     Phi[:,n] = np.fft.fft(np.roll(sequence, -n), axis=-1)
    return Phi





def shd_num_degrees(max_order : int):
    """
    Returns a list of mode indices for each order
    when order = n, the degrees are only non-zero for -n <= degree <= n

    Parameters
    ----------
    max_order : int
        is the maximum order that is included

    Returns
    -------
    degree : list of ndarrays of shape (2*order+1)
        so the ndarrays will grow larger for higher list indices
    """
    degree = []
    for n in range(max_order+1):
        pos_degrees = np.arange(n+1)
        degree_n = np.concatenate((-np.flip(pos_degrees[1:]), pos_degrees))
        degree.append(degree_n)
    return degree

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
        pos_degrees = np.arange(n+1)
        degree_n = np.concatenate((-np.flip(pos_degrees[1:]), pos_degrees))
        degree.append(degree_n)

        order.append(n*np.ones_like(degree_n))
    degree = np.concatenate(degree)
    order = np.concatenate(order)
    return order, degree

def shd_min_order(wavenumber, radius):
    """
    Returns the minimum order of the spherical harmonics that should be used
    for a given wavenumber and radius

    Here according to the definition in Katzberg et al., Spherical harmonic
    representation for dynamic sound-field measurements

    Parameters
    ----------
    wavenumber : ndarray of shape (num_freqs)
    radius : float
        represents r_max in the Katzberg paper

    Returns
    -------
    M_f : ndarray of shape (num_freqs)
        contains an integer which is the minimum order of the spherical harmonics
    """
    return np.ceil(wavenumber * radius).astype(int)







