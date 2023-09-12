import numpy as np
import scipy.linalg as splin
import scipy.spatial.distance as spdist
import scipy.special as spspec

import aspcol.kernelinterpolation as ki
import aspcol.utilities as util
import aspcol.filterdesign as fd
import est_sf
import pseq


def pseq_nearest_neighbour(p, seq, pos, pos_eval):
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

    p : ndarray of shape (M, seq_len)
        sound pressure in time domain at M microphone positions
    seq : ndarray of shape (seq_len)
        the training signal used for the measurements
        seq should be defined so that the following holds
        p(n) = sum_{i=0}^{I-1} h(i) seq[n-i]
    """
    rir = pseq.decorrelate(p, seq)

    fft_len = rir.shape[-1]
    rir_freq = np.fft.rfft(rir, axis=-1).T
    k = fd.get_real_wavenum(fft_len, samplerate, c)

    return est_ki_diffuse_freq(rir_freq, pos, pos_eval, k, reg_param)

def est_ki_diffuse_freq(p_freq, pos, pos_eval, k, reg_param):
    """
    Parameters
    ----------
    p_freq : ndarray of shape ()
    array : ArrayCollection
    k : ndarray of shape (num_freq)
    
    """
    est_filt = ki.get_krr_parameters(ki.kernel_helmholtz_3d, reg_param, pos_eval, pos, k)
    p_ki = est_filt @ p_freq[:,:,None]
    return np.squeeze(p_ki, axis=-1)




def est_shd_dynamic_conjugate_symmetric(p, pos, pos_eval, sequence, samplerate, c, reg_param, verbose=False):
    """
    more correct than the standard version
    Parameters
    ----------
    p : ndarray of shape (M, B)
    pos : ndarray of shape (N, 3)
    sequence : ndarray of shape (B)
    sim_info : SimulatorInfo object
    """
    M = p.shape[0]
    B = p.shape[1]
    N = B * M

    p_vec = p.reshape(-1)

    k = fd.get_wavenum(B, samplerate, c)
    real_freqs = fd.get_real_freqs(B, samplerate)
    num_real_freqs = len(real_freqs)

    Phi = sequence_stft_multiperiod(sequence[:B], M)

    #division by pi is a correction for the sinc function
    dist_mat = np.sqrt(np.sum((np.expand_dims(pos,1) - np.expand_dims(pos,0))**2, axis=-1))  / np.pi 
    
    psi = np.zeros((N, N), dtype = complex)

    psi += np.sinc(dist_mat * k[0]) * Phi[0,:,None] * Phi[0,None,:].conj()
    psi += np.sinc(dist_mat * k[B//2]) * Phi[B//2,:,None] * Phi[B//2,None,:].conj()
    assert B % 2 == 0 #last two lines is only correct if B is even

    for f in range(1, num_real_freqs-1):
        phi_rank1_matrix = Phi[f,:,None] * Phi[f,None,:].conj()
        psi += 2*np.real(np.sinc(dist_mat * k[f]) * phi_rank1_matrix)

    noise_cov = reg_param * np.eye(N)
    #right_side = np.linalg.solve(psi + noise_cov, p_vec)
    right_side = splin.solve(psi + noise_cov, p_vec, assume_a = "pos")

    right_side = Phi.conj() * right_side[None,:]

    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for f in range(num_real_freqs):
        #if f != 0 :
        #print(f"Freq: {f}")
        
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





def est_shd_dynamic(p, pos, pos_eval, sequence, samplerate, c, reg_param, verbose=False):
    """
    probably wrong, should not sample all frequencies from 0 to I

    Parameters
    ----------
    p : ndarray of shape (M, B)
    pos : ndarray of shape (N, 3)
    sequence : ndarray of shape (B)
    sim_info : SimulatorInfo object
    """
    M = p.shape[0]
    B = p.shape[1]
    N = B * M

    p_vec = p.reshape(-1)

    k = fd.get_wavenum(B, samplerate, c)
    real_freqs = fd.get_real_freqs(B, samplerate)
    num_real_freqs = len(real_freqs)

    Phi = sequence_stft_multiperiod(sequence[:B], M)

    #division by pi is a correction for the sinc function
    dist_mat = np.sqrt(np.sum((np.expand_dims(pos,1) - np.expand_dims(pos,0))**2, axis=-1))  / np.pi 
    
    psi = np.zeros((N, N), dtype = complex)
    for f in range(B):
        phi_rank1_matrix = Phi[f,:,None] * Phi[f,None,:].conj()
        psi += np.sinc(dist_mat * k[f]) * phi_rank1_matrix

    noise_cov = reg_param * np.eye(N)
    #right_side = np.linalg.solve(psi + noise_cov, p_vec)
    right_side = splin.solve(psi + noise_cov, p_vec, assume_a = "pos")

    right_side = Phi.conj() * right_side[None,:]

    est_sound_pressure = np.zeros((num_real_freqs, pos_eval.shape[0]), dtype=complex)
    for f in range(num_real_freqs):
        #if f != 0 :
        #print(f"Freq: {f}")
        
        kernel_val = ki.kernel_helmholtz_3d(pos_eval, pos, k[f:f+1]).astype(complex)[0,:,:]
        est_sound_pressure[f, :] = np.sum(kernel_val * right_side[f,None,:], axis=-1)

    if verbose:
        diagnostics = {}
        diagnostics["condition number"] = np.linalg.cond(psi).tolist()
        diagnostics["smallest eigenvalue"] = splin.eigh(psi, subset_by_index=(0,0), eigvals_only=True).tolist()
        diagnostics["largest eigenvalue"] = splin.eigh(psi, subset_by_index=(N-1, N-1), eigvals_only=True).tolist()
        return est_sound_pressure, diagnostics
    else:
        return est_sound_pressure





def est_shd_dynamic_pressure_ip(p, pos, pos_eval, sequence, samplerate, c, reg_param):
    """
    First estimate sound pressure, and then decorrelate the pseq at each position. 


    Parameters
    ----------
    p : ndarray of shape (M, B)
    pos : ndarray of shape (N, 3)
    sequence : ndarray of shape (B)
    sim_info : SimulatorInfo object
    """
    est_freq = est_sf.est_shd_dynamic_conjugate_symmetric(p, pos, pos_eval, samplerate, c, reg_param)
    est_time = np.fft.irfft(est_freq, axis=0).T
    rir_est = pseq.decorrelate(est_time, sequence[:est_time.shape[-1]])
    return np.fft.rfft(rir_est, axis=-1).T





def est_spatial_spectrum_dynamic(p, pos, pos_eval, sequence, samplerate, c, r_max, verbose=False):
    """
    Method from Katzberg et al. 
    SPHERICAL HARMONIC REPRESENTATION FOR DYNAMIC SOUND-FIELD MEASUREMENTS
    
    Assumes that the spherical harmonics should be expanded around the origin (0,0,0)
    
    Parameters
    ----------
    p : ndarray of shape (M, B)
    pos : ndarray of shape (N, 3)
    pos_eval : ndarray of shape (num_eval, 3)
    sequence : ndarray of shape (B)
        Assumes the sequence is periodic
    """
    M = p.shape[0]
    B = p.shape[1]
    N = B * M
    num_eval = pos_eval.shape[0]

    p = p.reshape(-1)


    k = fd.get_wavenum(B, samplerate, c)
    num_freqs = len(k)
    real_freqs = fd.get_real_freqs(B, samplerate)
    num_real_freqs = len(real_freqs)

    phi = sequence_stft_multiperiod(sequence[:B], M)
    max_orders = min_order_spher_harm(k, r_max)
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

    a, residue, rank, singular_values = np.linalg.lstsq(Sigma + 0*np.eye(*Sigma.shape), p, rcond=None)


    # reconstruction
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
    #assert ord_idx == len(a)

    if verbose:
        diagnostics = {}
        diagnostics["residue"] = residue.tolist()
        diagnostics["condition number"] = np.linalg.cond(Sigma).tolist()
        diagnostics["smallest singular value"] = splin.svdvals(Sigma)[0].tolist()
        diagnostics["largest singular"] = splin.svdvals(Sigma)[-1].tolist()
        diagnostics["r_max"] = r_max
        return rir_est, diagnostics
    return rir_est


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

def min_order_spher_harm(wavenumber, radius):
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



def sequence_stft_multiperiod(sequence, num_periods):
    """
    Assumes that the sequence is periodic.
    Assumes that sequence argument only contains one period
    
    Returns
    -------
    Phi : ndarray of shape (seq_len, num_periods*seq_len)
    """
    Phi = sequence_stft(sequence)
    return np.tile(Phi, (1, num_periods))

def sequence_stft(sequence):
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





