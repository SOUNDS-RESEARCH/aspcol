import numpy as np
import scipy.linalg as splin

import aspcore.matrices as mat


def solve_power_weighted_qos_uplink(R, noise_pow, sinr_targets, max_pow, audio_cov, tolerance=1e-12, max_iters=20, verbose=False):
    """ Calculates a control filter and power allocation that solves the signal-weighted uplink Quality of Service problem in the time-domain
    
    The control filter is SINR-optimal and takes the spectral characteristics of the audio signal into account. In order to use it 
    for sound zone control it should first be power-normalized with normalize_beamformer, and then scaled according to power_alloc_qos_downlink
    It is the proposed method of [brunnstromSignaltointerferenceplusnoise2023]. 
    
    Parameters
    ----------
    R : ndarray of shape (num_zones, num_zones, bf_len, bf_len)
        R[z,i,:,:] is R_{zi} from the paper below, can be seen in eq. (5)
        The first axis is the room impulse responses from loudspeakers to zone z
        The second axis is the desired audio signal for zone i 
    noise_power : ndarray of shape (num_zones,)
    sinr_targets : ndarray of shape (num_zones,)
    max_pow : float
    audio_cov : ndarray of shape (num_zones, bf_len, bf_len)
        defined in the paper below right above (7) or in equation (8)
    tolerance : float
        The relative mean square difference between two subsequent iterations at which 
        the algorithm is considered to have converged. 
    max_iters : int
        The number of iterations after which the algorithm terminates regardless of convergence.
        For infeasible max_pow and/or sinr_targets, the algorithm will never converge. 
        For feasible parameters, depending on the tolerance, the algorithm generally converges in 3-7 iterations

    Returns
    -------
    w : ndarray of shape (num_zones, bf_len)
        the SINR-optimal control filter
    q : ndarray of shape (num_zones,)
        the uplink power allocation
    
    References
    ----------
    [brunnstromSignaltointerferenceplusnoise2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Signal-to-interference-plus-noise ratio based optimization for sound zone control,” IEEE Open Journal of Signal Processing, vol. 4, pp. 257–266, 2023, doi: 10.1109/OJSP.2023.3246398.
    """
    num_zones = R.shape[0]

    q = np.zeros((num_zones))
    n = 0
    beamformers = []
    power_vectors = []

    is_feasible = False
    while True:
        if verbose:
            print(f"Iter {n}")
        w = _beamformer_minmax_weighted_uplink(q, R, audio_cov)
        w = normalize_beamformer(w)
        s = np.squeeze(np.array([w[k,:,None].T @ audio_cov[k,:,:] @ w[k,:,None] for k in range(num_zones)]))
        if is_feasible:
            q = _power_alloc_qos_uplink(w, R, s, sinr_targets)
        else:
            q, c = _power_alloc_minmax_uplink(w, R, s, sinr_targets, max_pow, verbose=verbose)
            if verbose:
                print(f"capacity: {c}")
            
            if _power_alloc_qos_feasibility_spectral_radius(_link_gain_uplink(w, R), sinr_targets) < 1:
                is_feasible = True
            
        beamformers.append(w)
        power_vectors.append(q)
        sinr_balance_diff = _sinr_balance_difference_uplink(q, w, R, noise_pow, sinr_targets)
        if verbose:
            print(f"SINR balance difference: {sinr_balance_diff}")
            print(f"Total power: {np.sum(q)}")
            print(f"Uplink feasibility spectral radius: {_power_alloc_qos_feasibility_spectral_radius(_link_gain_uplink(w, R), sinr_targets)}")
            print(f"Downlink feasibility spectral radius: {_power_alloc_qos_feasibility_spectral_radius(_link_gain_downlink(w, R), sinr_targets)}")
        if n >= 1:
            pow_diff = np.mean(np.abs(power_vectors[-1] - power_vectors[-2])**2)
            bf_diff = np.mean(np.abs(beamformers[-1] - beamformers[-2])**2)
            if verbose:
                print(f"Power mean square difference: {pow_diff}")
                print(f"Beamformer mean square difference: {bf_diff}")
            
            if (is_feasible and pow_diff < tolerance and bf_diff < tolerance) or n == max_iters:
                break
        n += 1
    return w, q

def sum_pow_of_mat(bf_mat):
    """
    bf_mat is the beamformer matrix obtained from semidefinite relaxation, 
        of shape (num_freqs, num_zones, num_sources, num_sources)

    return the sum of traces for each frequency, so an array of shape 
        (num_freqs,)
    """
    return np.real_if_close(np.sum(np.trace(bf_mat, axis1=-2, axis2=-1), axis=-1))

def sum_pow_of_vec(bf_vec):
    """

    Parameters
    ----------
    bf_vec : ndarray of shape (num_freqs, num_zones, num_sources)
        complex or real beamformer vector

    Returns
    -------
    The sum power for each frequency, so an array
        of shape (num_freqs,)
    
    """
    return np.sum(np.abs(bf_vec)**2, axis=(-2, -1))

def sum_weighted_pow(bf_vec, weighting_mat):
    """

    Parameters
    ----------
    bf_vec is complex or real beamformer vector of shape
        (num_zones, bf_len)
    weighting_mat is shape (num_zones, bf_len, bf_len)

    Returns
    -------
    Weighted sum power : float    
    """
    num_zones = bf_vec.shape[0]
    return np.sum([np.squeeze(bf_vec[k,:,None].T @ weighting_mat[k,:,:] @ bf_vec[k,:,None]) for k in range(num_zones)])

def _select_solution_eigenvalue(opt_mat, verbose=False):
    """
    opt_mat is of shape (..., beamformer_len, beamformer_len)

    returns vector of shape (..., beamformer_len)
    
    """
    assert opt_mat.shape[-2] == opt_mat.shape[-1]
    return mat.broadcast_func(opt_mat, _select_solution_eigenvalue_single_freq, out_shape=(opt_mat.shape[-1],), dtype=opt_mat.dtype, verbose=verbose)

def _select_solution_eigenvalue_single_freq(opt_mat, verbose=False):
    ev, evec = splin.eigh(opt_mat)
    if verbose:
        if ev[-1] / ev[-2] < 1e6:
                print(f"Ratio between 1st and 2nd eigval is {ev[-1] / ev[-2]}")
    return evec[:,-1] * np.sqrt(ev[-1])


def normalize_beamformer(w):
    """
    Normalizes the beamformer vector w_k
        so that each vector is length one, meaning ||w_k||_2 == 1
    """
    #w_norm = np.zeros_like(w)
    norm = splin.norm(w, axis=-1)
    w_norm = w / norm[...,None]
    return w_norm

def normalize_system(R, noise_pow):
    """
    Normalize spatial covariance matrix by noise powers to get
        unity noise power for all zones. Does not change the 
        downlink SINR, but does change uplink SINR. 

    Is desirable to use because strong duality is not guaranteed 
        when we have non-unity noise powers. 

    returns normalized_R, normalized_noise_pow
    """
    num_zones = noise_pow.shape[-1]
    assert noise_pow.ndim == 1
    assert len(noise_pow) == num_zones
    R_normalized = np.zeros_like(R)
    
    for k in range(num_zones):
        R_normalized[k,...] = R[k,...] / noise_pow[k]

    return R_normalized, np.ones_like(noise_pow)

def apply_power_vec(w, p):
    """
    w is (num_freq, num_zones, num_sources), or (num_zones, bf_len)
    """
    return np.sqrt(p[...,None]) * w

def extract_power_vec(w):
    """
    Opposite of apply_power_vec
    This function takes a non-unit-vector w and returns a unit vector w_norm
    along with the power p such that apply_power_vec(w_norm, p) = w

    w is (num_freq, num_zones, num_sources), or (num_zones, bf_len)
    """
    norm = splin.norm(w, axis=-1)
    w_norm = w / norm[...,None]
    return w_norm, norm**2

def _is_unit_vector(w):
    return np.allclose(np.linalg.norm(w, axis=-1), 1)

def _link_gain_downlink(w, R):
    return _link_gain(w, R)

def _link_gain_uplink(w, R):
    return _link_gain(w, R).T

def _link_gain(w, R):
    """
    R is (num_zones, bf_len, bf_len) or (num_zones, num_zones, bf_len, bf_len)
        if R.ndim == 4, the second index is the audio signal index. 

        returns the matrix G as defined in 
        'A general duality theory for uplink and downlink beamforming'
        which is defined as G_ik = w_k^T R_ik w_k
    """
    num_zones = w.shape[0]
    G = np.zeros((num_zones, num_zones))

    if R.ndim == 4:
        for k in range(num_zones):
            for i in range(num_zones):
                G[i,k] = np.real_if_close(np.squeeze(w[k,:,None].T.conj() @ R[i,k,:,:] @ w[k,:,None]))
    if R.ndim == 3:
        for k in range(num_zones):
            for i in range(num_zones):
                G[i,k] = np.real_if_close(np.squeeze(w[k,:,None].T.conj() @ R[i,:,:] @ w[k,:,None]))
    return G


def _sinr_downlink(p, w, R, noise_pow):
    assert _is_unit_vector(w)
    return _sinr(p, _link_gain_downlink(w, R), noise_pow)

def _sinr_uplink(p, w, R, noise_pow):
    assert _is_unit_vector(w)
    return _sinr(p, _link_gain_uplink(w, R), noise_pow)

def _sinr_margin_downlink(p, w, R, noise_pow, sinr_targets):
    return _sinr_downlink(p, w, R, noise_pow) - sinr_targets

def _sinr_margin_uplink(p, w, R, noise_pow, sinr_targets):
    return _sinr_uplink(p, w, R, noise_pow) - sinr_targets

def _sinr(p, gain_mat, noise_pow):
    """
    if gain_mat is obtained from the function link_gain
        then gain_mat = link_gain() matches the downlink SINR
        and gain_mat = link_gain().T matches the uplink SINR

    gain_mat is of shape (num_zones, num_zones)
    noise_pow is of shape (num_zones)

    returns a SINR for each zone, an array of shape (num_zones,)
    """
    assert p.ndim == 1
    assert gain_mat.ndim == 2
    assert gain_mat.shape[0] == gain_mat.shape[1]
    num_zones = gain_mat.shape[0]
    assert len(p) == num_zones
    assert len(noise_pow) == num_zones

    gain_mat_scaled = gain_mat @ np.diag(p)
    interference = _sum_interference(_interference_matrix(gain_mat_scaled))

    sinr_val = np.zeros((num_zones))
    for k in range(num_zones):
        sinr_val[k] = gain_mat_scaled[k,k] / (interference[k] + noise_pow[k])
    return sinr_val

def _sum_interference(interference_mat):
    return np.sum(interference_mat, axis=-1)
        
    
def power_alloc_qos_downlink(w, R, noise_pow, sinr_targets):
    assert _is_unit_vector(w)
    return _power_alloc_qos(_link_gain_downlink(w, R), noise_pow, sinr_targets)

def _power_alloc_qos_uplink(w, R, noise_pow, sinr_targets):
    assert _is_unit_vector(w)
    return _power_alloc_qos(_link_gain_uplink(w, R), noise_pow, sinr_targets)

def _power_alloc_qos(gain_mat, noise_pow, sinr_targets):
    """
    Minimizes the power when the SINR equals the sinr_targets

    Closed form derivations and proofs given in 
        'A general duality theory for uplink and downlink beamforming'
    
    """
    assert _power_alloc_qos_is_feasible(gain_mat, sinr_targets)
    if_mat = _interference_matrix(gain_mat)
    sig_mat = _signal_diag_matrix(gain_mat, sinr_targets)

    system_mat = np.eye(gain_mat.shape[-1]) - sig_mat @ if_mat
    answer_mat = sig_mat @ noise_pow[:,None]
    p = splin.solve(system_mat, answer_mat)

    assert np.all(p > 0)
    return p[:,0]

def _power_alloc_qos_is_feasible(gain_mat, sinr_targets):
    """
    Derivations in 'A General Duality Theory for Uplink and Downlink Beamforming'
    """
    return _power_alloc_qos_feasibility_spectral_radius(gain_mat, sinr_targets) < 1

def _power_alloc_qos_feasibility_spectral_radius(gain_mat, sinr_targets):
    """
    Derivations in 'A General Duality Theory for Uplink and Downlink Beamforming'
    """
    if_mat = _interference_matrix(gain_mat)
    sig_mat = _signal_diag_matrix(gain_mat, sinr_targets)
    ev = splin.eigvals(sig_mat @ if_mat)
    spectral_radius = np.max(np.abs(ev))
    return spectral_radius

def _power_alloc_minmax_downlink(w, R, noise_pow, sinr_targets, max_pow, verbose=False):
    assert _is_unit_vector(w)
    return _power_alloc_minmax(_link_gain_downlink(w, R), noise_pow, sinr_targets, max_pow, verbose=verbose)

def _power_alloc_minmax_uplink(w, R, noise_pow, sinr_targets, max_pow, verbose=False):
    assert _is_unit_vector(w)
    return _power_alloc_minmax(_link_gain_uplink(w, R), noise_pow, sinr_targets, max_pow, verbose=verbose)

def _power_alloc_minmax(gain_mat, noise_pow, sinr_targets, max_pow, verbose=False):
    """
    Tranpose the gain_mat to shift between the downlink and uplink solution
    
    """
    if_mat = _interference_matrix(gain_mat)
    D = _signal_diag_matrix(gain_mat, sinr_targets)
    coupling_mat = _extended_coupling_matrix_downlink(D, if_mat, noise_pow, max_pow)
    eigvals, eigvec = splin.eig(coupling_mat)
    #print(eigvals)

    max_ev_idx = np.argmax(np.real(eigvals))
    pow_vec = np.real_if_close(eigvec[:-1, max_ev_idx] / eigvec[-1, max_ev_idx])
    assert np.all(pow_vec > 0) #the dominant eigenvector should be positive
    if verbose:
        if not np.all([eigvals[i] <= 1e-10 for i in range(len(eigvals)) if i != max_ev_idx]): #only one eigenvalue should be positive (not sure if this should be true actually)
            print(f"Warning: Not only one eigenvalue is positive: {eigvals}")
    capacity = np.real_if_close(1 / eigvals[max_ev_idx])
    return pow_vec, capacity

def _interference_matrix(gain_mat):
    """
    The inference matrix is the link gain but with diagonal elements
        set to zero. 
        
    According to the definition written at the function 
        link_gain, to get the sum interference for a zone k, you take 
        np.sum(if_mat, axis=0)[k] (to be verified)
    """
    if_mat = np.zeros_like(gain_mat)
    if_mat[...] = gain_mat
    np.fill_diagonal(if_mat, 0)
    return if_mat

def _signal_diag_matrix(gain_mat, sinr_targets):
    return np.diag(sinr_targets / np.diag(gain_mat))

def _extended_coupling_matrix_downlink(signal_mat, if_mat, noise_pow, max_pow):
    """
    See documentation for power_assignment_minmax_downlink
    Returns equation (12) from Schubert & Boche
    """
    num_zones = signal_mat.shape[0]
    ext_dim = num_zones + 1

    noise_pow = noise_pow[:,None]           #make into column vector
    max_pow_vec = (1 / max_pow) * np.ones((1, num_zones))

    coupling_mat = np.zeros((ext_dim, ext_dim))
    coupling_mat[:-1, :-1] = signal_mat @ if_mat    #top left block
    coupling_mat[:-1, -1:] = signal_mat @ noise_pow                         #top right block
    coupling_mat[-1:, :-1] = max_pow_vec @ signal_mat @ if_mat                      #bottom left block
    coupling_mat[-1:, -1:] = max_pow_vec @ signal_mat @ noise_pow                          #bottom right block
    return coupling_mat

def _sinr_balance_difference_downlink(p, w, R, noise_pow, sinr_targets):
    return _sinr_balance_difference(p, _link_gain_downlink(w, R), noise_pow, sinr_targets)

def _sinr_balance_difference_uplink(p, w, R, noise_pow, sinr_targets):
    return _sinr_balance_difference(p, _link_gain_uplink(w, R), noise_pow, sinr_targets)

def _sinr_balance_difference(p, gain_mat, noise_pow, sinr_targets):
    sinr_val = _sinr(p, gain_mat, noise_pow)
    sinr_ratio  = sinr_targets / sinr_val
    return np.max(sinr_ratio) - np.min(sinr_ratio)

def _beamformer_minmax_weighted_uplink(q, R, audio_cov):
    num_zones = R.shape[0]
    bf_len = R.shape[-1]

    w = np.zeros((num_zones, bf_len), dtype=R.dtype)
    Q = np.zeros((bf_len, bf_len), dtype=R.dtype)

    for k in range(num_zones):
        Q.fill(0)
        Q += audio_cov[k,:,:]
        for i in range(num_zones):
            if k != i:
                if R.ndim == 3:
                    Q += q[i] * R[i,:,:]
                elif R.ndim == 4:
                    Q += q[i] * R[i,k,:,:]
        if R.ndim == 3:
            eigval, eigvec = splin.eigh(R[k,:,:], Q)
        elif R.ndim == 4:
            eigval, eigvec = splin.eigh(R[k,k,:,:], Q)
        w[k,:] = eigvec[:,-1]
    return w