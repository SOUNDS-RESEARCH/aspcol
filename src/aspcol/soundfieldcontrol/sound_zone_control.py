"""Algorithms for sound field control, in particular sound zone control. 


References
----------
[brunnstromSound2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Sound zone control for arbitrary sound field reproduction methods,” in European Signal Processing Conference (EUSIPCO), Helsinki, Finland, Sep. 2023. `[link] <https://doi.org/10.23919/EUSIPCO58844.2023.10289995>`__ \n
[brunnstromSignaltointerferenceplusnoise2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Signal-to-interference-plus-noise ratio based optimization for sound zone control,” IEEE Open Journal of Signal Processing, vol. 4, pp. 257–266, 2023, doi: 10.1109/OJSP.2023.3246398. `[link] <https://doi.org/10.1109/OJSP.2023.3246398>`__ \n
[leeFast2020] T. Lee, L. Shi, J. K. Nielsen, and M. G. Christensen, “Fast generation of sound zones using variable span trade-off filters in the DFT-domain,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 363–378, Dec. 2020, doi: 10.1109/TASLP.2020.3042701. `[link] <https://doi.org/10.1109/TASLP.2020.3042701>`__ \n 


"""
import numpy as np
import copy
import scipy.linalg as splin

import aspcore.matrices as mat


# ================= CONVENTIONAL SOUND ZONE CONTROL ================
def acc(Rb, Rd, reg=0):
    """Frequency-domain acoustic contrast control for sound zone control

    Will calculate the principal generalized eigenvector of (Rb, Rd+reg*I)

    Parameters
    ----------
    Rb : ndarray of shape (num_freq, num_ls, num_ls)
        spatial covariance associated with the bright zone
    Rd : ndarray of shape (num_freq, num_ls, num_ls)
        spatial covariance associated with the dark zones
    reg : float
        non-negative, applies l2 regularization to the loudspeaker effort, by adding
        reg * np.eye() to the dark zone spatial covariance. 

    Returns
    -------
    w : ndarray of shape (num_freq, num_ls)
        the control filter used to generate loudspeaker signals.
    """
    assert Rb.shape == Rd.shape
    assert Rb.shape[-1] == Rb.shape[-2]
    assert Rb.ndim == 3
    num_freq = Rd.shape[0]
    num_ls = Rd.shape[-1]
    
    if reg > 0:
        Rd_reg = Rd + reg*np.eye(num_ls)[None,:,:]
    else:
        Rd_reg = Rd
    
    w = np.zeros((num_freq, num_ls))
    for f in range(num_freq):
        eigvals, evec = splin.eigh(Rb[f,:,:], mat.ensure_pos_def_adhoc(Rd_reg[f,:,:], verbose=True))
        w[f,:] = evec[:,-1] * np.sqrt(eigvals[-1])
    return w

def pressure_matching_szc(Rb, Rd, rb, mu, reg=0):
    """Frequency-domain pressure matching for sound zone control
    
    Parameters
    ----------
    Rb : ndarray of shape (num_freq, num_ls, num_ls)
        spatial covariance associated with the bright zone
    Rd : ndarray of shape (num_freq, num_ls, num_ls)
        spatial covariance associated with the dark zones
    rb : ndarray of shape (num_freq, num_ls, num_virt_src)
        cross correlation between paths to bright zone and virtual source paths
    mu : int
        non-negative, weights effort between error in dark and bright zones
    reg : float
        non-negative, applies l2 regularization to the loudspeaker effort, by adding
        reg * np.eye() to the dark zone spatial covariance

    Returns
    -------
    w : ndarray of shape (num_freq, num_ls, num_virt_src)
        the control filter used to generate loudspeaker signals.
    """
    assert Rb.shape == Rd.shape
    assert Rb.shape[-1] == Rb.shape[-2]
    assert Rb.ndim == 3
    num_freq = Rd.shape[0]
    num_ls = Rd.shape[-1]
    if reg > 0:
        Rd_reg = Rd + reg*np.eye(num_ls)[None,:,:]
    else:
        Rd_reg = Rd

    w = np.linalg.solve(Rb + mu*Rd_reg, rb)
    return w

def vast(Rb, Rd, rb, mu, rank, reg=0):
    """Frequency domain variable span trade-off filter for sound zone control 

    Parameters
    ----------
    Rb : ndarray of shape (num_freq, num_ls, num_ls)
        spatial covariance associated with the bright zone
    Rd : ndarray of shape (num_freq, num_ls, num_ls)
        spatial covariance associated with the dark zones
    rb : ndarray of shape (num_freq, num_ls, num_virt_src)
        cross correlation between paths to bright zone and virtual source paths
    mu : int 
        non-negative, weights effort between error in dark and bright zones
    rank : int
        between 1 and num_ls. Applies low-rank approximation via GEVD. 
        Lower rank gives higher acoustic contrast, higher rank gives lower distortion
    reg : float
        non-negative, applies l2 regularization to the loudspeaker effort, by adding
        reg * np.eye() to the dark zone spatial covariance

    Returns
    -------
    w : ndarray of shape (num_freq, num_ls, num_virt_src)
        the control filter used to generate loudspeaker signals. 
        loudspeaker signals can be generated from the virtual source signals 
        v : (num_freq, num_virt_src, 1) as w @ v. 

    References
    ----------
    [leeFast2020] T. Lee, L. Shi, J. K. Nielsen, and M. G. Christensen, “Fast generation of sound zones using variable span trade-off filters in the DFT-domain,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 363–378, Dec. 2020, doi: 10.1109/TASLP.2020.3042701.
    """
    assert Rb.shape == Rd.shape
    assert Rb.shape[-1] == Rb.shape[-2]
    assert Rb.ndim == 3
    assert Rb.shape[0:2] == rb.shape[0:2]
    assert rb.ndim == 3
    num_freq = Rb.shape[0]
    num_ls = Rb.shape[-1]

    if reg > 0:
        Rd_reg = Rd + reg*np.eye(num_ls)[None,:,:]
    else:
        Rd_reg = Rd

    eigval = []
    eigvec = []
    for f in range(num_freq):
        eva, eve = splin.eigh(Rb[f,:,:], Rd_reg[f,:,:], type=1)
        eigval.append(eva[None,:])
        eigvec.append(eve[None,:,:])
    eigval = np.concatenate(eigval, axis=0)
    eigvec = np.concatenate(eigvec, axis=0)
    eigval = eigval[:,-rank:]
    eigvec = eigvec[:,:,-rank:]
    diag_entries = 1 / (eigval + mu*np.ones(rank))
    diag_mat = np.concatenate([np.diag(diag_entries[i,:])[None,:,:] for i in range(num_freq)], axis=0)
    w = eigvec @ diag_mat @ np.moveaxis(eigvec.conj(),1,2) @ rb
    return w




def acc_time(Rb, Rd, reg_param=0):
    """Time domain acoustic contrast control
    
    Will calculate the principal generalized eigenvector of (Rb, Rd+reg*I)

    OLD DOCUMENTATION:
    cov_bright and cov_dark is the szc spatial covariance matrices
    = H.T @ H, where H is a convolution matrix made up of the RIR, 
    summed over the appropriate microphones for bright and dark zones. 

    num_ls is the number of loudspeakers, and therefore the number 
    of blocks (in each axis) that the cov matrices consists of
    """
    assert Rb.shape == Rd.shape
    assert Rb.shape[0] == Rd.shape[1]
    assert Rb.ndim == 2
    #Rd += 1e-4*np.eye(Rd.shape[0])
    if reg_param > 0:
        Rd_reg = Rd + reg_param*np.eye(Rd.shape[0])
    else:
        Rd_reg = Rd
    eigvals, evec = splin.eigh(Rb, mat.ensure_pos_def_adhoc(Rd_reg, verbose=True))
    
    #ir = evec[:,-1].reshape(1, num_ls, -1)
    #norm = np.sqrt(np.sum(ir**2))
    #ir /= norm
    #ir *= 1e4
    return evec[:,-1] * np.sqrt(eigvals[-1])

def acc_time_all_zones(R, reg_param=0):
    """Time-domain acoustic contrast control for all zones

    Parameters
    ----------
    R : ndarray of shape (num_zones, num_zones, bf_len, bf_len)
        R[k,i,:,:] means spatial covariance associated with RIRs 
        of zone k, and audio signal of zone i
    
    Returns
    -------
    beamformer vector : ndarray of shape (num_zones, bf_len)
    """
    num_zones = R.shape[0]
    bf_len = R.shape[-1]
    assert R.shape == (num_zones, num_zones, bf_len, bf_len)

    Rd = np.zeros((bf_len, bf_len))
    w = np.zeros((num_zones, bf_len))

    for k in range(num_zones):
        Rd.fill(0)
        Rb = R[k,k,:,:]
        for i in range(num_zones):
            if i != k:
                Rd += R[i,k,:,:]
        w[k,:] = acc_time(Rb, Rd, reg_param)

    return w




# ================== LOUDSPEAKER SIGNAL TRANSFORM ================
def szc_transform_mwf(Rb, Rd, mu=1):
    """Calculates the linear transformation to apply to loudspeaker signals to obtain sound zones. 

    A special case of szc_transform_mwf_gevd, where rank is equal to the number of loudspeakers

    Parameters
    ----------
    Rb : ndarray of shape (num_freqs, num_ls, num_ls)
        spatial covariance associated with the bright zone
    Rd : ndarray of shape (num_freqs, num_ls, num_ls)
        spatial covariance associated with the dark zones
    mu : int
        non-negative, weights effort between error in dark and bright zones

    Returns
    -------
    W : ndarray of shape (num_freqs, num_ls, num_ls)
        The matrix representing the linear transformation of the loudspeaker signals. 
        Calculated as W = (Rb + mu*Rd)^{-1} Rb

    References
    ----------
    [brunnstromSound2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Sound zone control for arbitrary sound field reproduction methods,” in European Signal Processing Conference (EUSIPCO), Helsinki, Finland, Sep. 2023.
    """

    mat_to_invert = Rb + mu*Rd
    W = np.linalg.solve(mat_to_invert, Rb)
    return W

def szc_transform_mwf_gevd(Rb, Rd, rank, mu=1):
    """Calculates the linear transformation to apply to loudspeaker signals to obtain sound zones. 

    Parameters
    ----------
    Rb : ndarray of shape (num_freqs, num_ls, num_ls)
        spatial covariance associated with the bright zone
    Rd : ndarray of shape (num_freqs, num_ls, num_ls)
        spatial covariance associated with the dark zones
    mu : int
        non-negative, weights effort between error in dark and bright zones
    rank : int
        between 1 and num_ls. Applies low-rank approximation via GEVD. 
        Lower rank gives higher acoustic contrast, higher rank gives lower distortion

    Returns
    -------
    W : ndarray of shape (num_freqs, num_ls, num_ls)
        The matrix representing the linear transformation of the loudspeaker signals
    
    References
    ----------
    [brunnstromSound2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Sound zone control for arbitrary sound field reproduction methods,” in European Signal Processing Conference (EUSIPCO), Helsinki, Finland, Sep. 2023.
    """
    num_freqs = Rb.shape[0]
    W = np.zeros(Rb.shape, dtype=Rb.dtype)

    for f in range(num_freqs):
        eigvals, eigvec = splin.eigh(Rb[f,:,:], mu*Rd[f,:,:]) #largest eigenvalue is last
        eigvec = np.flip(eigvec, axis=-1)
        eigvec_invt = splin.inv(eigvec).T.conj()
        eigvals = np.flip(eigvals, axis=-1)
        eigvals = eigvals / (np.ones_like(eigvals) + eigvals)
        for r in range(rank):
            W[f,:,:] += eigvals[r] * eigvec[:,r:r+1] @ eigvec_invt[:,r:r+1].conj().T
    return W

