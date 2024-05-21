"""Algorithms for sound field control, in particular sound zone control. 


References
----------
[brunnstromSound2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Sound zone control for arbitrary sound field reproduction methods,” in European Signal Processing Conference (EUSIPCO), Helsinki, Finland, Sep. 2023. `[link] <https://doi.org/10.23919/EUSIPCO58844.2023.10289995>`_ \n
[brunnstromSignaltointerferenceplusnoise2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Signal-to-interference-plus-noise ratio based optimization for sound zone control,” IEEE Open Journal of Signal Processing, vol. 4, pp. 257–266, 2023, doi: 10.1109/OJSP.2023.3246398. `[link] <https://doi.org/10.1109/OJSP.2023.3246398>`_ \n
[leeFast2020] T. Lee, L. Shi, J. K. Nielsen, and M. G. Christensen, “Fast generation of sound zones using variable span trade-off filters in the DFT-domain,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 363–378, Dec. 2020, doi: 10.1109/TASLP.2020.3042701. `[link] <https://doi.org/10.1109/TASLP.2020.3042701>`_ \n 


"""
import numpy as np
import copy
import scipy.linalg as splin

import aspcol.correlation as cr
import aspcol.matrices as mat
import aspcol.filterclasses as fc
import aspcol.filterdesign as fd

# ================== UTILITIES ==================================
def beamformer_vec_to_ir(bf_vec, num_ls):
    """Transforms a beamformer vector into a shape more appropriate for a filter impulse response

    Parameters
    ----------
    bf_vec : ndarray of shape (num_ls*filt_len, 1) or (num_zones, num_ls*filt_len)

    Returns
    -------
    ir : ndarray of shape (1, num_ls, filt_len) or (num_zones, num_ls, filt_len)
    """
    bf_vec = np.squeeze(bf_vec)
    if bf_vec.ndim == 1:
        return bf_vec.reshape(num_ls, -1)
    elif bf_vec.ndim == 2:
        num_zones = bf_vec.shape[0]
        return bf_vec.reshape(num_zones, num_ls, -1)
    else:
        raise ValueError("Wrong dimensions for bf_vec")


def freq_to_time_beamformer(w, num_freqs):
    """Tranforms a frequency domain beamformer / control filter to a time domain impulse response
    
    Parameters
    ----------
    w : ndarray of shape ()
    num_freqs : int

    Returns
    -------
    ir : ndarray of shape ()
    """
    w = fd.insert_negative_frequencies(w, True)
    ir,_ = fd.fir_from_freqs_window(w, num_freqs-1)
    ir = np.moveaxis(ir, 0,1)
    ir = ir[:1,:,:]
    return ir



def spatial_cov_freq_superpos(Hb, Hd, d=None):
    """Calculates spatial covariance matrices from transfer function matrices

    Assumes a superposition model, so the transfer functions are bright zone and dark zone
    respecively.

    Parameters
    ----------
    Hb : ndarray of shape (num_freq, num_mic_b, num_ls)
        transfer functions from loudspeakers to bright zone
    Hb : ndarray of shape (num_freq, num_mic_d, num_ls)
        transfer functions from loudspeakers to dark zone
    d : ndarray of shape (num_freq, num_mic_b, num_virt_src), optional
        desired sound pressure in the bright zone 

    Returns
    -------
    Rb : ndarray of shape (num_freq, num_ls, num_ls)
        spatial covariance for the bright zone
    Rd : ndarray of shape (num_freq, num_ls, num_ls)
        spatial covariance for the dark zone
    rb : ndarray of shape (num_freq, num_ls, num_virt_src)
        spatial cross-correlation between the desired pressure and the 
        transfer functions between loudspeakers and bright zone. Only 
        returned if d is provided.
    """
    assert Hb.ndim == 3 and Hd.ndim == 3
    assert Hb.shape[0] == Hd.shape[0]
    assert Hb.shape[2] == Hd.shape[2]
    num_micb = Hb.shape[1]
    num_micd = Hd.shape[1]
    Rb = np.moveaxis(Hb.conj(),1,2) @ Hb / num_micb
    Rd = np.moveaxis(Hd.conj(),1,2) @ Hd / num_micd
    
    if d is not None:
        assert d.ndim == 3
        assert d.shape[:2] == (Hb.shape[0], Hb.shape[1])
        rb = np.moveaxis(Hb.conj(),1,2) @ d / num_micb
        return Rb, Rd, rb
    else:
        return Rb, Rd

def fpaths_to_spatial_cov(arrays, fpaths, source_name, zone_names):
    """

    utility function to be used with aspsim package

    - arrays is ArrayCollection object
    - fpaths is the frequency domain RIRs, see function get_fpaths() in this module
    - source name is a string for a source in arrays
    - zone_names is a list of strings to microphones in arrays

    returns a spatial covariance matrix R of shape (num_freqs, num_zones, num_src, num_src)
    """
    num_sources = arrays[source_name].num
    num_freqs = fpaths[source_name][zone_names[0]].shape[0]
    num_zones = len(zone_names)

    H = [fpaths[source_name][zone_name] for zone_name in zone_names]
    R = np.zeros((num_freqs, num_zones, num_sources, num_sources), dtype=complex)
    for k in range(num_zones):
        for f in range(num_freqs):
            R[f,k,:,:] = mat.ensure_pos_semidef(H[k][f,:,:].T.conj() @ H[k][f,:,:])
        num_mics = H[k][f].shape[0]
        R[:,k,:,:] /= num_mics
    return R

def get_fpaths(arrays, num_freqs, samplerate):
    """utility function to be used with aspsim package. Deprecated, and will be removed in future versions.

    returns a dictionary with frequency domain RIRs 
        each entry has shape 
    """
    freqs = fd.get_frequency_values(num_freqs, samplerate)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in arrays.iter_paths():
        fpaths.setdefault(src.name, {})

        path_freq = np.fft.fft(path, n=num_freqs, axis=-1)

        new_axis_order = np.concatenate(
            ([path.ndim - 1], np.arange(path.ndim - 1))
        )
        path_freq = np.transpose(path_freq, new_axis_order)

        fpaths[src.name][mic.name] = np.moveaxis(path_freq,1,2)[:num_real_freqs,...]
    return fpaths, freqs


def paths_to_spatial_cov(arrays, source_name, zone_names, sources, filt_len, num_samples, margin=None):
    """
    utility function to be used with aspsim package. Deprecated, and will be removed in future versions.

    sources should be a list of the audio sources associated with each zone
        naturally the list of zone names and sources should be of the same length

    by default it will use as many samples as possible (only remove rir_len-1 samples 
        in the beginning since they haven't had time to propagate properly). 
        margin can be supplied if a specific number of samples should be removed instead.
        might give questionable result if you set margin to less than rir_len-1.
    

    Returns K^2 spatial covariance matrices R_{ki}, where k is the zones index of 
        the microphones and i is the zone index of the audio signal
        The returned array has shape (num_zones, num_zones, num_ls*ir_len, num_ls*ir_len)
        and is indexed as R_{ki} = R[k,i,:,:]
    """
    num_sources = arrays[source_name].num
    num_zones = len(zone_names)
    assert len(sources) == num_zones

    R = np.zeros((num_zones, num_zones, filt_len*num_sources, filt_len*num_sources), dtype=float)
    for k in range(num_zones):
        for i in range(num_zones):
            R[k,i,:,:] = mat.ensure_pos_semidef(spatial_cov(arrays.paths[source_name][zone_names[k]], sources[i], filt_len, num_samples, margin=margin))
    return R

def paths_to_spatial_cov_delta(arrays, source_name, zone_names, filt_len):
    """
    utility function to be used with aspsim package. Deprecated, and will be removed in future versions.

    See info for paths_to_spatial_cov
    """
    num_sources = arrays[source_name].num
    num_zones = len(zone_names)

    R = np.zeros((num_zones, num_zones, filt_len*num_sources, filt_len*num_sources), dtype=float)
    for k in range(num_zones):
        for i in range(num_zones):
            R[k,i,:,:] = mat.ensure_pos_semidef(spatial_cov_delta(arrays.paths[source_name][zone_names[k]], filt_len))
    return R

def rir_to_szc_cov(rir, ctrlfilt_len):
    """
    Takes a RIR of shape (num_ls, num_mic, ir_len) and
    turns it into the time domain sound zone control spatial
    covariance matrix made up of the blocks R_l1l2 = H_l1^T H_l2, 
    where H_l is a convolution matrix with RIRs associated with
    loudspeaker l

    output is of shape (num_ls*ctrlfilt_len, num_ls*ctrlfilt_len)
    """
    num_ls = rir.shape[0]
    num_mics = rir.shape[1]
    R = np.zeros((ctrlfilt_len*num_ls, ctrlfilt_len*num_ls))
    for m in range(num_mics):
        for l1 in range(num_ls):
            for l2 in range(num_ls):
                h1 = rir[l1,m,:]
                h2 = rir[l2,m,:]
                H1 = splin.convolution_matrix(h1, ctrlfilt_len ,mode="full")
                H2 = splin.convolution_matrix(h2, ctrlfilt_len ,mode="full")
                R[l1*ctrlfilt_len:(l1+1)*ctrlfilt_len, 
                    l2*ctrlfilt_len:(l2+1)*ctrlfilt_len] += H1.T @ H2
    R /= num_mics
    return R


def spatial_cov(ir, source, filt_len, num_samples, margin=None):
    """
    ir is the room impulse responses of shape (num_ls, num_mic, ir_len)
        from all loudspeakers to one of the zones. 

    source is a source object with get_samples(num_samples) method, which returns
        the audio signal that should be reproduced in the sound zones

    by default it will use as many samples as possible (only remove rir_len-1 samples 
        in the beginning since they haven't had time to propagate properly). 
        margin can be supplied if a specific number of samples should be removed instead.
        might give questionable result if you set margin to less than rir_len-1.
    
    The returned spatial covariance matrix is of size (num_ls*filt_len, num_ls*filt_len)

    """
    ir_len = ir.shape[-1]
    num_sources = ir.shape[0]
    if margin is None:
        margin = ir_len - 1

    rir_filt = fc.create_filter(ir=ir, sum_over_input=False)
    source = copy.deepcopy(source)
    in_sig = source.get_samples(num_samples+margin)
    in_sig = np.tile(in_sig, (num_sources, 1))
    out_sig = rir_filt.process(in_sig)
    out_sig = out_sig[...,margin:]
    out_sig = np.moveaxis(out_sig, 0, 1)
    R = cr.corr_matrix(out_sig, out_sig, filt_len, filt_len)
    return R


def spatial_cov_delta(ir, filt_len):
    """
    Calculates the spatial covariance matrices as if the input signal is a delta
    ir is the default shape given by arrays.paths (num_ls, num_mics, ir_len)
    
    Multiplies result with ir_len, because corr_matrix divides with the number of samples,
        but here that shouldn't be done to keep the correct scaling, as the values are just
        the ir and not filtered samples
        
    The returned spatial covariance matrix is of size (num_ls*filt_len, num_ls*filt_len)
    """
    ir_len = ir.shape[-1]
    ir = np.moveaxis(ir, 1, 0)
    R = cr.corr_matrix(ir, ir, filt_len, filt_len) * ir_len
    return R



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


# ================== SPATIAL AUDIO ===============================
def pressure_matching_sa(Hv, Hc):
    """Frequency domain pressure matching to generate loudspeaker signals from virtual source signals

    v is the virtual source sound pressure (num_freq, num_virtual_src, 1)
    
    Parameters
    ----------
    Hv : ndarray of shape (num_freq, num_mic, num_virtual_src)
        transfer functions from virtual sources to control points 
    Hc : ndarray of shape (num_freq, num_mic, num_ls)
        transfer functions from loudspeakers to control points 

    Returns
    -------
    beamformer w : ndarray of shape (num_freq, num_ls, num_virtual_source) 
        which should be applied to the virutal source sound pressure 
        as w @ v, where v : (num_freq, num_virtual_src, 1)

    Notes
    -----
    The beamformer is calculated as w = (H_c^H H_c)^{-1} H_c^H H_v. Definition can be found in (15) in [brunnstromSound2023]

    References
    ----------
    [brunnstromSound2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Sound zone control for arbitrary sound field reproduction methods,” in European Signal Processing Conference (EUSIPCO), Helsinki, Finland, Sep. 2023.
    """
    w = np.linalg.solve(np.moveaxis(Hc.conj(),-1,-2) @ Hc, np.moveaxis(Hc.conj(),-1,-2) @ Hv)
    return w

class PressureMatchingWOLA:
    """
    Sound field reproduction with pressure matching in the WOLA domain
    
    """
    def __init__(self, audio_src, Hv, Hc, block_size):
        self.ctrl_freq = self.update(Hv, Hc)
        self.audio_src = audio_src
        self.block_size = block_size

        self.wola = fc.WOLA(self.audio_src.num_channels, 1, 2*self.block_size, block_size)

        self.num_real_freq = Hv.shape[0]
        assert Hc.shape[0] == self.num_real_freq
        
    def get_samples(self):
        audio = self.audio_src.get_samples(self.block_size)
        self.wola.analysis(audio)
        #audio_freq = np.moveaxis(np.fft.rfft(audio, axis=-1), -1, 0)
        ls_sig = self.ctrl_freq @ np.moveaxis(self.wola.spectrum, -1, 0)
        return ls_sig

    def update(self, Hv, Hc):
        self.ctrl_freq = pressure_matching_sa(Hv, Hc)







# ==================== SINR BASED SOUND ZONE CONTROL =============================
def solve_power_weighted_qos_uplink(R, noise_pow, sinr_targets, max_pow, audio_cov, tolerance=1e-12, max_iters=20):
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
        print(f"Iter {n}")
        w = _beamformer_minmax_weighted_uplink(q, R, audio_cov)
        w = normalize_beamformer(w)
        s = np.squeeze(np.array([w[k,:,None].T @ audio_cov[k,:,:] @ w[k,:,None] for k in range(num_zones)]))
        if is_feasible:
            q = power_alloc_qos_uplink(w, R, s, sinr_targets)
        else:
            q, c = power_alloc_minmax_uplink(w, R, s, sinr_targets, max_pow)
            print(f"capacity: {c}")
            
            if _power_alloc_qos_feasibility_spectral_radius(link_gain_uplink(w, R), sinr_targets) < 1:
                is_feasible = True
            
        beamformers.append(w)
        power_vectors.append(q)
        sinr_balance_diff = sinr_balance_difference_uplink(q, w, R, noise_pow, sinr_targets)
        print(f"SINR balance difference: {sinr_balance_diff}")
        print(f"Total power: {np.sum(q)}")
        print(f"Uplink feasibility spectral radius: {_power_alloc_qos_feasibility_spectral_radius(link_gain_uplink(w, R), sinr_targets)}")
        print(f"Downlink feasibility spectral radius: {_power_alloc_qos_feasibility_spectral_radius(link_gain_downlink(w, R), sinr_targets)}")
        if n >= 1:
            pow_diff = np.mean(np.abs(power_vectors[-1] - power_vectors[-2])**2)
            bf_diff = np.mean(np.abs(beamformers[-1] - beamformers[-2])**2)
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

def select_solution_eigenvalue(opt_mat, verbose=False):
    """
    opt_mat is of shape (..., beamformer_len, beamformer_len)

    returns vector of shape (..., beamformer_len)
    
    """
    assert opt_mat.shape[-2] == opt_mat.shape[-1]
    return mat.broadcast_func(opt_mat, _select_solution_eigenvalue, out_shape=(opt_mat.shape[-1],), dtype=opt_mat.dtype, verbose=verbose)

def _select_solution_eigenvalue(opt_mat, verbose=False):
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

def link_gain_downlink(w, R):
    return _link_gain(w, R)

def link_gain_uplink(w, R):
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


def sinr_downlink(p, w, R, noise_pow):
    assert _is_unit_vector(w)
    return _sinr(p, link_gain_downlink(w, R), noise_pow)

def sinr_uplink(p, w, R, noise_pow):
    assert _is_unit_vector(w)
    return _sinr(p, link_gain_uplink(w, R), noise_pow)

def sinr_margin_downlink(p, w, R, noise_pow, sinr_targets):
    return sinr_downlink(p, w, R, noise_pow) - sinr_targets

def sinr_margin_uplink(p, w, R, noise_pow, sinr_targets):
    return sinr_uplink(p, w, R, noise_pow) - sinr_targets

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
    return _power_alloc_qos(link_gain_downlink(w, R), noise_pow, sinr_targets)

def power_alloc_qos_uplink(w, R, noise_pow, sinr_targets):
    assert _is_unit_vector(w)
    return _power_alloc_qos(link_gain_uplink(w, R), noise_pow, sinr_targets)

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

def power_alloc_minmax_downlink(w, R, noise_pow, sinr_targets, max_pow):
    assert _is_unit_vector(w)
    return _power_alloc_minmax(link_gain_downlink(w, R), noise_pow, sinr_targets, max_pow)

def power_alloc_minmax_uplink(w, R, noise_pow, sinr_targets, max_pow):
    assert _is_unit_vector(w)
    return _power_alloc_minmax(link_gain_uplink(w, R), noise_pow, sinr_targets, max_pow)

def _power_alloc_minmax(gain_mat, noise_pow, sinr_targets, max_pow):
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

def sinr_balance_difference_downlink(p, w, R, noise_pow, sinr_targets):
    return _sinr_balance_difference(p, link_gain_downlink(w, R), noise_pow, sinr_targets)

def sinr_balance_difference_uplink(p, w, R, noise_pow, sinr_targets):
    return _sinr_balance_difference(p, link_gain_uplink(w, R), noise_pow, sinr_targets)

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