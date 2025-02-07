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

import aspcore.correlation as cr
import aspcore.matrices as mat
import aspcore.filter as fc
import aspcore.filterdesign as fd
import aspcore.fouriertransform as ft
import aspcore.montecarlo as mc

import aspcol.kernelinterpolation as ki

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
    w : complex ndarray of shape ()
    num_freqs : int

    Returns
    -------
    ir : ndarray of shape ()
    """
    w = ft.insert_negative_frequencies(w, True)
    ir,_ = fd.fir_from_freqs_window(w, num_freqs-1)
    ir = np.moveaxis(ir, 0,1)
    ir = ir[:1,:,:]
    return ir


def spatial_cov_freq(rir_freq):
    """Calculates spatial covariance matrices from frequency domain room impulse responses

    Parameters
    ----------
    rir_freq : ndarray of shape (num_freq, num_sources, num_mics)
        complex transfer functions from the sources to the microphones

    Returns
    -------
    spatial_cov : ndarray of shape (num_freq, num_sources, num_sources)
        spatial covariance matrix calculated directly from the room impulse responses

    Notes
    -----
    The resulting spatial covariance matrix is only guaranteed to represent the true spatial covariance
    matrix at the microphone points. To represent the spatial covariance matrix taken over a continuous 
    region, the microphone points must both be densely and evenly spaced. 
    """
    assert rir_freq.ndim == 3
    num_sources = rir_freq.shape[1]
    num_mics = rir_freq.shape[2]

    R = rir_freq @ np.moveaxis(rir_freq, 1,2).conj() / num_mics
    return R





def spatial_cov_freq_kernel(krr_params, pos_mic, wave_num, integral_pos_func, integral_volume, num_mc_samples, kernel_func=None, kernel_args=None):
    """Calculates spatial covariance matrices in the frequency domain using kernel interpolation

    Assumes that the kernel function is diagonal, i.e. that there are no cross-terms between the sources. Both standard KRR and 
    directionally weighted KRR is supported.
    
    Parameters
    ----------
    krr_params : ndarray of shape (num_freq, num_source, num_mic)
        the parameters a that together with a kernel function represents the estimated sound field. Can be
        calculated by e.g. get_krr_params from the kernelinterpolation module
    pos_mic : ndarray of shape (num_mic, 3)
        positions of the microphones
    wave_num : ndarray of shape (num_freqs)
        the wavenumbers of all considered frequencies, defined as 2 pi f / c, where c is the speed of sound

    Returns
    -------
    spatial_cov
    """
    if kernel_func is None:
        assert kernel_args is None, "If kernel_func is None, kernel_args must also be None"
        return _spatial_cov_freq_kernel_diffuse(krr_params, pos_mic, wave_num, integral_pos_func, integral_volume, num_mc_samples)
        #kernel_func = ki.kernel_helmholtz_3d

    assert krr_params.ndim == 3
    num_source = krr_params.shape[1]
    num_freq = krr_params.shape[2]

    #currently assumes we can use the same kernel for all sources, i.e. both kernel and mic positions are the same
    def integrand(r):
        kappa = kernel_func(r, pos_mic, wave_num, *kernel_args)
        if kappa.ndim == 3:
            kappa = kappa[:,None,:,:]
        kappa = np.moveaxis(kappa, -2, -1)
        cov_mat = kappa[:,:,:,None,None,:] * kappa[:,None,None,:,:,:].conj()
        #cov_mat = np.moveaxis(cov_mat, -2, -1)
        return cov_mat
        #return kappa[...,None,:] * kappa[...,None,:,:].conj()

    num_mic = pos_mic.shape[0]
    integral_val = mc.integrate(integrand, integral_pos_func, num_mc_samples, integral_volume)
    #integral_val += 1e-10*np.eye(num_mic)[None,:,:]
    #weighting_mat = np.transpose(P,(0,2,1)).conj() @ integral_val @ P

    R = np.sum(np.sum(krr_params[:,:,:,None,None] * integral_val, axis=2) * krr_params[:,None,:,:].conj(), axis=-1)
    R /= integral_volume #Normalization so that it corresponds to the space-discrete covariance
    return R


def _spatial_cov_freq_kernel_diffuse(krr_params, pos_mic, wave_num, integral_pos_func, integral_volume, num_mc_samples, kernel_func=None, kernel_args=None):
    """Calculates spatial covariance matrices in the frequency domain using kernel interpolation
    
    Parameters
    ----------
    krr_params : ndarray of shape (num_freq, num_source, num_mic)
        the parameters a that together with a kernel function represents the estimated sound field. Can be
        calculated by e.g. get_krr_params from the kernelinterpolation module
    pos_mic : ndarray of shape (num_mic, 3)
        positions of the microphones
    wave_num : ndarray of shape (num_freqs)
        the wavenumbers of all considered frequencies, defined as 2 pi f / c, where c is the speed of sound

    
    Returns
    -------
    spatial_cov
    """
    if kernel_func is None:
        kernel_func = ki.kernel_helmholtz_3d
    if kernel_args is None:
        kernel_args = []

    assert krr_params.ndim == 3
    num_source = krr_params.shape[1]
    num_freq = krr_params.shape[2]

    #currently assumes we can use the same kernel for all sources, i.e. both kernel and mic positions are the same
    def integrand(r):
        kappa = kernel_func(r, pos_mic, wave_num, *kernel_args)
        kappa = np.moveaxis(kappa, -2, -1)
        return kappa[:,:,None,:] * kappa[:,None,:,:].conj()

    num_mic = pos_mic.shape[0]
    integral_val = mc.integrate(integrand, integral_pos_func, num_mc_samples, integral_volume)
    #integral_val += 1e-10*np.eye(num_mic)[None,:,:]
    #weighting_mat = np.transpose(P,(0,2,1)).conj() @ integral_val @ P

    R = krr_params @ integral_val @ np.moveaxis(krr_params, 1,2).conj()
    R /= integral_volume #Normalization so that it corresponds to the space-discrete covariance
    return R




def spatial_cov_freq_superpos(Hb, Hd, d=None):
    """Calculates spatial covariance matrices from transfer function matrices

    Assumes a superposition model, so the transfer functions are bright zone and dark zone
    respecively.

    Parameters
    ----------
    Hb : ndarray of shape (num_freq, num_mic_bright, num_ls)
        complex transfer functions from loudspeakers to bright zone
    Hb : ndarray of shape (num_freq, num_mic_dark, num_ls)
        complex transfer functions from loudspeakers to dark zone
    d : ndarray of shape (num_freq, num_mic_bright, num_virt_src), optional
        desired complex sound pressure in the bright zone 

    Returns
    -------
    Rb : ndarray of shape (num_freq, num_ls, num_ls)
        Hermitian spatial covariance for the bright zone
    Rd : ndarray of shape (num_freq, num_ls, num_ls)
        Hermitian spatial covariance for the dark zone
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

def rir_to_szc_cov(rir, ctrlfilt_len):
    """Takes room impulse responses and computes a spatial covariance matrix for sound zone control

    Parameters
    ----------
    rir : ndarray of shape (num_ls, num_mic, ir_len)
        turns it into the time domain sound zone control spatial
        covariance matrix made up of the blocks R_l1l2 = H_l1^T H_l2, 
        where H_l is a convolution matrix with RIRs associated with
        loudspeaker l

    Returns
    -------
        szc_cov : ndarray of shape (num_ls*ctrlfilt_len, num_ls*ctrlfilt_len)

    Notes
    -----
    This should probably be equivalent to spatial_cov_delta. Better write a test and check.  
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
    """Computes time-domain spatial covariance matrix from room impulse responses

    The spatial covariance matrix is 
    $R = \\begin{bmatrix} = $

    Parameters
    ----------
    ir : ndarray of shape (num_ls, num_mic, ir_len)
        room impulse responses from loudspeakers to microphones
    source : Source object
        source object that generates the audio that should be reproduced in the sound zones. 
        The only requirement is that it is an object with a get_samples(num_samples) method, returning a 
        ndarray of shape (num_ls, num_samples).
    filt_len : int
        The length of the desired filter impulse response
    num_samples : int
        The number of samples to use for the spatial covariance matrix. Higher will give a more accurate estimate,
        but take longer to compute. 
    margin : int, optional
        by default the function will use as many samples as possible, which means only removing rir_len-1 samples 
        in the beginning of the filtered source signal, since those samples haven't had time to propagate properly. 
        margin can be supplied if a specific number of samples should be removed instead.
        might give questionable result if you set margin to less than rir_len-1.

    Returns
    -------
    R : ndarray of shape (num_ls*filt_len, num_ls*filt_len)
        The spatial covariance matrix
    
    Notes
    -----
    \\begin{equation}
        \\bm{R}_{zi} = \\expect \\bigl[ \\mathbb{X}_i^\\top (n) \\bm{H}_z \\bm{H}_z^{\\top} \\mathbb{X}_i(n)\\bigr] \\in \\mathbb{R}^{LI\\times LI}
    \\end{equation}

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
        
    Returns
    -------
        spatial_covariance : ndarray of size (num_ls*filt_len, num_ls*filt_len)
    """
    ir_len = ir.shape[-1]
    ir = np.moveaxis(ir, 1, 0)
    R = cr.corr_matrix(ir, ir, filt_len, filt_len) * ir_len
    return R









def fpaths_to_spatial_cov(arrays, fpaths, source_name, zone_names):
    """utility function to be used with aspsim package. Deprecated, and will be removed in future versions.

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

def paths_to_spatial_cov(arrays, source_name, zone_names, sources, filt_len, num_samples, margin=None):
    """utility function to be used with aspsim package. Deprecated, and will be removed in future versions.

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
    """utility function to be used with aspsim package. Deprecated, and will be removed in future versions.

    See info for paths_to_spatial_cov
    """
    num_sources = arrays[source_name].num
    num_zones = len(zone_names)

    R = np.zeros((num_zones, num_zones, filt_len*num_sources, filt_len*num_sources), dtype=float)
    for k in range(num_zones):
        for i in range(num_zones):
            R[k,i,:,:] = mat.ensure_pos_semidef(spatial_cov_delta(arrays.paths[source_name][zone_names[k]], filt_len))
    return R