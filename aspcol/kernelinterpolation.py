"""Interpolation of a sound field taking physical properties of sound into account. [1, 2]

The estimation methods are written generally to allow for any kernel function to be given as argument, 
and then functions implementing the kernel functions associated with the papers below are provided. 

A number of kernels are implemented:
* Gaussian kernel
* Diffuse kernel in 2D [6] and 3D [1]
* Directional kernel in 3D [2, 4]
* Reciprocal kernel for RIR estimation [3]

References
----------
`[1] <doi.org/10.1109/IWAENC.2018.8521334>`_  N. Ueno, S. Koyama, and H. Saruwatari, “Kernel ridge regression with constraint of Helmholtz equation for sound field interpolation,” in 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan: IEEE, Sep. 2018, pp. 436–440. doi: 10.1109/IWAENC.2018.8521334.
`[2] <doi.org/10.1109/TSP.2021.3070228>`_ N. Ueno, S. Koyama, and H. Saruwatari, “Directionally weighted wave field estimation exploiting prior information on source direction,” IEEE Transactions on Signal Processing, vol. 69, pp. 2383–2395, Apr. 2021, doi: 10.1109/TSP.2021.3070228. 
`[3] <doi.org/10.1109/SAM48682.2020.9104256>`_  J. G. C. Ribeiro, N. Ueno, S. Koyama, and H. Saruwatari, “Kernel interpolation of acoustic transfer function between regions considering reciprocity,” in 2020 IEEE 11th Sensor Array and Multichannel Signal Processing Workshop (SAM), Jun. 2020, pp. 1–5. doi: 10.1109/SAM48682.2020.9104256.
`[4] <doi.org/10.1109/TASLP.2021.3107983>`_  S. Koyama, J. Brunnström, H. Ito, N. Ueno, and H. Saruwatari, “Spatial active noise control based on kernel interpolation of sound field,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3052–3063, Aug. 2021, doi: 10.1109/TASLP.2021.3107983.
`[5] <doi.org/10.1109/ICASSP43922.2022.9746550>`_  J. Brunnström, S. Koyama, and M. Moonen, “Variable span trade-off filter for sound zone control with kernel interpolation weighting,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), May 2022, pp. 1071–1075. doi: 10.1109/ICASSP43922.2022.9746550. 
`[6] <doi.org/10.1109/ICASSP.2019.8683067>`_ H. Ito, S. Koyama, N. Ueno, and H. Saruwatari, “Feedforward spatial active noise control based on kernel interpolation of sound field,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom: IEEE, May 2019, pp. 511–515. doi: 10.1109/ICASSP.2019.8683067.

"""
import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import numba as nb

import aspcol.utilities as util
import aspcol.filterdesign as fd
import aspcol.montecarlo as mc

def kernel_gaussian(points1, points2, scale):
    """
    Guassian kernel, also known as the radial basis function kernel. 

    Parameters
    ----------
    points1 : ndarray of shape (num_points1, spatial_dim)
    points2 : ndarray of shape (num_points2, spatial_dim)
    scale : ndarray of shape (num_scales,)

    Returns
    -------
    ndarray of shape (num_scales, num_points1, num_points2)
    """
    dist_mat = distfuncs.cdist(points1, points2)**2
    return np.exp(-scale[:,None,None]**2 * dist_mat[None,:,:])

def kernel_helmholtz_2d(points1, points2, wave_num):
    """
    Parameters
    ----------
    points1 : ndarray of shape (num_points1, 2)
    points2 : ndarray of shape (num_points2, 2)
    wave_num : ndarray of shape (num_freqs,)

    Returns
    -------
    ndarray of shape (num_freqs, num_points1, num_points2)
    """
    dist_mat = distfuncs.cdist(points1, points2)
    return special.j0(dist_mat[None,:,:] * wave_num[:,None,None])


def kernel_helmholtz_3d_slow(points1, points2, wave_num):
    """
    Identical to kernel_helmholtz_3d, but is not JIT compiled by numba. 
    This is faster if the kernel is only evaluated once, but slower if it is evaluated many times
    
    Parameters
    ----------
    points1 : ndarray of shape (num_points1, 3)
    points2 : ndarray of shape (num_points2, 3)
    wave_num : ndarray of shape (num_freqs,)

    Returns
    -------
    ndarray of shape (num_freqs, num_points1, num_points2)
    """
    distMat = distfuncs.cdist(points1, points2)
    return special.spherical_jn(0, distMat[None,:,:] * wave_num[:,None,None])


@nb.njit
def kernel_helmholtz_3d(points1, points2, wave_num):
    """
    Diffuse kernel for 3D sound field interpolation. Defined in 
    'Spatial active noise control based on kernel interpolation 
    of sound field' by Koyama et al.

    Parameters
    ----------
    points1 : ndarray of shape (num_points1, 3)
    points2 : ndarray of shape (num_points2, 3)
    wave_num : ndarray of shape (num_freqs,)

    Returns
    -------
    ndarray of shape (num_freqs, num_points1, num_points2)
    """
    #distMat = distfuncs.cdist(points1, points2)

    dist_mat = np.sqrt(np.sum((np.expand_dims(points1,1) - np.expand_dims(points2,0))**2, axis=-1))
    return np.sinc(np.expand_dims(dist_mat,0) * wave_num.reshape(-1, 1,1) / np.pi)


def kernel_directional_3d_slow(points1, points2, wave_num, angle, beta):    
    """
    Identical to kernel_directional_3d, but is not JIT compiled by numba. 
    This is faster if the kernel is only evaluated once, but slower if it is evaluated many times

    In addition, this only allows for a single angle to be evaluated at a time.

    Parameters
    ----------
    points1 : ndarray of shape (num_points1, 3)
    points2 : ndarray of shape (num_points2, 3)
    wave_num : ndarray of shape (num_freqs,)
    angle : tuple (theta, phi) defined as in util.spherical2cart
    beta : sets the strength of the directional weighting

    Returns
    -------
    ndarray of shape (num_freqs, num_points1, num_points2)
    """
    rDiff = points1[:,None,:] - points2[None,:,:]
    angleFactor = beta * util.spherical2cart(np.ones((1,1)), np.array(angle)[None,:])[None,None,...]
    posFactor = 1j * wave_num[:,None,None,None] * rDiff[None,...]
    return special.spherical_jn(0, 1j*np.sqrt(np.sum((angleFactor + posFactor)**2, axis=-1)))

@nb.njit
def kernel_directional_3d(points1, points2, wave_num, direction_vec, beta):
    """
    Directionally weighted kernel for 3D sound field interpolation. 
    Defined in 'Spatial active noise control based on kernel interpolation 
    of sound field' by Koyama, Brunnström, Ito, Ueno, Saruwatari.
    
    Parameters
    ----------
    points1 : ndarray of shape (num_points1, 3)
    points2 : ndarray of shape (num_points2, 3)
    wave_num : ndarray of shape (num_freqs,)
    direction_vec : ndarray of shape (num_angles, 3)
        unit vectors describing the arrival direction
    beta : float nonnegative 
        sets the strength of the directional weighting

    Returns
    -------
    ndarray of shape (num_freqs, num_angles, num_points1, num_points2)
    """
    angle_term = 1j * beta * direction_vec.reshape((1,-1,1,1,direction_vec.shape[-1]))
    pos_term = wave_num.reshape((-1,1,1,1,1)) * (points1.reshape((1,1,-1,1,points1.shape[-1])) - points2.reshape((1,1,1,-1,points2.shape[-1])))
    return np.sinc(np.sqrt(np.sum((angle_term - pos_term)**2, axis=-1)) / np.pi)


def kernel_reciprocal_3d(points1, points2, wave_num):
    """
    Reciprocal kernel for room impulse response interpolation. Definition found in
    'Kernel interpolation of acoustic transfer function between regions considering reciprocity'
    by Ribeiro, Ueno, Koyama, Saruwatari.

    Parameters
    ----------
    points1 : 2-tuple of (mic_points1, src_points1)
        where mic_points ndarray of shape (num_mic1, 3)
        and src_points ndarray of shape (num_src1, 3)
    points2 : same type of object as points1, 
        although the ndarray shapes can be different
    wave_num : ndarray of shape (num_freqs,)

    Returns
    -------
    ndarray of shape (num_freqs, num_mic1*num_src1, num_mic2*num_src2)
        When flattened, the index for microphone m, speaker l is m+l*M. i.e.
        the microphone index changes faster. 
    """
    wave_num = wave_num[:,None,None,None,None]
    mic_dist = distfuncs.cdist(points1[0], points2[0])[None,None,:,None,:]
    src_dist = distfuncs.cdist(points1[1], points2[1])[None,:,None,:,None]
    mix_dist1 = distfuncs.cdist(points1[0], points2[1])[None,None,:,:,None]
    mic_dist2 = distfuncs.cdist(points1[1], points2[0])[None,:,None,None,:]

    k_val = 0.5 * (special.spherical_jn(0, wave_num * mic_dist) * \
                        special.spherical_jn(0, wave_num * src_dist)) + \
                        (special.spherical_jn(0, wave_num * mix_dist1) * \
                        special.spherical_jn(0, wave_num * mic_dist2))
    k_val = np.reshape(k_val, k_val.shape[:3]+(-1,))
    k_val = np.reshape(k_val, (k_val.shape[0], -1,k_val.shape[-1]))
    return k_val


def get_kernel_weighting_filter(kernel_func, reg_param, mic_pos, integral_domain, 
                                mc_samples, num_freq, samplerate, c, *args):
    """ 
    Calculates kernel weighting filter A(w) in frequency domain
    see 'Spatial active noise control based on kernel interpolation of sound field' by Koyama et al.     

    Parameters
    ----------
    kernel_func : function
        with calling signature kernel_func(points1, points2, waveNum, \*args)
    reg_param : float
    mic_pos : ndarray of shape (num_mics, spatial_dim)
    integral_domain : instance of any Region object found is aspsim package
    mc_samples : int
        how many monte carlo samples to be drawn for integration
    num_freq : int
    samplerate : int
    c : float
        speed of sound
    *args : arguments needed for kernel function except points1, points2, waveNum

    Returns
    -------
    ndarray of shape (num_freq, num_mics, num_mics)
    
    For both diffuse and directional kernel P^H = P, so the hermitian tranpose should not do anything
    It is left in place in case a kernel function in the future changes that identity. 
    """
    freqs = fd.get_frequency_values(num_freq, samplerate)
    wave_num = 2 * np.pi * freqs / c

    def integrable_func(r):
        kappa = kernel_func(r, mic_pos, wave_num, *args)
        kappa = np.transpose(kappa,(0,2,1))
        return kappa.conj()[:,:,None,:] * kappa[:,None,:,:]

    num_mics = mic_pos.shape[0]
    K = kernel_func(mic_pos, mic_pos, wave_num, *args)
    P = np.linalg.pinv(K + reg_param * np.eye(num_mics))

    integral_value = mc.integrate(integrable_func, integral_domain.sample_points, mc_samples, integral_domain.volume)
    weighting_filter = np.transpose(P,(0,2,1)).conj() @ integral_value @ P

    weighting_filter = fd.insert_negative_frequencies(weighting_filter, even=True)
    return weighting_filter



def get_krr_parameters(kernel_func, reg_param, output_arg, data_arg, *args):
    """
    Calculates parameter vector or matrix given a kernel function for Kernel Ridge Regression.
    The returned parameter Z is the optimal interpolation filter from the data points to
    the output points. Apply filter as Z @ y, where y are the labels for data at data_arg positions
    
    Parameters
    ----------
    data_arg : ndarray (num_data_points, data_dim)
    output_arg : ndarray (num_out_points, data_dim)
    kernel_func : function 
        with calling signature kernel_func(output_arg, data_arg, \*args)
        should return ndarray (..., num_out_points, num_data_points)
    
    Returns
    -------
    params : ndarray (..., num_out_points, num_data_points)
    """
    K = kernel_func(data_arg, data_arg, *args)
    K_reg = K + reg_param * np.eye(K.shape[-1])
    kappa = np.moveaxis(kernel_func(output_arg, data_arg, *args), -1, -2)

    params = np.moveaxis(np.linalg.solve(np.moveaxis(K_reg, -1, -2), kappa), -1, -2)
    return params


def soundfield_interpolation_fir(
    to_points, from_points, ir_len, reg_param, num_freq, spatial_dims, samplerate, c
):
    """
    Convenience function for calculating the time domain causal FIR interpolation filter
    from a set of points to another set of points. 
    
    Parameters
    ----------
    to_points : ndarray of shape (num_to_points, spatial_dims)
    from_points : ndarray of shape (num_from_points, spatial_dims)
    ir_len : int
        length of the impulse response
    reg_param : float
    num_freq : int
    spatial_dims : int
    samplerate : int
    c : float
        speed of sound

    Returns
    -------
    ndarray of shape (ir_len, num_to_points, num_from_points)    
    """
    assert num_freq > ir_len
    freq_filter = soundfield_interpolation(
        to_points, from_points, num_freq, reg_param, spatial_dims, samplerate, c
    )
    ki_filter,_ = fd.fir_from_freqs_window(freq_filter, ir_len)
    return ki_filter

def soundfield_interpolation(
    to_points, from_points, num_freq, reg_param, spatial_dims, samplerate, c
):
    """ Convenience function for calculating the frequency domain interpolation filter
    from a set of points to another set of points. 
    
    Parameters
    ----------
    to_points : ndarray of shape (num_to_points, spatial_dims)
    from_points : ndarray of shape (num_from_points, spatial_dims)
    num_freq : int
    reg_param : float
    spatial_dims : int
    samplerate : int
    c : float
        speed of sound

    Returns
    -------
    ndarray of shape (num_freq, num_to_points, num_from_points)
    """
    if spatial_dims == 3:
        kernel_func = kernel_helmholtz_3d
    elif spatial_dims == 2:
        kernel_func = kernel_helmholtz_2d
    else:
        raise ValueError

    assert num_freq % 2 == 0

    freqs = fd.get_frequency_values(num_freq, samplerate)#[:, None, None]
    wave_num = 2 * np.pi * freqs / c
    ip_params = get_krr_parameters(kernel_func, reg_param, to_points, from_points, wave_num)
    ip_params = fd.insert_negative_frequencies(ip_params, even=True)
    return ip_params




def analytic_kernel_weighting_disc_2d(error_mic_pos, freq, reg_param, trunc_order, radius, c):
    """
    Analytic solution of the kernel interpolation weighting filter integral
    in the frequency domain for a disc in 2D. See definition in 'Feedforward 
    spatial active noise control based on kernel interpolation of sound field'
    by Ito, Koyama, Ueno, Saruwatari.

    Parameters
    ----------
    error_mic_pos : ndarray of shape (num_mics, spatial_dim)
    freq : float or ndarray of shape (num_freqs,)
    reg_param : float
    trunc_order : int
    radius : float
    c : float

    Returns
    -------
    ndarray of shape (num_freqs, num_mics, num_mics)
    """
    if isinstance(freq, (int, float)):
        freq = np.array([freq])
    if len(freq.shape) == 1:
        freq = freq[:, np.newaxis, np.newaxis]
    wave_number = 2 * np.pi * freq / c
    K = special.j0(wave_number * distfuncs.cdist(error_mic_pos, error_mic_pos))
    P = np.linalg.pinv(K + reg_param * np.eye(K.shape[-1]))
    S = _get_s(trunc_order, wave_number, error_mic_pos)
    gamma = _get_gamma(trunc_order, wave_number, radius)
    A = (
        np.transpose(P.conj(), (0, 2, 1))
        @ np.transpose(S.conj(), (0, 2, 1))
        @ gamma
        @ S
        @ P
    )
    return A

def _get_gamma(maxOrder, k, R):
    matLen = 2 * maxOrder + 1
    diagValues = _small_gamma(np.arange(-maxOrder, maxOrder + 1), k, R)
    gamma = np.zeros((diagValues.shape[0], matLen, matLen))

    gamma[:, np.arange(matLen), np.arange(matLen)] = diagValues
    return gamma

def _small_gamma(mu, k, R):
    Jfunc = special.jv((mu - 1, mu, mu + 1), k * R)
    return np.pi * (R ** 2) * ((Jfunc[:, 1, :] ** 2) - Jfunc[:, 0, :] * Jfunc[:, 2, :])

def _get_s(maxOrder, k, positions):
    r, theta = util.cart2pol(positions[:, 0], positions[:, 1])

    mu = np.arange(-maxOrder, maxOrder + 1)[:, np.newaxis]
    S = special.jv(mu, k * r) * np.exp(theta * mu * (-1j))
    return S