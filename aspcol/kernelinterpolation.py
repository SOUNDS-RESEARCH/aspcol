import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import numba as nb

import aspcol.utilities as util
import aspcol.filterdesign as fd
import aspcol.montecarlo as mc

def kernel_gaussian(points1, points2, scale):
    dist_mat = distfuncs.cdist(points1, points2)**2
    return np.exp(-scale[:,None,None]**2 * dist_mat[None,:,:])

def kernel_helmholtz_2d(points1, points2, wave_num):
    """points1 is shape (numPoints1, 2)
        points2 is shape (numPoints2, 2)
        waveNum is shape (numFreqs)

        returns shape (numFreqs, numPoints1, numPoints2)
    """
    dist_mat = distfuncs.cdist(points1, points2)
    return special.j0(dist_mat[None,:,:] * wave_num[:,None,None])


def kernel_helmholtz_3d_slow(points1, points2, wave_num):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)

        returns shape (numFreqs, numPoints1, numPoints2)
    """
    distMat = distfuncs.cdist(points1, points2)
    return special.spherical_jn(0, distMat[None,:,:] * wave_num[:,None,None])


@nb.njit
def kernel_helmholtz_3d(points1, points2, wave_num):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)

        returns shape (numFreqs, numPoints1, numPoints2)
    """
    #distMat = distfuncs.cdist(points1, points2)

    dist_mat = np.sqrt(np.sum((np.expand_dims(points1,1) - np.expand_dims(points2,0))**2, axis=-1))
    return np.sinc(np.expand_dims(dist_mat,0) * wave_num.reshape(-1, 1,1) / np.pi)


def kernel_directional_3d(points1, points2, wave_num, angle, beta):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)
        angle is tuple (theta, phi) defined as in util.spherical2cart
        beta sets the strength of the directional weighting
        
        returns shape (numFreqs, numPoints1, numPoints2)
    """
    rDiff = points1[:,None,:] - points2[None,:,:]
    angleFactor = beta * util.spherical2cart(np.ones((1,1)), np.array(angle)[None,:])[None,None,...]
    posFactor = 1j * wave_num[:,None,None,None] * rDiff[None,...]
    return special.spherical_jn(0, 1j*np.sqrt(np.sum((angleFactor + posFactor)**2, axis=-1)))

@nb.njit
def kernel_directional_vec_3d(points1, points2, wave_num, direction_vec, beta):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)
        directionVec is shape (numAngles, 3), where each (1,3) is a unit vector
        
        returns shape (numFreqs, numAngles, numPoints1, numPoints2)
    
        Implements the same kernel function as kernel_directional_3d, 
        but accepts the direction in the form of a unit vector
        and is vectorized over angles as well as frequencies
    """
    angle_term = 1j * beta * direction_vec.reshape((1,-1,1,1,direction_vec.shape[-1]))
    pos_term = wave_num.reshape((-1,1,1,1,1)) * (points1.reshape((1,1,-1,1,points1.shape[-1])) - points2.reshape((1,1,1,-1,points2.shape[-1])))
    return np.sinc(np.sqrt(np.sum((angle_term - pos_term)**2, axis=-1)) / np.pi)


def kernel_reciprocal_3d(points1, points2, wave_num):
    """points is a tuple (micPoints, srcPoints),
        where micPoints is ndarray of shape (numMic, 3)
        srcPoints is ndarray of shape (numSrc, 3)

        waveNum is shape (numFreq)
        output has shape (numFreq, numMic1*numSrc1, numMic2*numSrc2)
        they are placed according to micIdx+numMics*srcIdx

        When flattened, the index for microphone m, speaker l is m+l*M. i.e.
        the microphone index changes faster. 

    From the paper: Kernel interpolation of acoustic transfer 
                function between regions considering reciprocity"""
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
    """Calculates kernel weighting filter A(w) in frequency domain
        see 'Spatial active noise control based on kernel interpolation of sound field' by Koyama et al.     

        kernelFunc is one of kernelHelmholtz3d, kernelHelmholtz2d, kernelDirectional3d as defined in this module
        regParam is positive scalar
        micPos is ndarray with shape (numPositions, spatialDimension)
        integralDomain is subclass of Region object, found in soundfield/geometry module
        mcSamples is integer, how many monte carlo samples to be drawn for integration
        samplerate is integer
        c is speed of sound
        *args are arguments needed for kernel function except points1, points2, waveNum
        
        For both diffuse and directional kernel P^H = P, so the hermitian tranpose should not do anything
        It is left in place in case a kernel function in the future changes that identity. """
    freqs = fd.get_frequency_values(num_freq, samplerate)
    wave_num = 2 * np.pi * freqs / c

    def integrableFunc(r):
        kappa = kernel_func(r, mic_pos, wave_num, *args)
        kappa = np.transpose(kappa,(0,2,1))
        return kappa.conj()[:,:,None,:] * kappa[:,None,:,:]

    num_mics = mic_pos.shape[0]
    K = kernel_func(mic_pos, mic_pos, wave_num, *args)
    P = np.linalg.pinv(K + reg_param * np.eye(num_mics))

    integral_value = mc.integrate(integrableFunc, integral_domain.sample_points, mc_samples, integral_domain.volume)
    weighting_filter = np.transpose(P,(0,2,1)).conj() @ integral_value @ P

    weighting_filter = fd.insert_negative_frequencies(weighting_filter, even=True)
    return weighting_filter



def get_krr_parameters(kernel_func, reg_param, output_arg, data_arg, *args):
    """Calculates parameter vector or matrix given a kernel function for Kernel Ridge Regression.
        The returned parameter Z is the optimal interpolation filter from the data points to
        the output points. Apply filter as Z @ y, where y are the labels for data at data_arg positions
    
    Parameters
    ----------
    data_arg : ndarray (num_data_points, data_dim)
    output_arg : ndarray (num_out_points, data_dim)
    kernel_func : function 
        with calling signature kernel_func(output_arg, data_arg, *args)
        should return ndarray (..., num_out_points, num_data_points)
    
    Returns
    -------
    params : ndarray (..., num_out_points, num_data_points)
    """
    K = kernel_func(data_arg, data_arg, *args)
    K_reg = K + reg_param * np.eye(K.shape[-1])
    kappa = np.moveaxis(kernel_func(output_arg, data_arg, *args), -1, -2)

    params = np.moveaxis(np.linalg.solve(K_reg, kappa), -1, -2)
    return params


def soundfield_interpolation_fir(
    to_points, from_points, ir_len, reg_param, num_freq, spatial_dims, samplerate, c
):
    """Convenience function for calculating the time domain causal FIR interpolation filter
    from a set of points to a set of points. """
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
    from a set of points to a set of points. """
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
    """Analytic solution of the kernel interpolation weighting filter integral
        in the frequency domain for a disc in 2D. """
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