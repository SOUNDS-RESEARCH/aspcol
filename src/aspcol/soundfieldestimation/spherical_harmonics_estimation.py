"""Functions for handling the spherical harmonic wave function and estimating the sound field coefficients

All math and translation theorems are taken from [martinMultiple2006] unless otherwise stated.

References
----------
[martinMultiple2006] P. A. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles, vol. 107. in Encyclopedia of mathematics and its applications, vol. 107. Cambridge, UK: Cambridge University Press, 2006. \n
[brunnstromBayesianSubmitted] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted. \n
[uenoSound2018] N. Ueno, S. Koyama, and H. Saruwatari, “Sound field recording using distributed microphones based on harmonic analysis of infinite order,” IEEE Signal Process. Lett., vol. 25, no. 1, pp. 135–139, Jan. 2018, doi: 10.1109/LSP.2017.2775242. \n


Equivalence of spherical harmonic definitions
---------------------------------------------
The scipy spherical harmonic implementation is 
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html

The Legendre polynomial (referred to as Legendre function of the first kind in Scipy documentation)
definition given in Scipy is (37) after inserting the definition of the Hypergeometric function. 
The definition used in Martin is (30), the Rodrigues formula.
The two definitions are equivalent, therefore the Legendre polynomial is consistent. 
https://mathworld.wolfram.com/LegendrePolynomial.html

Martins definition of the Associated Legendre Polynomial $P_n^m(t)$, found in (A.1) in [martinMultiple2006]
is $P_n^m(t) = (1-t^2)^{m/2} \\frac{d^m}{dt^m} P_n(t)$. Scipy has almost the same definition, $S_n^m(t) = (-1)^m P_n^m(t)$,
where S_n^m(t) denotes the Scipy definition. Therefore they are equivalent except for that Scipy includes
the Condon-Shortley phase in the ALP. 

The definition of the spherical harmonics in (3.6) in [martinMultiple2006] is equivalent to Scipy's definition,
except for the inclusion of an additional factor (-1)^m, i.e. the Condon-Shortley phase. Because the 
same factor is included in Scipy's ALP but not Martin ALP, the spherical harmonic definitions are equivalent. 
"""
import numpy as np
import aspcol.kernelinterpolation as ki
import aspcol.sphericalharmonics as shd

# ====================== SHD ESTIMATION ======================

def measurement_omni(pos, exp_center, max_order, wave_num):
    T = shd.translation_operator(pos - exp_center, wave_num, max_order, 0)
    T = np.sum(T, axis=-2) # inner product with omni directionality
    return T

def measurement_conj_omni(pos, exp_center, max_order, wave_num):
    """Returns the adjoint measurement operator Xi^H

    Can be applied as shd_coeffs(pos) = Xi^H @ p
    where p is complex sound pressure at the measurement points
    
    Parameters
    ----------
    pos : ndarray of shape (num_pos, 3)
        positions of the measurement points where the sound pressure was measured
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order = int
        maximum order of the spherical harmonics that is output by the operator
    wave_num : ndarray of shape (num_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    
    Returns
    -------
    operator : ndarray of shape (num_freqs, num_coeffs, num_pos)
        adjoint of the measurement operator
    """
    T = shd.translation_operator(exp_center - pos, wave_num, 0, max_order)
    T = np.sum(T, axis=-1) # inner product with omni directionality
    return np.moveaxis(T, 1,2)

def apply_measurement_omni(shd_coeffs, pos, exp_center, wave_num):
    """Applies the measurement operator to a sequence of spherical harmonic coefficients
    
    The operator is a function from a complex infinite (l2) sequence to a complex vector
    Xi : l^2(C) -> C^M
    
    Parameters
    ----------
    shd_coeffs : ndarray of shape (num_freq, num_coeffs,)
        spherical harmonic coefficients
    pos : ndarray of shape (num_pos, 3)
        positions of the measurement points where the sound pressure should be estimated
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.

    Returns
    -------
    p_est : ndarray of shape (num_freq, num_mic,)
        complex sound pressure at the measurement points
    """
    rel_pos = pos - exp_center
    #num_coeffs = shd_coeffs.shape[1]
    #max_order = shd_max_order(num_coeffs) 
    shd_translated = shd.translate_shd_coeffs(shd_coeffs, rel_pos, wave_num, 0)
    p_est = np.sum(shd_translated, axis=-1) # must be multiplied by dir.conj() when not omnidirectional
    return p_est


def apply_measurement_conj_omni(vec, pos, exp_center, max_order, wave_num):
    """Applies the conjugate measurement operator to a complex vector

    The operator is a function from a complex vector to a complex infinite (l2) sequence
    Xi^H : C^M -> l^2(C)
    
    Parameters
    ----------
    vec : ndarray of shape (num_freqs, num_pos)
        complex vector
    pos : ndarray of shape (num_pos, 3)
        positions of the measurement points where the sound pressure was measured
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order = int
        maximum order of the spherical harmonics that is output by the operator
    wave_num : ndarray of shape (num_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound. 
    
    Returns
    -------
    shd_coeffs : ndarray of shape (num_freqs, num_coeffs,)
        coefficients of the infinite sequence
    """
    rel_pos = exp_center - pos
    directivity = shd.directivity_omni()
    dir_translated = shd.translate_shd_coeffs(directivity, rel_pos, wave_num, max_order)
    return np.sum(dir_translated * vec[:,:,None], axis=1)

def apply_measurement(shd_coeffs, pos, exp_center, wave_num, dir_coeffs=None):
    """Applies the measurement operator to a sequence of spherical harmonic coefficients
    
    The operator is a function from a complex infinite (l2) sequence to a complex vector
    Xi : l^2(C) -> C^M

    Assumes for now that the same directivity is used for all microphones
    
    Parameters
    ----------
    shd_coeffs : ndarray of shape (num_freq, num_coeffs,)
        spherical harmonic coefficients representing the sound field being measured
    pos : ndarray of shape (num_mic, 3)
        positions of the measurement points where the sound pressure should be estimated
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    dir_coeffs : ndarray of shape (num_freqs, num_coeffs) or (num_freqs, num_mic, num_coeffs), optional
        coefficients of the directivity function. 
        If None, the directivity is assumed to be omnidirectional

    Returns
    -------
    p_est : ndarray of shape (num_freq, num_mic,)
        complex sound pressure at the measurement points
    """
    num_mic = pos.shape[0]
    num_freq = wave_num.shape[0]
    num_coeffs = shd_coeffs.shape[1]
    assert shd_coeffs.shape == (num_freq, num_coeffs,)
    assert pos.shape == (num_mic, 3)
    assert exp_center.shape == (1, 3)
    assert wave_num.shape == (num_freq,)

    rel_pos = pos - exp_center
    if dir_coeffs is None:
        directivity = shd.directivity_omni()
    else:
        directivity = dir_coeffs

    if directivity.ndim == 2:
        directivity = directivity[:,None,:]
    assert directivity.ndim == 3
    assert directivity.shape[0] == num_freq or directivity.shape[0] == 1
    assert directivity.shape[1] == num_mic or directivity.shape[1] == 1

    max_order_dir = shd.shd_max_order(directivity.shape[-1])
    #max_order = shd_max_order(num_coeffs)
    shd_translated = shd.translate_shd_coeffs(shd_coeffs, rel_pos, wave_num, max_order_dir)
    if shd_translated.ndim == 2:
        shd_translated = shd_translated[:,None,:]

    p_est = np.sum(shd_translated * np.conj(directivity), axis=-1)
    return p_est


def apply_measurement_conj(vec, pos, exp_center, max_order, wave_num, dir_coeffs=None):
    """Applies the conjugate measurement operator to a complex vector

    The operator is a function from a complex vector to a complex infinite (l2) sequence
    Xi^H : C^M -> l^2(C)
    
    Parameters
    ----------
    vec : ndarray of shape (num_freqs, num_pos)
        complex vector
    pos : ndarray of shape (num_pos, 3)
        positions of the measurement points where the sound pressure was measured
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order = int
        maximum order of the spherical harmonics that is output by the operator
    wave_num : ndarray of shape (num_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound. 
    dir_coeffs : ndarray of shape (num_freqs, num_coeffs) or (num_freqs, num_pos, num_coeffs), optional
        coefficients of the directivity function. If None, the directivity is assumed to be omnidirectional
    
    Returns
    -------
    shd_coeffs : ndarray of shape (num_freqs, num_coeffs,)
        coefficients of the infinite sequence
    """
    rel_pos = exp_center - pos
    if dir_coeffs is None:
        directivity = shd.directivity_omni()
    else:
        directivity = dir_coeffs
    max_order_dir = shd.shd_max_order(directivity.shape[-1])
    
    dir_translated = shd.translate_shd_coeffs(directivity, rel_pos, wave_num, max_order)
    return np.sum(dir_translated * vec[:,:,None], axis=1)

def inf_dimensional_shd(p, pos, exp_center, max_order, wave_num, reg_param, dir_coeffs=None):
    """Estimates the spherical harmonic coefficients with Bayesian inference, allows for arbitrary directivity
    
    Implements the method in Natsuki Ueno, Shoichi Koyama, Hiroshi Saruwatai,
    'Sound Field Recording Using Distributed Microphones Based on Harmonic Analysis of Infinite Order'

    Parameters
    ----------
    p : ndarray of shape (num_real_freqs, num_mics)
        complex sound pressure for each microphone and frequency
    pos : ndarray of shape (num_mics, 3)
        positions of the microphones
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order : int
        maximum order of the spherical harmonics
    wave_num : float
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound
    reg_param : float
        regularization parameter. Must be non-negative
    dir_coeffs : ndarray of shape (num_real_freqs, num_coeffs) or (num_real_freqs, num_mics, num_coeffs)
        coefficients of the directivity function. If None, the directivity is assumed to be omnidirectional
        if the array is 2-dimensional, the same directivity is assumed for all microphones
    
    Returns
    -------
    shd_coeffs : complex ndarray of shape (num_real_freqs, num_coeffs)
        harmonic coefficients of the estimated sound field
    """
    assert exp_center.shape == (3,) or exp_center.shape == (1, 3)
    num_mic = pos.shape[0]

    psi = shd.translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, wave_num)
    psi_plus_noise_cov = psi + np.eye(num_mic) * reg_param
    regression_vec = np.linalg.solve(psi_plus_noise_cov,  p)

    shd_coeffs = apply_measurement_conj(regression_vec, pos, exp_center, max_order, wave_num, dir_coeffs=dir_coeffs)
    return shd_coeffs


def inf_dimensional_shd_omni(p, pos, exp_center, max_order, wave_num, reg_param):
    """Estimates the spherical harmonic coefficients with Bayesian inference
    
    Assumes all microphones are omnidirectional.
    Is identical to posterior_mean_omni, if reg_param = noise_power / prior_variance. So 
    this function should be a wrapper for posterior_mean_omni or removed. 

    Parameters
    ----------
    p : ndarray of shape (num_real_freqs, num_mics)
        complex sound pressure for each microphone and frequency
    pos : ndarray of shape (num_mics, 3)
        positions of the microphones
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order : int
        maximum order of the spherical harmonics
    wave_num : ndarray of shape (num_real_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound
    reg_param : float or ndarray of shape (num_real_freqs,)
        regularization parameter. Must be non-negative
    
    Returns
    -------
    shd_coeffs : complex ndarray of shape (num_real_freqs, num_coeffs)
        harmonic coefficients of the estimated sound field

    References
    ----------
    Natsuki Ueno, Shoichi Koyama, Hiroshi Saruwatai, 'Sound Field Recording Using Distributed 
    Microphones Based on Harmonic Analysis of Infinite Order'
    """
    num_mic = pos.shape[0]
    num_freq = wave_num.shape[0]

    if isinstance(reg_param, np.ndarray):
        assert reg_param.ndim == 1
        assert reg_param.shape[0] == num_freq
        assert np.all(reg_param >= 0)
    else:
        assert reg_param >= 0
        reg_param = np.ones(num_freq) * reg_param
    noise_cov = np.eye(num_mic)[None,...] * reg_param[:,None,None]

    psi = ki.kernel_helmholtz_3d(pos, pos, wave_num)
    psi_plus_noise_cov = psi + noise_cov
    regression_vec = np.linalg.solve(psi_plus_noise_cov,  p)

    shd_coeffs = apply_measurement_conj_omni(regression_vec, pos, exp_center, max_order, wave_num)
    return shd_coeffs

def inf_dimensional_shd_omni_prior(p, pos, exp_center, max_order, wave_num, reg_param, prior_mean):
    """Estimates spherical harmonic coefficients with Bayesian inference of an infinite sequence of spherical harmonics

    Implements the method in Natsuki Ueno, Shoichi Koyama, Hiroshi Saruwatai,
    'Sound Field Recording Using Distributed Microphones Based on Harmonic Analysis of Infinite Order'

    Parameters
    ----------
    p : ndarray of shape (num_real_freqs, num_mics)
        complex sound pressure for each microphone and frequency
    pos : ndarray of shape (num_mics, 3)
        positions of the microphones
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order : int
        maximum order of the spherical harmonics
    reg_param : float
        regularization parameter. Must be non-negative
    wave_num : ndarray of shape (num_real_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound
    prior_mean : ndarray of shape (num_real_freqs, num_coeffs,)
        mean of the prior distribution for the sherical harmonic coefficients
    
    Returns
    -------
    shd_coeffs : complex ndarray of shape (num_real_freqs, num_coeffs)
        harmonic coefficients of the estimated sound field
    """
    num_mic = pos.shape[0]

    psi = ki.kernel_helmholtz_3d(pos, pos, wave_num) 
    psi_plus_noise_cov = psi + np.eye(num_mic) * reg_param
    p_prior = apply_measurement_omni(prior_mean, pos, exp_center, max_order, wave_num)
    regression_vec = np.linalg.solve(psi_plus_noise_cov,  p - p_prior)

    shd_coeffs = prior_mean + apply_measurement_conj_omni(regression_vec, pos, exp_center, max_order, wave_num)
    return shd_coeffs




def posterior_mean_omni(p, pos, exp_center, max_order, wave_num, prior_covariance, noise_power, prior_mean=None):
    """Computes the posterior mean of the spherical harmonic coefficients, under a Gaussian assumption
    
    Parameters
    ----------
    p : ndarray of shape (num_real_freqs, num_mics)
        complex sound pressure for each microphone and frequency
    pos : ndarray of shape (num_mics, 3)
        positions of the microphones
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order : int
        maximum order of the spherical harmonics
    wave_num : ndarray of shape (num_real_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound
    noise_power : float
        variance of the noise in the measurement equation. The noise covariance matrix
        is assumed to be a identity matrix scaled by this value
    prior_mean : ndarray of shape (num_real_freqs, num_coeffs), optional
        mean of the prior distribution for the sherical harmonic coefficients
        If not provided, the prior mean is assumed to be zero
    prior_covariance : ndarray of shape (num_real_freqs, num_coeffs, num_coeffs), 
        or ndarray of shape (num_real_freqs,) if the covariance is a scalar matrix
        or float if the covariance is a scalar matrix, identical for all frequencies. 

    Returns
    -------
    shd_coeffs : complex ndarray of shape (num_real_freqs, num_coeffs)
        posterior mean of the harmonic coefficients
    """
    num_mic = pos.shape[0]
    num_freq = wave_num.shape[0]
    num_coeffs = shd.shd_num_coeffs(max_order)
    assert p.shape == (num_freq, num_mic)
    assert wave_num.ndim == 1

    measure = measurement_omni(pos, exp_center, max_order, wave_num)
    measure_conj = measurement_conj_omni(pos, exp_center, max_order, wave_num)

    # Prior mean
    if prior_mean is not None:
       p_prior = measure @ prior_mean[...,None]
       p = p - p_prior[...,0]

    # Prior covariance
    if isinstance(prior_covariance, (int, float)):
        assert prior_covariance >= 0
        prior_covariance = np.ones(num_freq) * prior_covariance
    if prior_covariance.ndim == 1:
        assert prior_covariance.shape[0] == num_freq
        assert np.all(prior_covariance >= 0)
        psi = ki.kernel_helmholtz_3d(pos, pos, wave_num) * prior_covariance[:,None,None]
    elif prior_covariance.ndim == 3:
        assert prior_covariance.shape == (num_freq, num_coeffs, num_coeffs)
        psi = measure @ prior_covariance @ measure_conj
    else:
        raise ValueError("Invalid shape of prior covariance")
  
    # Noise covariance
    if isinstance(noise_power, (int, float)):
        assert noise_power >= 0
        noise_cov = np.eye(num_mic)[None,...] * noise_power
    elif isinstance(noise_power, np.ndarray):
        assert noise_power.ndim == 1
        assert noise_power.shape[0] == num_freq
        assert np.all(noise_power >= 0)
        noise_cov = np.eye(num_mic)[None,...] * noise_power[:,None,None]

    psi_plus_noise_cov = psi + noise_cov
    regression_vec = np.linalg.solve(psi_plus_noise_cov, p)

    if prior_covariance.ndim == 1:
        shd_coeffs = prior_covariance[:,None,None] * measure_conj @ regression_vec[...,None]
    elif prior_covariance.ndim == 3:
        shd_coeffs = prior_covariance @ measure_conj @ regression_vec[...,None]
    shd_coeffs = np.squeeze(shd_coeffs, axis=-1)

    if prior_mean is not None:
        shd_coeffs += prior_mean
    return shd_coeffs

def _posterior_mean_omni_scalar_covariance(p, pos, exp_center, max_order, wave_num, prior_variance, noise_power, prior_mean = None):
    """A special case of posterior_mean_omni_arbitrary_cov, and should be removed eventually
    """
    if prior_mean is not None:
        p_prior = measurement_omni(pos, exp_center, max_order, wave_num) @ prior_mean[...,None]
        p = p - p_prior[...,0]

    num_mic = pos.shape[0]

    psi = ki.kernel_helmholtz_3d(pos, pos, wave_num) 
    psi_plus_noise_cov = psi + np.eye(num_mic) * noise_power / prior_variance 
    regression_vec = np.linalg.solve(psi_plus_noise_cov, p)

    shd_coeffs = measurement_conj_omni(pos, exp_center, max_order, wave_num) @ regression_vec[...,None]#np.linalg.inv(psi_plus_noise_cov) @ p[:,:,None]
    shd_coeffs = np.squeeze(shd_coeffs, axis=-1)

    if prior_mean is not None:
        shd_coeffs += prior_mean
    return shd_coeffs

def _posterior_covariance_omni_scalar_covariance(pos, exp_center, max_order, wave_num, prior_variance, noise_power):
    """ Returns the posterior covariance of the spherical harmonic coefficients

    Assumes Gaussian priors for the spherical harmonic coefficients and Gaussian noise in the measurement equation

    Parameters
    ----------
    pos : ndarray of shape (num_mics, 3)
        positions of the microphones
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order : int
        maximum order of the spherical harmonics
    wave_num : ndarray of shape (num_real_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound
    prior_variance : float
        variance of the prior distribution for the sherical harmonic coefficients. The prior
        covariance matrix is assumed to be an identity matrix scaled by this value.
    noise_power : float
        variance of the noise in the measurement equation. The noise covariance matrix
        is assumed to be a identity matrix scaled by this value
    
    Returns
    -------
    cov : ndarray of shape (num_real_freqs, num_coeffs, num_coeffs)
        posterior covariance of the spherical harmonic coefficients
    
    References
    ----------
    Natsuki Ueno, Shoichi Koyama, Hiroshi Saruwatai, 'Sound Field Recording 
    Using Distributed Microphones Based on Harmonic Analysis of Infinite Order'
    """
    num_mic = pos.shape[0]
    num_freq = wave_num.shape[0]
    num_coeffs = shd.shd_num_coeffs(max_order)

    assert prior_variance.ndim == 1
    assert prior_variance.shape[0] == num_freq
    if isinstance(noise_power, (int, float)):
        noise_power = np.ones(num_freq) * noise_power
    assert noise_power.ndim == 1
    assert noise_power.shape[0] == num_freq

        #assert noise_power.shape[0] == num_freq
    #assert isinstance(noise_power, (int, float)) # not implemented for array noise power

    psi = ki.kernel_helmholtz_3d(pos, pos, wave_num) #* prior_variance[:,None,None]
    noise_cov = np.eye(num_mic)[None,:,:] * noise_power[:,None,None] / prior_variance[:,None,None]
    psi_plus_noise_cov = psi + noise_cov

    measure = measurement_omni(pos, exp_center, max_order, wave_num)
    measure_conj = measurement_conj_omni(pos, exp_center, max_order, wave_num)
    cov = measure_conj @ np.linalg.solve(psi_plus_noise_cov, measure)

    # cov = measurement_conj_omni(pos, exp_center, max_order, wave_num) @ \
    #         np.linalg.inv(psi_plus_noise_cov) @ \
    #             measurement_omni(pos, exp_center, max_order, wave_num)
    cov = prior_variance[:,None,None] * (np.eye(num_coeffs)[None,:,:] - cov)
    return cov

def posterior_covariance_omni(pos, exp_center, max_order, wave_num, prior_covariance, noise_power):
    """Returns the posterior covariance of the spherical harmonic coefficients
    Assumes Gaussian priors for the spherical harmonic coefficients and Gaussian noise in the measurement equation
    
    Parameters
    ----------
    pos : ndarray of shape (num_mics, 3)
        positions of the microphones
    exp_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order : int
        maximum order of the spherical harmonics
    wave_num : ndarray of shape (num_real_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound
    noise_power : float
        variance of the noise in the measurement equation. The noise covariance matrix
        is assumed to be a identity matrix scaled by this value
    prior_covariance : ndarray of shape (num_real_freqs, num_coeffs, num_coeffs), 
        or ndarray of shape (num_real_freqs,) if the covariance is a scalar matrix
        or float if the covariance is a scalar matrix, identical for all frequencies. 
    """
    num_mic = pos.shape[0]
    num_freq = wave_num.shape[0]
    num_coeffs = shd.shd_num_coeffs(max_order)
    assert wave_num.ndim == 1

    if isinstance(prior_covariance, (int, float)):
        assert prior_covariance >= 0
        prior_covariance = np.ones(num_freq) * prior_covariance
    if prior_covariance.ndim == 1:
        return _posterior_covariance_omni_scalar_covariance(pos, exp_center, max_order, wave_num, prior_covariance, noise_power)
    assert prior_covariance.shape == (num_freq, num_coeffs, num_coeffs)

    measure = measurement_omni(pos, exp_center, max_order, wave_num)
    measure_conj = measurement_conj_omni(pos, exp_center, max_order, wave_num)

    # Noise covariance
    if isinstance(noise_power, (int, float)):
        assert noise_power >= 0
        noise_cov = np.eye(num_mic)[None,...] * noise_power
    elif isinstance(noise_power, np.ndarray):
        assert noise_power.ndim == 1
        assert noise_power.shape[0] == num_freq
        assert np.all(noise_power >= 0)
        noise_cov = np.eye(num_mic)[None,...] * noise_power[:,None,None]

    psi = measure @ prior_covariance @ measure_conj
    psi_plus_noise_cov = psi + noise_cov

    measure_times_cov = measure @ prior_covariance
    cov_times_measure_conj = prior_covariance @ measure_conj
    main_mat = cov_times_measure_conj @ np.linalg.solve(psi_plus_noise_cov, measure_times_cov)

    return prior_covariance - main_mat
