import numpy as np
import itertools as it
import scipy.linalg as splin
import scipy.special as special
import scipy.spatial.distance as distance

import aspcol.kernelinterpolation as ki
import aspcol.utilities as utils
import aspcol.filterdesign as fd

import aspcol.sf_analysis_ueno as sau

import wigners


# ==================== BASIC SHD FUNCTIONALITY ====================

def shd_max_order(num_coeffs):
    """Returns the maximum order of the spherical harmonics that is required to represent a given number of coefficients

    Parameters
    ----------
    num_coeffs : int
        number of coefficients

    Returns
    -------
    max_order : int
        maximum order of the spherical harmonics that is required to represent the coefficients
    """
    order = np.sqrt(num_coeffs) - 1
    assert order == int(order)
    return int(order)

def shd_num_coeffs(max_order):
    """Number of coefficients required to represent a spherical harmonic function of a given order
    
    Parameters
    ----------
    max_order : int
        maximum order of the spherical harmonics
    
    Returns
    -------
    num_coeffs : int
        number of coefficients required to represent a spherical harmonic function of the given order
    """
    return (max_order+1)**2


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







def shd_basis(pos, order, degree, wavenum):
    """Spherical harmonic basis function for sound field in 3D

    Implements: sqrt(4pi) j_order(kr) Y_order^degree(polar angle, zenith angle)
    degree and order of the spherical harmonics might be swapped according to some
    definitions.
    
    Parameters
    ----------
    pos : ndarray of shape (num_pos, 3)
        positions of the evaluation points
    order : ndarray of shape (num_coeffs,)
        order of the spherical harmonics. Can be any non-negative integer
    degree : ndarray of shape (num_coeffs,)
        degree of the spherical harmonics. Must satisfy |degree[i]| <= order[i] for all i
    wave_num : float or ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular 
        frequency and c is the speed of sound

    Returns
    -------
    f : ndarray of shape (num_freq, num_coeffs, num_pos)
        values of the spherical harmonic basis function at the evaluation points

    Notes
    -----
    See more detailed derivations in any of:
    [1] Natsuki Ueno, Shoichi Koyama, Hiroshi Saruwatai, 'Sound Field Recording Using 
    Distributed Microphones Based on Harmonic Analysis of Infinite Order'
    [2] J. Brunnström, M.B. Moeller, M. Moonen, 'Bayesian sound field estimation using
    moving microphones'
    """
    #Parse arguments to get correct shapes
    radius, angles = utils.cart2spherical(pos)
    radius = radius[None, None, ...]
    angles = angles[None, None, ...]

    order = order[None, :, None]
    degree = degree[None, :, None]

    if isinstance(wavenum, (int, float)):
        wavenum = np.array([wavenum])
    assert wavenum.ndim == 1
    wavenum = wavenum[:, None, None]

    # Calculate the function values
    f = np.sqrt(4*np.pi) * special.spherical_jn(order, wavenum * radius) * special.sph_harm(degree, order, angles[...,0], angles[...,1])
    return f


def reconstruct_pressure(shd_coeffs, pos, expansion_center, wavenum):
    """Returns the complex sound pressure at some positions using the provided spherical harmonic coefficients
    
    Parameters
    ----------
    shd_coeffs : ndarray of shape (num_freq, num_coeffs,)
        spherical harmonic coefficients of the sound field
    pos : ndarray of shape (num_pos, 3)
        positions of the evaluation points
    expansion_center : ndarray of shape (1, 3)
        expansion center of the spherical harmonics
    max_order : int
        maximum order of the spherical harmonics. Could be removed in a future version of 
        the function, but must currently equal the max_order of the provided coefficients
    wavenum : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound. 

    Returns
    -------
    p_est : ndarray of shape (num_freq, num_pos,)
        complex sound pressure at the evaluation points
    """
    num_coeffs = shd_coeffs.shape[1]
    max_order = shd_max_order(num_coeffs)

    rel_pos = pos - expansion_center
    order, degree = shd_num_degrees_vector(max_order)
    basis_values = shd_basis(rel_pos, order, degree, wavenum)
    p_est = np.sum(shd_coeffs[...,None] * basis_values, axis=1)
    return p_est













# ==================== TRANSLATION OF SHD COEFFICIENTS ====================

def translation(pos, wave_num, max_order_input, max_order_output):
    """Creates a translation operator for a sequence of spherical harmonic coefficients to another expansion center
    
    The operator can be applied to a sequence of spherical harmonic coefficients to translate
    the expansion center of the sequence. 
    shd_coeffs(pos) = T(pos - pos_orig) @ shd_coeffs(pos_orig)

    Parameters
    ----------
    pos : ndarray of shape (num_pos, 3)
        position argument to the translation operator
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    max_order_input : int
        maximum order of the coefficients that should be translated
    max_order_output : int
        maximum order of the translated coefficients
        
    Returns
    -------
    shd_coeffs_translated : ndarray of shape (num_freq, num_pos, num_coeffs_output, num_coeffs_input)
        translated coefficients
    """
    test_mat = sau.trjmat3d(max_order_output, max_order_input, pos[0,0], pos[0,1], pos[0, 2], wave_num[0])
    num_freqs = wave_num.shape[0]
    num_pos = pos.shape[0]

    T_all = np.zeros((num_freqs, num_pos, test_mat.shape[0], test_mat.shape[1]), dtype=complex)
    for m in range(num_pos):
        for f in range(num_freqs):
            T_all[f, m, :, :] = sau.trjmat3d(max_order_output, max_order_input, pos[m,0], pos[m,1], pos[m, 2], wave_num[f])
    return T_all


def translate_shd_coeffs(shd_coeffs, pos, wave_num, max_order_input, max_order_output):
    """Translates the provided shd_coeffs to another expansion center
    
    shd_coeffs(pos_orig + pos) = translate(pos) shd_coeffs(pos_orig)
    
    Parameters
    ----------
    shd_coeffs : ndarray of shape (num_freqs, num_coeffs,)
        coefficients of the sequence
    pos : ndarray of shape (num_pos, 3)
        position argument to the translation operator
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    max_order_output : int
        maximum order of the coefficients that should be translated
    max_order_output : int
        maximum order of the translated coefficients

    Returns
    -------
    shd_coeffs_translated : ndarray of shape (num_freqs, num_pos, num_coeffs,)
        translated coefficients
        if num_pos == 1, the returned array will have shape (num_freqs, num_coeffs,)
    """
    T_all = translation(pos, wave_num, max_order_input, max_order_output)
    translated_coeffs = T_all @ shd_coeffs[:,None,:,None]

    num_pos = pos.shape[0]
    if num_pos == 1:
        translated_coeffs = np.squeeze(translated_coeffs, axis=1)
    return np.squeeze(translated_coeffs, axis=-1)


def translation_operator(pos, wave_num, max_order_input, max_order_output):
    """Translation operator for harmonic coefficients, such that 
    shd_coeffs(pos_orig + pos) = T(pos) @ shd_coeffs(pos_orig)
    
    Parameters
    ----------
    pos : ndarray of shape (num_pos, 3)
        position argument to the translation operator
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    max_order_input : int
        maximum order of the coefficients that should be translated
    max_order_output : int
        maximum order of the translated coefficients

    Returns
    -------
    T : ndarray of shape (num_freqs, num_pos, num_coeffs_output, num_coeffs_input)
        translation operator, such that shd_coeffs(pos_orig + pos) = T(pos) @ shd_coeffs(pos_orig)
    """
    num_coeffs_input = shd_num_coeffs(max_order_input)
    num_coeffs_output = shd_num_coeffs(max_order_output)
    num_pos = pos.shape[0]
    num_freq = wave_num.shape[0]

    tr_op = np.zeros((num_freq, num_pos, num_coeffs_output, num_coeffs_input), dtype=complex)

    orders_input, degrees_input = shd_num_degrees_vector(max_order_input)
    orders_output, degrees_output = shd_num_degrees_vector(max_order_output)
    for f in range(num_freq):
        for m in range(num_pos):
            for out_idx, (l1, m1) in enumerate(zip(orders_output, degrees_output)):
                for in_idx, (l2, m2) in enumerate(zip(orders_input, degrees_input)):
                    sum_val = 0
                    for q in range(l1+l2+1):
                        if np.abs(m2-m1) <= q:
                            basis_val = np.conj(shd_basis(pos[m:m+1,:], np.array([q]), np.array([m2-m1]), wave_num[f]))
                            g = gaunt_coefficient(l1, m1, l2, -m2, q)
                            new_val = (1j)**q * basis_val * g
                            new_val *= (-1.0)**(m1) * (1j)**(l2-l1)
                            sum_val += new_val

                tr_op[f, m, out_idx, in_idx] = np.squeeze(sum_val)
    tr_op *= np.sqrt(4*np.pi)
    return tr_op

def gaunt_coefficient(l1, m1, l2, m2, l3):
    """Gaunt coefficient G(l1, m1, l2, m2, l3)
    
    As defined by P. A. Martin, 2006, 'Multiple scattering: 
    Interaction of time-harmonic waves with N obstacles'. 
    Defined on page 83, equation (3.71). Argument order is the same as in the reference
    
    Parameters
    ----------
    l1 : int
        Spherical harmonic order
    m1 : int
        Spherical harmonic degree
    l2 : int
        Spherical harmonic order
    m2 : int
        Spherical harmonic degree
    l3 : int
        Spherical harmonic order
    
    Notes
    -----
    The relationship between this and triple_harmonic_integral 
    (the latter is the definition of gaunt coefficient given by Sympy)
    is I(l1, l2, l3, m1, m2, m3) = delta(m1+m2+m3,0) * (-1)**(m3) * gaunt(l1, m1, l2, m2, l3)
    
    Can be seen on the final equation on page 328 in Multiple scattering: Interaction of time-harmonic waves with N obstacles
    by P. A. Martin, 2006. 

    A recursive algorithm can be found in:
    Fast evaluation of Gaunt coefficients: recursive approach - Yu-lin Xu, 1997
    """
    f1 = (-1.0)**(m1+m2)
    f2 = np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi))
    f3 = wigners.wigner_3j(l1,l2,l3,m1,m2,-m1-m2)
    f4 = wigners.wigner_3j(l1,l2,l3,0,0,0)
    return f1 * f2 * f3 * f4

def triple_harmonic_integral(l1, l2, l3, m1, m2, m3):
    """Integral of the product of three spherical harmonics

    Defined as int_{\omega} Y_{l1,m1}(\omega) Y_{l2,m2}(\omega) Y_{l3,m3}(\omega) d\omega
    where \omega is the angle. It is sometimes (in Sympy for example) called gaunt coefficient.
    
    Parameters
    ----------
    l1, l2, l3 : int
        Spherical harmonic orders
    m1, m2, m3 : int
        Spherical harmonic degrees
    
    """
    f1 = np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi))
    f2 = wigners.wigner_3j(l1,l2,l3,m1,m2,m3)
    f3 = wigners.wigner_3j(l1,l2,l3,0,0,0)
    return f1 * f2 * f3

def gaunt_coef_lookup(max_order):
    """
    Returns a lookup table for the gaunt coefficients
    """
    all_params = np.array([i for i in _all_gaunt_parameters(max_order)])

    tbl = np.zeros((max_order, 2*max_order+1, max_order, 2*max_order+1, max_order))

    val_ref = []
    for i in range(all_params.shape[0]):
        (l1, m1, l2, m2, l3) = all_params[i]
        tbl[l1, m1, l2, m2, l3] = gaunt_coefficient(l1, m1, l2, m2, l3)
    val_ref = np.array(val_ref)
    return tbl

def _all_gaunt_parameters(max_order):
    """ This might not include all parameters"""
    all_orders = it.product(range(max_order), range(max_order), range(max_order))
    for l1, l2, l3 in all_orders:
        all_modes = it.product(range(0, l1), range(0, l2))
        for m1, m2 in all_modes:
            if abs(l1-l2) <= l3 <= l1+l2:
                if abs(m1 + m2) <= l3:
                    if (l1 + l2 + l3) % 2 == 0:
                        yield l1, m1, l2, m2, l3






# =================== DIRECTIVITY DEFINITIONS ===================

def directivity_omni(max_order = 0):
    """Harmonic coefficients of an omnidirectional directivity function
    
    """
    if max_order == 0:
        return np.ones((1,1), complex)
    else:
        num_coeffs = shd_num_coeffs(max_order)
        dir = np.zeros((1, num_coeffs), dtype=complex)
        dir[0,0] = 1
        return dir
    
def directivity_linear(A, d_mic, max_order = 1):
    """Harmonic coefficients of a linear directivity function

    The directivity function is defined as
    c(d, omega) = A + A * d^T d_mic

    Omni directivity is obtained by setting A = 0
    Cardoid directivity is obtained by setting A = 1/2
    Figure-8 directivity is obtained by setting A = 1
    
    Parameters
    ----------
    A : float
        directivity parameter. Must be between 0 and 1. 
    d_mic : ndarray of shape (1, 3)
        direction of the microphone. It is the angle at which a cardioid microphone would give the strongest response
        It is the outward direction, so from the center of the microphone towards the sources
        Must be a unit vector
    max_order : int
        maximum order of the spherical harmonics. If set higher to 1, 
        the coefficients of the higher orders will be zero. 
    
    Returns
    -------
    dir_coeffs : ndarray of shape (1, num_coeffs)
        coefficients of the directivity function
    """
    assert max_order >= 1
    assert 0 <= A <= 1
    radius, angles = utils.cart2spherical(-d_mic)
    assert np.isclose(radius, 1)
    dir_coeffs = np.zeros((1, shd_num_coeffs(max_order)), dtype=complex)
    dir_coeffs[0,0] = 1 - A

    harm = special.sph_harm(np.array([-1,0,1]), 1, angles[...,0], angles[...,1])

    dir_coeffs[0,1:4] = (-1j * A / 3) * np.sqrt(4*np.pi) * harm.conj()
    return dir_coeffs
















# ====================== SHD ESTIMATION ======================

def measurement_omni(pos, exp_center, max_order, wave_num):
    T = translation(pos - exp_center, wave_num, max_order, 0)
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
    T = translation(exp_center - pos, wave_num, 0, max_order)
    T = np.sum(T, axis=-1) # inner product with omni directionality
    return np.moveaxis(T, 1,2)

def apply_measurement_omni(shd_coeffs, pos, exp_center, max_order, wave_num):
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
    max_order = int
        maximum order of the spherical harmonics that is input by the operator. Is in theory 
        redundant information and could be removed in a future version of the function.
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.

    Returns
    -------
    p_est : ndarray of shape (num_freq, num_mic,)
        complex sound pressure at the measurement points
    """
    rel_pos = pos - exp_center
    shd_translated = translate_shd_coeffs(shd_coeffs, rel_pos, wave_num, max_order, 0)
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
    directivity = directivity_omni()
    dir_translated = translate_shd_coeffs(directivity, rel_pos, wave_num, 0, max_order)
    return np.sum(dir_translated * vec[:,:,None], axis=1)


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
    dir_coeffs : ndarray of shape (num_freqs, num_coeffs)
        coefficients of the directivity function. If None, the directivity is assumed to be omnidirectional
    
    Returns
    -------
    shd_coeffs : ndarray of shape (num_freqs, num_coeffs,)
        coefficients of the infinite sequence
    """
    rel_pos = exp_center - pos
    if dir_coeffs is None:
        directivity = directivity_omni()
    else:
        directivity = dir_coeffs
    max_order_dir = shd_max_order(directivity.shape[1])
    
    dir_translated = translate_shd_coeffs(directivity, rel_pos, wave_num, max_order_dir, max_order)
    return np.sum(dir_translated * vec[:,:,None], axis=1)

def translated_inner_product(pos1, pos2, dir_coeffs1, dir_coeffs2, wave_num):
    """
    
    <T(r_1 - r_2, omega_m) gamma_2(omega_m), gamma_1(omega_m)>
    
    Parameters
    ----------
    pos1 : ndarray of shape (num_pos1, 3)
        positions of the first set of measurement points
    pos2 : ndarray of shape (num_pos2, 3)
        positions of the second set of measurement points
    dir_coeffs1 : ndarray of shape (num_freqs, num_coeffs1)
        coefficients of the directivity function for the first set of measurement points
    dir_coeffs2 : ndarray of shape (num_freqs, num_coeffs2)
        coefficients of the directivity function for the second set of measurement points
    wave_num : ndarray of shape (num_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.

    Returns
    -------
    psi : ndarray of shape (num_freqs, num_pos1, num_pos2)
        inner product of the translated directivity functions
    """
    max_order1 = shd_max_order(dir_coeffs1.shape[1])
    max_order2 = shd_max_order(dir_coeffs2.shape[1])
    num_pos1 = pos1.shape[0]
    num_pos2 = pos2.shape[0]
    num_freqs = len(wave_num)

    pos_diff = pos1[:,None,:] - pos2[None,:,:]
    pos_diff = pos_diff.reshape(-1, 3)

    translated_coeffs2 = translate_shd_coeffs(dir_coeffs2, pos_diff, wave_num, max_order2, max_order1)
    translated_coeffs2 = translated_coeffs2.reshape(num_freqs, num_pos1, num_pos2, -1)

    inner_product_matrix = np.sum(translated_coeffs2 * dir_coeffs1.conj()[:,None,None,:], axis=-1)
    return inner_product_matrix

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
    dir_coeffs : ndarray of shape (num_real_freqs, num_coeffs)
        coefficients of the directivity function. If None, the directivity is assumed to be omnidirectional
    
    Returns
    -------
    shd_coeffs : complex ndarray of shape (num_real_freqs, num_coeffs)
        harmonic coefficients of the estimated sound field
    """
    num_mic = pos.shape[0]

    psi = translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, wave_num)
    psi_plus_noise_cov = psi + np.eye(num_mic) * reg_param
    regression_vec = np.linalg.solve(psi_plus_noise_cov,  p)

    shd_coeffs = apply_measurement_conj(regression_vec, pos, exp_center, max_order, wave_num, dir_coeffs=dir_coeffs)
    return shd_coeffs


def inf_dimensional_shd_omni(p, pos, exp_center, max_order, wave_num, reg_param):
    """Estimates the spherical harmonic coefficients with Bayesian inference
    
    Implements the method in Natsuki Ueno, Shoichi Koyama, Hiroshi Saruwatai,
    'Sound Field Recording Using Distributed Microphones Based on Harmonic Analysis of Infinite Order'

    Assumes all microphones are omnidirectional

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
    
    Returns
    -------
    shd_coeffs : complex ndarray of shape (num_real_freqs, num_coeffs)
        harmonic coefficients of the estimated sound field
    """
    num_mic = pos.shape[0]

    psi = ki.kernel_helmholtz_3d(pos, pos, wave_num) 
    psi_plus_noise_cov = psi + np.eye(num_mic) * reg_param
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




def posterior_mean_omni_arbitrary_cov(p, pos, exp_center, max_order, wave_num, noise_power, prior_mean, prior_covariance):
    num_mic = pos.shape[0]
    measure = measurement_omni(pos, exp_center, max_order, wave_num)
    measure_conj = measurement_conj_omni(pos, exp_center, max_order, wave_num)

    if prior_mean is not None:
       p_prior = measure @ prior_mean[...,None]
       p = p - p_prior[...,0]

    psi = measure @ prior_covariance @ measure_conj
    psi_plus_noise_cov = psi + np.eye(num_mic) * noise_power 
    regression_vec = np.linalg.solve(psi_plus_noise_cov, p)

    shd_coeffs = prior_covariance @ measure_conj @ regression_vec[...,None]
    shd_coeffs = np.squeeze(shd_coeffs, axis=-1)

    if prior_mean is not None:
        shd_coeffs += prior_mean
    return shd_coeffs

def posterior_mean_omni(p, pos, exp_center, max_order, wave_num, prior_variance, noise_power, prior_mean = None):
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

def posterior_covariance_omni(pos, exp_center, max_order, wave_num, prior_variance, noise_power):
    """ Returns the posterior covariance of the spherical harmonic coefficients

    Given the model in Natsuki Ueno, Shoichi Koyama, Hiroshi Saruwatai,
    'Sound Field Recording Using Distributed Microphones Based on Harmonic Analysis of Infinite Order'
    
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
    
    """
    num_mic = pos.shape[0]
    num_coeffs = shd_num_coeffs(max_order)

    psi = ki.kernel_helmholtz_3d(pos, pos, wave_num)
    psi_plus_noise_cov = psi + np.eye(num_mic) * noise_power / prior_variance

    cov = measurement_conj_omni(pos, exp_center, max_order, wave_num) @ \
            np.linalg.inv(psi_plus_noise_cov) @ \
                measurement_omni(pos, exp_center, max_order, wave_num)
    cov = prior_variance * (np.eye(num_coeffs) - cov)
    return cov

def posterior_covariance_omni_arbitrary_cov(pos, exp_center, max_order, wave_num, noise_power, prior_covariance):
    num_mic = pos.shape[0]
    measure = measurement_omni(pos, exp_center, max_order, wave_num)
    measure_conj = measurement_conj_omni(pos, exp_center, max_order, wave_num)

    psi = measure @ prior_covariance @ measure_conj
    psi_plus_noise_cov = psi + np.eye(num_mic) * noise_power 

    measure_times_cov = measure @ prior_covariance
    cov_times_measure_conj = prior_covariance @ measure_conj
    main_mat = cov_times_measure_conj @ np.linalg.solve(psi_plus_noise_cov, measure_times_cov)

    return prior_covariance - main_mat




def inf_dimensional_shd_dynamic_omni(p, pos, pos_eval, sequence, samplerate, c, reg_param, verbose=False):
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
    shd_coeffs : ndarray of shape (num_real_freqs, num_eval)
        time-domain harmonic coefficients of the estimated sound field
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
    


