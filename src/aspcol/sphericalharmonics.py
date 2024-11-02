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
import itertools as it
import scipy.special as special

import aspcol.kernelinterpolation as ki
import aspcol.utilities as utils

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
    return int((max_order+1)**2)


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

def shd_min_order(wave_num, radius):
    """
    Returns the minimum order of the spherical harmonics that should be used
    for a given wavenumber and radius

    Here according to the definition in Katzberg et al., Spherical harmonic
    representation for dynamic sound-field measurements

    Parameters
    ----------
    wave_num : ndarray of shape (num_freqs)
    radius : float
        represents r_max in the Katzberg paper

    Returns
    -------
    M_f : ndarray of shape (num_freqs)
        contains an integer which is the minimum order of the spherical harmonics
    """
    return np.ceil(wave_num * radius).astype(int)

def shd_coeffs_order_degree_to_index(order, degree):
    return order**2 + order + degree





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
    [brunnstromBayesianSubmitted] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted.
    [uenoSound2018] N. Ueno, S. Koyama, and H. Saruwatari, “Sound field recording using distributed microphones based on harmonic analysis of infinite order,” IEEE Signal Process. Lett., vol. 25, no. 1, pp. 135–139, Jan. 2018, doi: 10.1109/LSP.2017.2775242.
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
    if not shd_coeffs.ndim == 2:
        shd_coeffs = shd_coeffs[None, ...]
        wavenum = np.array([wavenum])
    assert shd_coeffs.ndim == 2
    assert wavenum.ndim == 1
    assert shd_coeffs.shape[0] == wavenum.shape[0]

    num_coeffs = shd_coeffs.shape[1]
    max_order = shd_max_order(num_coeffs)

    rel_pos = pos - expansion_center
    order, degree = shd_num_degrees_vector(max_order)
    basis_values = shd_basis(rel_pos, order, degree, wavenum)
    p_est = np.sum(shd_coeffs[...,None] * basis_values, axis=1)
    return p_est













# ==================== TRANSLATION OF SHD COEFFICIENTS ====================
def translate_shd_coeffs(shd_coeffs, pos, wave_num, max_order_output):
    """Translates the provided shd_coeffs to another expansion center
    
    shd_coeffs(pos_orig + pos) = translate(pos) shd_coeffs(pos_orig)
    
    Parameters
    ----------
    shd_coeffs : ndarray of shape (num_freqs, num_coeffs,) or (num_freqs, num_pos, num_coeffs,)
        coefficients of the sequence
    pos : ndarray of shape (num_pos, 3)
        position argument to the translation operator
    wave_num : ndarray of shape (num_freq,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.
    max_order_output : int
        maximum order of the translated coefficients

    Returns
    -------
    shd_coeffs_translated : ndarray of shape (num_freqs, num_pos, num_coeffs,)
        translated coefficients
        if num_pos == 1, the returned array will have shape (num_freqs, num_coeffs,)
    """
    if shd_coeffs.ndim == 2:
        shd_coeffs = shd_coeffs[:,None,:]
    assert pos.ndim == 2
    assert pos.shape[1] == 3
    assert wave_num.ndim == 1
    assert shd_coeffs.ndim == 3
    assert shd_coeffs.shape[0] == wave_num.shape[0] or shd_coeffs.shape[0] == 1
    assert shd_coeffs.shape[1] == pos.shape[0] or shd_coeffs.shape[1] == 1
    num_coeffs = shd_coeffs.shape[-1]

    max_order_input = shd_max_order(num_coeffs)

    T_all = translation_operator(pos, wave_num, max_order_input, max_order_output)
    translated_coeffs = T_all @ shd_coeffs[...,None]

    num_pos = pos.shape[0]
    if num_pos == 1:
        translated_coeffs = np.squeeze(translated_coeffs, axis=1)
    return np.squeeze(translated_coeffs, axis=-1)

def translation_operator(pos, wave_num, max_order_input, max_order_output):
    """Translation operator for harmonic coefficients, such that 
    shd_coeffs(pos_orig + pos) = T(pos) @ shd_coeffs(pos_orig)

    Defined according to $T = \\hat{S}^T $, where $\\hat{S}$ is the basis translation matrix defined in [martinMultiple2006].
    This definition is consistent with the definition (6) in [uenoSound2018]. 
    
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

    References
    ----------
    [martinMultiple2006] P. A. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles, vol. 107. in Encyclopedia of mathematics and its applications, vol. 107. Cambridge, UK: Cambridge University Press, 2006.
    [uenoSound2018] N. Ueno, S. Koyama, and H. Saruwatari, “Sound field recording using distributed microphones based on harmonic analysis of infinite order,” IEEE Signal Process. Lett., vol. 25, no. 1, pp. 135–139, Jan. 2018, doi: 10.1109/LSP.2017.2775242.
    """

    S = basis_translation_3_80(pos, wave_num, max_order_output, max_order_input)
    T = np.moveaxis(S, -1, -2)
    return T

def basis_translation_3_80(pos, wave_num, max_order_input, max_order_output):
    """Translation operator for shd basis function, such that 
    shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)x

    Implemented according to equation 3.80 in 
    P. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles.
    
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
        translation operator, such that shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)
    """
    num_coeffs_input = shd_num_coeffs(max_order_input)
    num_coeffs_output = shd_num_coeffs(max_order_output)
    num_pos = pos.shape[0]
    num_freq = wave_num.shape[0]

    tr_op = np.zeros((num_freq, num_pos, num_coeffs_output, num_coeffs_input), dtype=complex)
    orders_input, degrees_input = shd_num_degrees_vector(max_order_input)
    orders_output, degrees_output = shd_num_degrees_vector(max_order_output)

    gaunt_set = _calculate_gaunt_set(max_order_input, max_order_output)

    max_tot_order = int(max_order_input + max_order_output)
    orders_tot, degrees_tot = shd_num_degrees_vector(max_tot_order)
    all_q = np.arange(max_tot_order+1)
    #nu_plus_n = orders_output[:,None] + orders_input[None,:]
    #mu_minus_m = degrees_output[:,None] - degrees_input[None,:]
    # all_q = np.arange(nu_plus_n)[nu_plus_n >= all_q]
    
    radius, angles = utils.cart2spherical(pos)
    radius = radius[None, None, ...]
    angles = angles[None, ...]

    orders_tot = orders_tot[:, None]
    degrees_tot = degrees_tot[:, None]

    wave_num = wave_num[:, None, None]
    all_q = all_q[None,:,None]

    bessel = special.spherical_jn(all_q, wave_num * radius) # ordered as (num_freq, num_q, num_pos)
    harm = np.conj(special.sph_harm(degrees_tot, orders_tot, angles[...,0], angles[...,1])) # ordered as (num_coeffs_tot, num_pos)

    for out_idx, (n, m) in enumerate(zip(orders_output, degrees_output)):
        for in_idx, (nu, mu) in enumerate(zip(orders_input, degrees_input)):
            q_array = np.arange(np.abs(mu-m), n+nu+1)
            harm_idxs = shd_coeffs_order_degree_to_index(q_array, mu-m)

            basis_val = bessel[:,q_array,:] * harm[None,harm_idxs, :]
            g = (1j)**q_array[None,:,None] * gaunt_set[out_idx, in_idx, q_array][None,:,None]
            factor = 4*np.pi * (1j)**(nu-n) * (-1.0)**(m)
            sum_val = factor * np.sum(basis_val * g, axis=1)
            tr_op[..., out_idx, in_idx] = sum_val

    for p in range(num_pos):
        if np.sum(np.abs(pos[p,:])) == 0: 
            tr_op[:, p, :, :] = np.eye(num_coeffs_output, num_coeffs_input)[None,:,:]
    return tr_op

def _calculate_gaunt_set(max_order_input, max_order_output):
    """Numpy function for precaculating the Gaunt coefficients for the translation operator.

    Used to avoid having to write a jax implementation of wigner coefficients. 
    """
    num_coeffs_input = shd_num_coeffs(max_order_input)
    num_coeffs_output = shd_num_coeffs(max_order_output)
    orders_input, degrees_input = shd_num_degrees_vector(max_order_input)
    orders_output, degrees_output = shd_num_degrees_vector(max_order_output)
    gaunt = np.zeros((num_coeffs_output, num_coeffs_input, max_order_input+max_order_output+1), dtype=float)
    for out_idx, (n, m) in enumerate(zip(orders_output, degrees_output)):
        for in_idx, (nu, mu) in enumerate(zip(orders_input, degrees_input)):
            for q in range(n+nu+1):
                if np.abs(mu-m) <= q:
                    g = gaunt_coefficient(n, m, nu, -mu, q)
                    gaunt[out_idx, in_idx, q] = g
    return gaunt


def basis_translation_3_92(pos, wave_num, max_order_input, max_order_output):
    """Translation operator for harmonic basis functions, such that 
    shd_basis(pos_orig + pos) = T(pos) @ shd_basis(pos_orig)

    Taken directly from 3.92 in P. Martin, Multiple scattering: Interaction of 
    time-harmonic waves with N obstacles

    
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
        for p in range(num_pos):
            if np.sum(np.abs(pos[p,:])) == 0:
                tr_op[f, p, :, :] = np.eye(num_coeffs_output, num_coeffs_input)
            else:
                for out_idx, (n, m) in enumerate(zip(orders_output, degrees_output)):
                    for in_idx, (nu, mu) in enumerate(zip(orders_input, degrees_input)):
                        sum_val = 0
                        q0 = _linearity_formula_lower_limit(n, nu, m, -mu)
                        Q = (n + nu - q0) / 2
                        assert Q == int(Q)
                        Q = int(Q)
                        for q in range(Q+1):
                        #for q in range(n+nu+1):
                            if np.abs(m-mu) <= (q0+2*q):
                                basis_val = shd_basis(pos[p:p+1,:], np.array([q0+2*q]), np.array([m-mu]), wave_num[f])
                                #basis_val = shd_basis(pos[p:p+1,:], np.array([q]), np.array([mu-m]), wave_num[f])
                                g = gaunt_coefficient(n, m, nu, -mu, q0+2*q)
                                sum_val += (-1)**q * basis_val * g
                        sum_val *= np.sqrt(4*np.pi) * (-1)**(nu + mu + Q)
                        tr_op[f, p, out_idx, in_idx] = np.squeeze(sum_val)
    return tr_op

def _linearity_formula_lower_limit(l1, l2, m1, m2):
    """ Lower limit of the sum in the linearity formula for the Gaunt coefficient
    described in (3.74) in Multiple scattering: Interaction of time-harmonic waves with N obstacles.

    func(l1, l2, m1, m2) is equivalent to func(l1, l1, m2, m1)
    """
    if np.abs(l1 - l2) >= np.abs(m1 + m2):
        return np.abs(l1 - l2)
    elif np.abs(l1 - l2) < np.abs(m1 + m2) and ((l1 + l2 + np.abs(m1 + m2)) % 2 == 0):
        return np.abs(m1 + m2)
    elif np.abs(l1 - l2) < np.abs(m1 + m2) and ((l1 + l2 + np.abs(m1 + m2)) % 2 == 1):
        return np.abs(m1 + m2) + 1
    else:
        raise ValueError("Something went wrong")




def gaunt_coefficient(l1, m1, l2, m2, l3):
    """Gaunt coefficient G(l1, m1, l2, m2, l3)
    
    Defined on page 83, equation (3.71) in [martinMultiple2006]. Argument order is the same as in the reference
    
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

    Returns
    -------
    G : float
        Gaunt coefficient for the given arguments

    References
    ----------
    [martinMultiple2006] P. A. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles, vol. 107. in Encyclopedia of mathematics and its applications, vol. 107. Cambridge, UK: Cambridge University Press, 2006.
    
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

    Defined as $\\int_{\\omega} Y_{l1,m1}(\\omega) Y_{l2,m2}(\\omega) Y_{l3,m3}(\\omega) d\\omega$
    where $\\omega$ is the angle. It is sometimes (in Sympy for example) called gaunt coefficient.
    
    Parameters
    ----------
    l1, l2, l3 : int
        Spherical harmonic orders
    m1, m2, m3 : int
        Spherical harmonic degrees

    Returns
    -------
    integral_value : float
        the value of the triple harmonic integral
    
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

    Parameters
    ----------
    max_order : int
        maximum order of the spherical harmonics. If set higher than 0, the coefficients of the higher orders will be zero.

    Returns
    -------
    dir_coeffs : ndarray of shape (1, num_coeffs)
        coefficients of the directivity function

    References
    ----------
    [uenoSound2018] N. Ueno, S. Koyama, and H. Saruwatari, “Sound field recording using distributed microphones based on harmonic analysis of infinite order,” IEEE Signal Process. Lett., vol. 25, no. 1, pp. 135–139, Jan. 2018, doi: 10.1109/LSP.2017.2775242.
    [brunnstromBayesianSubmitted] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted.

    Notes
    -----
    The directivity coefficients are defined in more detail in the supplementary notes of [uenoSound2018] and in [brunnstromBayesianSubmitted]
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
    d_mic : ndarray of shape (num_mic, 3)
        direction of the microphone(s). It is the angle at which a cardioid microphone would give the strongest response
        It is the outward direction, so from the center of the microphone towards the sources
        Must be a unit vector
    max_order : int
        maximum order of the spherical harmonics. If set higher than 1, 
        the coefficients of the higher orders will be zero. 
    
    Returns
    -------
    dir_coeffs : ndarray of shape (num_mic, num_coeffs)
        coefficients of the directivity function

    References
    ----------
    [uenoSound2018] N. Ueno, S. Koyama, and H. Saruwatari, “Sound field recording using distributed microphones based on harmonic analysis of infinite order,” IEEE Signal Process. Lett., vol. 25, no. 1, pp. 135–139, Jan. 2018, doi: 10.1109/LSP.2017.2775242.
    [brunnstromBayesianSubmitted] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted.

    Notes
    -----
    The directivity coefficients are defined in more detail in the supplementary notes of [uenoSound2018] and in [brunnstromBayesianSubmitted]
    """
    num_mic = d_mic.shape[0]
    assert d_mic.shape == (num_mic, 3)
    assert max_order >= 1
    assert 0 <= A <= 1
    radius, angles = utils.cart2spherical(d_mic)
    assert np.allclose(radius, 1)
    
    dir_coeffs = np.zeros((num_mic, shd_num_coeffs(max_order)), dtype=complex)
    order, degree = shd_num_degrees_vector(max_order) 

    dir_coeffs[:, order == 0] = 1 - A
    degrees_to_set = degree[order == 1]
    harm = special.sph_harm(degrees_to_set[None,:], 1, angles[...,0:1], angles[...,1:2])

    dir_coeffs[:, order == 1] = (-1j * A / 3) * np.sqrt(4*np.pi) * harm.conj()
    return dir_coeffs
















# ====================== SHD ESTIMATION ======================

def measurement_omni(pos, exp_center, max_order, wave_num):
    T = translation_operator(pos - exp_center, wave_num, max_order, 0)
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
    T = translation_operator(exp_center - pos, wave_num, 0, max_order)
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
    shd_translated = translate_shd_coeffs(shd_coeffs, rel_pos, wave_num, 0)
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
    dir_translated = translate_shd_coeffs(directivity, rel_pos, wave_num, max_order)
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
        directivity = directivity_omni()
    else:
        directivity = dir_coeffs

    if directivity.ndim == 2:
        directivity = directivity[:,None,:]
    assert directivity.ndim == 3
    assert directivity.shape[0] == num_freq or directivity.shape[0] == 1
    assert directivity.shape[1] == num_mic or directivity.shape[1] == 1

    max_order_dir = shd_max_order(directivity.shape[-1])
    #max_order = shd_max_order(num_coeffs)
    shd_translated = translate_shd_coeffs(shd_coeffs, rel_pos, wave_num, max_order_dir)
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
        directivity = directivity_omni()
    else:
        directivity = dir_coeffs
    max_order_dir = shd_max_order(directivity.shape[-1])
    
    dir_translated = translate_shd_coeffs(directivity, rel_pos, wave_num, max_order)
    return np.sum(dir_translated * vec[:,:,None], axis=1)

def translated_inner_product(pos1, pos2, dir_coeffs1, dir_coeffs2, wave_num):
    """Computes the inner product of translated sequences of coefficients
    
    $\\langle T(r_1 - r_2, \\omega_m) \\gamma_2(\\omega_m), \\gamma_1(\\omega_m) \\rangle$
    
    Parameters
    ----------
    pos1 : ndarray of shape (num_pos1, 3)
        positions of the first set of measurement points
    pos2 : ndarray of shape (num_pos2, 3)
        positions of the second set of measurement points
    dir_coeffs1 : ndarray of shape (num_freqs, num_coeffs1) or (num_freqs, num_pos1, num_coeffs1)
        coefficients of the directivity function for the first set of measurement points
    dir_coeffs2 : ndarray of shape (num_freqs, num_coeffs2) or (num_freqs, num_pos1, num_coeffs1)
        coefficients of the directivity function for the second set of mea surement points
    wave_num : ndarray of shape (num_freqs,)
        wavenumber, defined as w / c where w is the angular frequency
        and c is the speed of sound.

    Returns
    -------
    psi : ndarray of shape (num_freqs, num_pos1, num_pos2)
        inner product of the translated directivity functions
    """
    assert pos1.ndim == 2 and pos2.ndim == 2
    assert pos1.shape[-1] == 3 and pos2.shape[-1] == 3
    num_pos1 = pos1.shape[0]
    num_pos2 = pos2.shape[0]

    assert wave_num.ndim == 1
    num_freqs = wave_num.shape[0]
    
    if dir_coeffs1.ndim == 2:
        dir_coeffs1 = dir_coeffs1[:,None,:]
    if dir_coeffs2.ndim == 2:
        dir_coeffs2 = dir_coeffs2[:,None,:]
    assert dir_coeffs1.ndim == 3 and dir_coeffs2.ndim == 3
    assert dir_coeffs1.shape[1] == num_pos1 or dir_coeffs1.shape[1] == 1
    assert dir_coeffs2.shape[1] == num_pos2 or dir_coeffs2.shape[1] == 1
    assert dir_coeffs1.shape[0] == num_freqs or dir_coeffs1.shape[0] == 1
    assert dir_coeffs2.shape[0] == num_freqs or dir_coeffs2.shape[0] == 1

    max_order1 = shd_max_order(dir_coeffs1.shape[-1])

    pos_diff = pos1[:,None,:] - pos2[None,:,:]

    translated_coeffs2 = np.stack([translate_shd_coeffs(dir_coeffs2, pos_diff[m,:,:], wave_num, max_order1) for m in range(num_pos1)], axis=1)
    inner_product_matrix = np.sum(translated_coeffs2 * dir_coeffs1.conj()[:,:,None,:], axis=-1)
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

    psi = translated_inner_product(pos, pos, dir_coeffs, dir_coeffs, wave_num)
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
    num_coeffs = shd_num_coeffs(max_order)
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
    num_coeffs = shd_num_coeffs(max_order)

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
    num_coeffs = shd_num_coeffs(max_order)
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
