import numpy as np
import pathlib
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
import itertools as it
from sympy.physics.wigner import wigner_3j, gaunt

import scipy.special as special

import aspcore.montecarlo as mc
import aspcore.fouriertransform as ft

import aspcol.sphericalharmonics as sph
import aspcol.utilities as utils

import plot_methods as plm
import aspcol.kernelinterpolation as ki

import aspcol.planewaves as pw
import aspcol.soundfieldestimation as sfe

import wigner
import wigners

from aspsim.simulator import SimulatorSetup

import matplotlib.pyplot as plt






# ========= GAUNT COEFFICIENTS =========
def _gaunt_coef_ueno(l1, m1, l2, m2, l3):
    """Gaunt coefficients

    Taken directly from https://github.com/sh01k/MeshRIR/blob/main/example/sf_func.py
    """
    m3 = -m1 - m2
    l = int((l1 + l2 + l3) / 2)

    t1 = l2 - m1 - l3
    t2 = l1 + m2 - l3
    t3 = l1 + l2 - l3
    t4 = l1 - m1
    t5 = l2 + m2

    tmin = max([0, max([t1, t2])])
    tmax = min([t3, min([t4, t5])])

    t = np.arange(tmin, tmax+1)
    gl_tbl = np.array(special.gammaln(np.arange(1, l1+l2+l3+3)))
    G = np.sum( (-1.)**t * np.exp( -np.sum( gl_tbl[np.array([t, t-t1, t-t2, t3-t, t4-t, t5-t])] )  \
                                  +np.sum( gl_tbl[np.array([l1+l2-l3, l1-l2+l3, -l1+l2+l3, l])] ) \
                                  -np.sum( gl_tbl[np.array([l1+l2+l3+1, l-l1, l-l2, l-l3])] ) \
                                  +np.sum( gl_tbl[np.array([l1+m1, l1-m1, l2+m2, l2-m2, l3+m3, l3-m3])] ) * 0.5 ) ) \
        * (-1.)**( l + l1 - l2 - m3) * np.sqrt( (2*l1+1) * (2*l2+1) * (2*l3+1) / (4*np.pi) )
    return G


def gaunt_sympy(l1, l2, l3, m1, m2, m3):
    f1 = np.sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*np.pi))
    f2 = wigner_3j(l1,l2,l3,m1,m2,m3)
    f3 = wigner_3j(l1,l2,l3,0,0,0) 
    return f1*f2*f3


def test_gaunt_coef_dev():
    max_order = 3

    all_params = np.array([i for i in sph._all_gaunt_parameters(max_order)])
    #all_params = all_params[7:8]

    val = []
    val_ref = []
    for i in range(all_params.shape[0]):
        args = all_params[i]
        val.append(sph.gaunt_coefficient(*args))
        val_ref.append(_gaunt_coef_ueno(*args))

    val_ref = np.array(val_ref)
    val = np.array(val)
    assert np.allclose(val, val_ref)


def test_gaunt_coef():
    max_order = 3
    all_params = np.array([i for i in sph._all_gaunt_parameters(max_order)])
    val = sph.gaunt_coefficient(all_params[:, 0], all_params[:, 1], all_params[:, 2], all_params[:, 3], all_params[:, 4])

    val_ref = []
    for i in range(all_params.shape[0]):
        args = all_params[i]
        val_ref.append(_gaunt_coef_ueno(*args))
    val_ref = np.array(val_ref)
    assert np.allclose(val, val_ref)


def test_gaunt_coef_lookup():
    max_order = 6
    all_params = np.array([i for i in sph._all_gaunt_parameters(max_order)])

    tbl = sph.gaunt_coef_lookup(max_order)
    val = tbl[all_params[:, 0], all_params[:, 1], all_params[:, 2], all_params[:, 3], all_params[:, 4]]

    val_ref = []
    for i in range(all_params.shape[0]):
        args = all_params[i]
        val_ref.append(_gaunt_coef_ueno(*args))
    val_ref = np.array(val_ref)
    assert np.allclose(val, val_ref)


# ============ GAUNT AND WIGNER TESTS ============

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

def test_relationship_between_gaunt_and_triple_harmonic_integral():
    max_order = 5
    all_params = np.array([i for i in sph._all_gaunt_parameters(max_order)])
    num_param_sets = all_params.shape[0]
    assert num_param_sets > 0 

    for i in range(num_param_sets):
        (l1, m1, l2, m2, l3) = all_params[i]
        val1 = sph.triple_harmonic_integral(l1, l2, l3, m1, m2, -m1-m2)
        val2 = (-1)**(m1+m2) * sph.gaunt_coefficient(l1, m1, l2, m2, l3)
        assert np.allclose(val1, val2)


def test_triple_harmonic_integral_against_sympy():
    max_order = 3
    all_params = np.array([i for i in sph._all_gaunt_parameters(max_order)])
    num_param_sets = all_params.shape[0]
    assert num_param_sets > 0 

    for i in range(num_param_sets):
        (l1, m1, l2, m2, l3) = all_params[i]
        val1 = sph.triple_harmonic_integral(l1, l2, l3, m1, m2, -m1-m2)
        val2 = float(gaunt(l1, l2, l3, m1, m2, -m1-m2).n(16))
        assert np.allclose(val1, val2)

def test_all_wigner_3j_implementations_are_equal():
    max_order = 8
    all_params = np.array([i for i in sph._all_gaunt_parameters(max_order)])
    num_param_sets = all_params.shape[0]
    assert num_param_sets > 0 

    for i in range(num_param_sets):
        (l1, m1, l2, m2, l3) = all_params[i]

        m2_min, m2_max, wigner_val = wigner.wigner_3jm(l1,l2,l3,m1)
        val1 = wigner_val[m2 - int(m2_min)]

        val2 = wigners.wigner_3j(l1,l2,l3,m1,m2,-m1-m2)
        val3 = float(wigner_3j(l1,l2,l3,m1,m2,-m1-m2).n(16))
        assert np.allclose(val1, val2)
        assert np.allclose(val1, val3)



# =============== SPHERICAL HARMONIC TESTS ========================
def test_linearisation_formula_for_spherical_harmonics():
    """
    Equation (3.70) in Multiple scattering: Interaction of time-harmonic waves with N obstacles
    Relates the Gaunt coefficient to the product of two spherical harmonics. A simpler expression
    for the same thing is given between (3.74) and (3.75) in the same book.
    """
    rng = np.random.default_rng()
    direction = rng.uniform(-1, 1, size = (1,3))
    radius, angles = utils.cart2spherical(direction)

    max_order = 5
    for l1, l2 in it.product(range(max_order+1), range(max_order+1)):
        for m1, m2, in it.product(range(-l1, l1+1), range(-l2, l2+1)):
            q0 = _linearity_formula_lower_limit(l1, l2, m1, m2)
            Q = (l1 + l2 - q0) / 2
            assert utils.is_integer(Q)
            Q = int(Q)
            
            sum_val = 0
            for l3 in range(Q+1):
                g = sph.gaunt_coefficient(l1, m1, l2, m2, q0 + 2*l3)
                sum_val += g * special.sph_harm(m1+m2, q0 + 2*l3, angles[0,0], angles[0,1])

            comparison_val = special.sph_harm(m1, l1, angles[0,0], angles[0,1]) * special.sph_harm(m2, l2, angles[0,0], angles[0,1])
            assert np.allclose(sum_val, comparison_val)

def test_basis_function_equals_integral_of_spherical_harmonics_and_plane_waves():
    """This is the relationship that is used to define the directivity coefficients, so it
    must hold for the directional microphone model to make sense. 

    This is the Funk-Hecke formula, multiplied by sqrt(4 pi) on both sides. 
    Eq. 3.43 in Multiple scattering: interaction of time-harmonic waves with N obstacles. 

    basis_nu^mu(r) = frac{i^nu}{sqrt{4 pi}} int_S^2 Y_nu^mu(d_hat) exp(-ikr^T d_hat)    ds(d_hat)
    """
    rng = np.random.default_rng()
    max_order = 7
    pos = rng.uniform(low = -1, high=1, size=(1,3))
    freq = rng.uniform(low=100, high=1000)
    wave_num = freq / 343

    orders, degrees = sph.shd_num_degrees_vector(max_order)
    basis_val = np.squeeze(sph.shd_basis(pos, orders, degrees, wave_num))

    def calc_integral(num_samples):
        sphere_samples = rng.normal(loc = 0, scale = 1, size=(num_samples, 3))
        sphere_samples = sphere_samples / np.linalg.norm(sphere_samples, axis=-1)[:,None]
        rad, angles = utils.cart2spherical(sphere_samples)

        const_factor = 1j**orders / np.sqrt(4 * np.pi)
        sph_harm = special.sph_harm(degrees[:,None], orders[:,None], angles[...,0], angles[...,1])
        plane_wave = np.exp(-1j * wave_num * np.sum(pos * sphere_samples, axis=-1))
        
        sphere_area = 4 * np.pi
        int_value = const_factor * sphere_area * np.mean(sph_harm * plane_wave[None,:], axis=-1) # multiply by area?
        return int_value
    
    num_samples_all = 10 ** np.arange(2, 7)
    int_values = [calc_integral(ns) for ns in num_samples_all]
    mse = [np.mean(np.abs(basis_val - int_val)**2) for int_val in int_values]

    # UNCOMMENT TO SEE ERROR EVOLUTION
    # fig, ax = plt.subplots(1,1, figsize=(8,6))
    # ax.plot(np.log10(num_samples_all), 10 * np.log10(mse))
    # ax.set_title("Error between basis function and integral")
    # ax.set_xlabel("Number of samples [log10]")
    # ax.set_ylabel("Mean square error [dB]")
    # plt.show()

    assert all([mse[i] >= mse[i+1] for i in range(len(mse)-1)]) # integral should be more exact with more samples
    assert mse[-1] < 1e-5 #integral should be very close to basis function

def test_plane_wave_with_positive_exponential_expressed_in_spherical_harmonics():
    """This is the Rayleigh expansion formula
    Eq. 3.69 in Multiple scattering: interaction of time-harmonic waves with N obstacles.
    It expressed a single plane wave in terms of spherical harmonics.

    exp(ik r^T d) = sum_{nu, mu} varphi_{nu}^{mu}(r) sqrt(4 pi) i^{nu} conj{Y_{nu}^{mu}(d)}
    
    It can be viewed as a regular spherical decomposition, with coefficients defined as
    alpha_{nu}^{mu} = sqrt(4 pi) i^{nu} conj{Y_{nu}^{mu}(d)}
    and with an expansion centered around the origin.
    """
    rng = np.random.default_rng()
    max_order = 40
    num_pos = 10000
    # pos = np.concatenate((rng.uniform(low = -1, high=1, size=(num_pos,2)), np.zeros((num_pos, 1))), axis=-1)
    pos = rng.uniform(low = -1, high=1, size=(num_pos,3))
    freq = rng.uniform(low=100, high=1000)
    wave_num = 2 * np.pi * freq / 343
    
    planewave_direction = mc.uniform_random_on_sphere(1, rng)
    rad, angles = utils.cart2spherical(planewave_direction)
    orders, degrees = sph.shd_num_degrees_vector(max_order)

    # Calculate harmonic coefficients
    const_factor = 1j**orders * np.sqrt(4 * np.pi)
    sph_harm = special.sph_harm(degrees, orders, angles[...,0], angles[...,1])
    shd_coeffs = const_factor * np.conj(sph_harm)
    shd_plane_wave = np.squeeze(sph.reconstruct_pressure(shd_coeffs, pos, np.zeros((1, 3)), wave_num))

    plane_wave = np.squeeze(pw.plane_wave(pos, planewave_direction, wave_num))
    plane_wave = np.conj(plane_wave) # conjugate because we need exp(ikr^T d)
    
    # UNCOMMENT TO SEE PLANE WAVE AND THE APPROXIMATE PLANE WAVE FROM SPHERICAL HARMONICS
    #plm.image_scatter_freq_response(plane_wave, np.array([freq]), pos, dot_size=10)
    #plm.image_scatter_freq_response(sph_plane_wave, np.array([freq]), pos, dot_size=10)
    #plt.show()
    mse = np.mean(np.abs(plane_wave - shd_plane_wave)**2) / np.mean(np.abs(plane_wave)**2)
    assert mse < 1e-10

def test_plane_wave_with_negative_exponential_expressed_in_spherical_harmonics():
    """This is the Rayleigh expansion formula, but for plane waves of the form exp(-ikr^T d)
    It is a slight reformulation of Eq. 3.69 in Multiple scattering: interaction of time-harmonic waves with N obstacles.
    Using identity for conjugation of spherical harmonics to retain the form of the expression as a
    sperical harmonic decomposition. 

    exp(-ik r^T d) = sum_{nu, mu}  sqrt(4 pi) (-i)^{nu} (-1)^{mu} Y_{nu}^{-mu}(d) varphi_{nu}^{mu}(r)
    
    It can be viewed as a regular spherical decomposition, with coefficients defined as
    alpha_{nu}^{mu} = sqrt(4 pi) (-1)^{mu} (-i)^{nu} Y_{nu}^{-mu}(d)
    and with an expansion centered around the origin.
    """
    rng = np.random.default_rng()
    max_order = 40
    num_pos = 10000
    #pos = np.concatenate((rng.uniform(low = -1, high=1, size=(num_pos,2)), np.zeros((num_pos, 1))), axis=-1)
    pos = rng.uniform(low = -1, high=1, size=(num_pos,3))
    freq = rng.uniform(low=100, high=1000)
    wave_num = 2 * np.pi * freq / 343
    
    planewave_direction = mc.uniform_random_on_sphere(1, rng)
    rad, angles = utils.cart2spherical(planewave_direction)
    orders, degrees = sph.shd_num_degrees_vector(max_order)
    neg_degrees = -degrees

    # Calculate harmonic coefficients
    const_factor = (-1.0)**(degrees) * (-1j)**orders * np.sqrt(4 * np.pi)
    sph_harm = special.sph_harm(neg_degrees, orders, angles[...,0], angles[...,1])
    shd_coeffs = const_factor * sph_harm
    shd_plane_wave = np.squeeze(sph.reconstruct_pressure(shd_coeffs, pos, np.zeros((1, 3)), wave_num))

    plane_wave = np.squeeze(pw.plane_wave(pos, planewave_direction, wave_num))
    
    # UNCOMMENT TO SEE PLANE WAVE AND THE APPROXIMATE PLANE WAVE FROM SPHERICAL HARMONICS
    # plm.image_scatter_freq_response(plane_wave, np.array([freq]), pos, dot_size=10)
    # plm.image_scatter_freq_response(shd_plane_wave, np.array([freq]), pos, dot_size=10)
    # plt.show()
    mse = np.mean(np.abs(plane_wave - shd_plane_wave)**2) / np.mean(np.abs(plane_wave)**2)
    assert mse < 1e-10


# ============ TRANSLATION OPERATOR ============
def show_translation_operator_addition_property():
    """
    The translation operator should satisfy 
    T(r + r') = T(r) T(r')
    """
    rng = np.random.default_rng(123456)
    pos1 = rng.normal(size = (1,3))
    pos2 = rng.normal(size = (1,3))
    pos_added = pos1 + pos2
    wave_num = np.array([2 * np.pi * 1000 / 343])
    max_order_input = 2
    max_order_mid = 20
    max_order_output = 2

    T1 = sph.translation_operator(pos1, wave_num, max_order_input, max_order_mid)
    T2 = sph.translation_operator(pos2, wave_num, max_order_mid, max_order_output)
    Tadded = sph.translation_operator(pos_added, wave_num, max_order_input, max_order_output)
    T = T2 @ T1

    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    clr = axes[0].matshow(np.squeeze(np.abs(T)))
    plt.colorbar(clr, ax = axes[0])
    clr = axes[1].matshow(np.squeeze(np.real(T)))
    plt.colorbar(clr, ax = axes[1])
    clr = axes[2].matshow(np.squeeze(np.imag(T)))
    plt.colorbar(clr, ax = axes[2])

    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    clr = axes[0].matshow(np.squeeze(np.abs(Tadded)))
    plt.colorbar(clr, ax = axes[0])
    clr = axes[1].matshow(np.squeeze(np.real(Tadded)))
    plt.colorbar(clr, ax = axes[1])
    clr = axes[2].matshow(np.squeeze(np.imag(Tadded)))
    plt.colorbar(clr, ax = axes[2])
    plt.show()



# ================ SHD BASIS TRANSLATION ================

def test_basis_translation_addition_property_as_function_of_max_order_mid():
    """The translation operator should satisfy T(r + r') = T(r) T(r')
    This test shows that as we increase the truncation order of the
    internal dimension of T(r) T(r'), i.e. the one dimension that disappears
    after multiplication, the MSE should go towards zero. 
    """
    rng = np.random.default_rng(123456)
    pos1 = rng.normal(size = (1,3))
    pos2 = rng.normal(size = (1,3))
    pos_added = pos1 + pos2
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order_input = 2
    max_order_output = 2
    Tadded = sph.basis_translation_3_92(pos_added, wave_num, max_order_input, max_order_output)
    norm_factor = np.mean(np.abs(Tadded)**2)

    max_order_mid = [i for i in range(2, 21, 2)]
    mse = []
    for mom in max_order_mid:
        T1 = sph.basis_translation_3_92(pos1, wave_num, max_order_input, mom)
        T2 = sph.basis_translation_3_92(pos2, wave_num, mom, max_order_output)
        T = T2 @ T1

        mse.append(10 * np.log10(np.mean(np.abs(T - Tadded)**2) / norm_factor))
    # Uncomment to see how the mse evolve with increasing max_order_mid
    # plt.plot(max_order_mid, mse)
    # plt.xlabel("max_order_mid")
    # plt.ylabel("MSE (dB)")
    # plt.show()
    assert all([mse[i] >= mse[i+1] for i in range(len(mse)-1)])


def test_basis_translation_with_negative_argument_is_hermitian_basis_translation():
    """The translation operator should satisfy T(-r) = T(r)^H
    """
    rng = np.random.default_rng()
    pos = rng.normal(size = (1,3))
    wave_num = np.array([2 * np.pi * 1000 / 343])
    max_order_input = 5
    max_order_output = 5

    T = np.squeeze(sph.basis_translation_3_92(pos, wave_num, max_order_input, max_order_output), axis=(0,1))
    T2 = np.squeeze(sph.basis_translation_3_92(-pos, wave_num, max_order_input, max_order_output), axis=(0,1))

    # plt.matshow(np.abs(T))
    # plt.colorbar()
    # plt.matshow(np.abs(T2))
    # plt.colorbar()
    # plt.show()

    assert np.allclose(T, T2.conj().T)

def test_basis_translation_with_zero_arguments_is_identity():
    """Identity given as (i) on page 88 in 
    P. Martin's Multiple scattering: Interaction of time-harmonic waves with N obstacles
    
    The translation operator should satisfy hat{S}(0) = I
    """
    pos = np.zeros((1,3))
    wave_num = np.array([2 * np.pi * 1000 / 343])
    max_order_input = 5
    max_order_output = 5
    T = np.squeeze(sph.basis_translation_3_92(pos, wave_num, max_order_input, max_order_output))
    assert np.allclose(T, np.eye(T.shape[0]))

def test_basis_translation_with_input_order_zero_gives_basis_function():
    """Identity given as (ii) on page 88 in 
    P. Martin's Multiple scattering: Interaction of time-harmonic waves with N obstacles
    """
    rng = np.random.default_rng()
    pos = rng.normal(size=(1,3))
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order_output = 5
    T = np.squeeze(sph.basis_translation_3_92(pos, wave_num, 0, max_order_output))

    orders, degrees = sph.shd_num_degrees_vector(max_order_output)
    basis_vals = np.squeeze(sph.shd_basis(pos, orders, degrees, wave_num))
    assert np.allclose(basis_vals, T)

def test_basis_translation_with_output_order_zero_gives_conjugated_basis_function():
    """Identity given as (iii) on page 88 in 
    P. Martin's Multiple scattering: Interaction of time-harmonic waves with N obstacles
    """
    rng = np.random.default_rng()
    pos = rng.normal(size=(1,3))
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order_input = 5
    T = np.squeeze(sph.basis_translation_3_92(pos, wave_num, max_order_input, 0))

    orders, degrees = sph.shd_num_degrees_vector(max_order_input)
    basis_vals = np.conj(np.squeeze(sph.shd_basis(pos, orders, degrees, wave_num)))
    for i, nu in enumerate(orders):
        basis_vals[i] = (-1)**nu * basis_vals[i]
    assert np.allclose(basis_vals, T)


def test_basis_translation_Martin_identity_3_93():
    """Identity given as (3.93) on page 92 in P. Martin,
    hat{S}_{n nu}^{m mu} = (-1)^{n+nu} hat{S}_{n nu}^{m mu}.conj()
    """
    rng = np.random.default_rng()
    pos = rng.normal(size=(1,3))
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order = 5
    T = np.squeeze(sph.basis_translation_3_92(pos, wave_num, max_order, max_order), axis=(0,1))
    orders, degrees = sph.shd_num_degrees_vector(max_order)
    for i, (n, m) in enumerate(zip(orders, degrees)):
        for j, (nu, mu) in enumerate(zip(orders, degrees)):
            assert np.allclose(T[j,i], (-1.0)**(n + nu) * np.conj(T[i,j]))

def test_basis_translation_Martin_identity_3_94():
    """Identity given as (3.94) on page 92 in P. Martin,
    hat{S}_{n nu}^{-m -mu} = (-1)^{m+mu} hat{S}_{n nu}^{m mu}.conj()
    """
    rng = np.random.default_rng()
    pos = rng.normal(size=(1,3))
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order = 5
    T = np.squeeze(sph.basis_translation_3_92(pos, wave_num, max_order, max_order), axis=(0,1))
    orders, degrees = sph.shd_num_degrees_vector(max_order)
    for i, (n, m) in enumerate(zip(orders, degrees)):
        for j, (nu, mu) in enumerate(zip(orders, degrees)):
            i2 = _find_index_of_order_degree(n, -m)
            j2 = _find_index_of_order_degree(nu, -mu)
            assert np.allclose(T[i2,j2], (-1.0)**(m + mu) * np.conj(T[i,j]))

def _find_index_of_order_degree(order, degree):
    orders, degrees = sph.shd_num_degrees_vector(order)
    idx = np.where((orders == order) & (degrees == degree))[0]
    return idx[0]


def test_basis_translation_3_80_is_equivalent_to_basis_translation_3_92():
    rng = np.random.default_rng()
    pos = rng.normal(size=(4,3))
    freq = rng.uniform(100, 1000, size=(2,))
    wave_num = 2 * np.pi * freq / 343
    max_order_input = rng.integers(1, 10)
    max_order_output = rng.integers(1, 10)
    T1 = sph.basis_translation_3_80(pos, wave_num, max_order_input, max_order_output)
    T2 = sph.basis_translation_3_92(pos, wave_num, max_order_input, max_order_output)
    assert np.allclose(T1, T2)




# ============== TRANSLATION OF SHD COEFFICIENTS ==============
def test_translation_operator_with_negative_argument_is_hermitian():
    """The translation operator should satisfy T(-r) = T(r)^H
    """
    rng = np.random.default_rng()
    pos = rng.normal(size = (1,3))
    wave_num = np.array([2 * np.pi * 1000 / 343])
    max_order_input = 5
    max_order_output = 5

    T = np.squeeze(sph.translation_operator(pos, wave_num, max_order_input, max_order_output), axis=(0,1))
    T2 = np.squeeze(sph.translation_operator(-pos, wave_num, max_order_input, max_order_output), axis=(0,1))
    assert np.allclose(T, T2.conj().T)

def test_translation_operator_addition_property_as_function_of_max_order_mid():
    """The translation operator should satisfy T(r + r') = T(r) T(r')
    This test shows that as we increase the truncation order of the
    internal dimension of T(r) T(r'), i.e. the one dimension that disappears
    after multiplication, the MSE should go towards zero. 
    """
    rng = np.random.default_rng(123456)
    pos1 = rng.normal(size = (1,3))
    pos2 = rng.normal(size = (1,3))
    pos_added = pos1 + pos2
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order_input = 3
    max_order_output = 3
    Tadded = sph.translation_operator(pos_added, wave_num, max_order_input, max_order_output)
    norm_factor = np.mean(np.abs(Tadded)**2)

    max_order_mid = [i for i in range(2, 21, 2)]
    mse = []
    for mom in max_order_mid:
        T1 = sph.translation_operator(pos1, wave_num, max_order_input, mom)
        T2 = sph.translation_operator(pos2, wave_num, mom, max_order_output)
        T = T2 @ T1

        mse.append(10 * np.log10(np.mean(np.abs(T - Tadded)**2) / norm_factor))
    # Uncomment to see how the mse evolve with increasing max_order_mid
    # plt.plot(max_order_mid, mse)
    # plt.xlabel("max_order_mid")
    # plt.ylabel("MSE (dB)")
    # plt.show()
    assert all([mse[i] >= mse[i+1] for i in range(len(mse)-1)])

def test_translation_operator_addition_property():
    """The translation operator should satisfy T(r + r') = T(r) T(r') 
    """
    rng = np.random.default_rng(123456)
    pos1 = rng.normal(size = (1,3))
    pos2 = rng.normal(size = (1,3))
    pos_added = pos1 + pos2
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order_input = 6
    max_order_output = 6
    Tadded = sph.translation_operator(pos_added, wave_num, max_order_input, max_order_output)

    max_order_mid = 18
    T1 = sph.translation_operator(pos1, wave_num, max_order_input, max_order_mid)
    T2 = sph.translation_operator(pos2, wave_num, max_order_mid, max_order_output)
    T = T2 @ T1

    mse = 10 * np.log10(np.mean(np.abs(T - Tadded)**2) / np.mean(np.abs(Tadded)**2))
    assert mse < -100


# def test_translation_operator_against_uenos_implementation():
#     """
#     It looks like uenos implemenation differs. Not sure how
#     to determine whether that's a mistake or not. 
#     """
#     rng = np.random.default_rng(21345654)
#     pos = rng.normal(size = (1,3))
#     freq = rng.uniform(100, 1000)
#     wave_num = np.array([2 * np.pi * freq / 343])
#     max_order_input = 5
#     max_order_output = 5

#     T_ueno = sph._translation_ueno(pos, wave_num, max_order_input, max_order_output)[0,0,:,:]
#     T = sph.translation_operator(pos, wave_num, max_order_input, max_order_output)[0,0,:,:]
#     # UNCOMMENT TO VIEW DIFFERENCES
#     # fig, axes = plt.subplots(1, 3, figsize=(14,4))
#     # axes[0].set_title("our")
#     # clr = axes[0].matshow(np.squeeze(np.abs(T)))
#     # plt.colorbar(clr, ax = axes[0])
#     # clr = axes[1].matshow(np.squeeze(np.real(T)))
#     # plt.colorbar(clr, ax = axes[1])
#     # clr = axes[2].matshow(np.squeeze(np.imag(T)))
#     # plt.colorbar(clr, ax = axes[2])

#     # fig, axes = plt.subplots(1, 3, figsize=(14,4))
#     # axes[0].set_title("Ueno")
#     # clr = axes[0].matshow(np.squeeze(np.abs(T_ueno)))
#     # plt.colorbar(clr, ax = axes[0])
#     # clr = axes[1].matshow(np.squeeze(np.real(T_ueno)))
#     # plt.colorbar(clr, ax = axes[1])
#     # clr = axes[2].matshow(np.squeeze(np.imag(T_ueno)))
#     # plt.colorbar(clr, ax = axes[2])

#     # fig, axes = plt.subplots(1, 3, figsize=(14,4))
#     # axes[0].set_title("Difference")
#     # clr = axes[0].matshow(np.squeeze(np.abs(T_ueno-T)))
#     # plt.colorbar(clr, ax = axes[0])
#     # clr = axes[1].matshow(np.squeeze(np.real(T_ueno-T)))
#     # plt.colorbar(clr, ax = axes[1])
#     # clr = axes[2].matshow(np.squeeze(np.imag(T_ueno-T)))
#     # plt.colorbar(clr, ax = axes[2])
#     # plt.show()
#     assert np.allclose(T, T_ueno)


def test_translation_operator_against_jax_compiled_implementation():
    rng = np.random.default_rng(21345654)
    pos = rng.normal(size = (3,3))
    freq = rng.uniform(100, 1000, size=(2,))
    wave_num = 2 * np.pi * freq / 343
    max_order_input = 2
    max_order_output = 2

    gaunt = sfe.calculate_gaunt_set(max_order_input, max_order_output)
    T_jax = sfe.translation_operator(pos, wave_num, max_order_input, max_order_output, gaunt)
    T = sph.translation_operator(pos, wave_num, max_order_input, max_order_output)

    T_jax = T_jax[0,0,:,:]
    T = T[0,0,:,:]
    # UNCOMMENT TO VIEW DIFFERENCES
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].set_title("Numpy")
    clr = axes[0].matshow(np.squeeze(np.abs(T)))
    plt.colorbar(clr, ax = axes[0])
    clr = axes[1].matshow(np.squeeze(np.real(T)))
    plt.colorbar(clr, ax = axes[1])
    clr = axes[2].matshow(np.squeeze(np.imag(T)))
    plt.colorbar(clr, ax = axes[2])

    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].set_title("Jax")
    clr = axes[0].matshow(np.squeeze(np.abs(T_jax)))
    plt.colorbar(clr, ax = axes[0])
    clr = axes[1].matshow(np.squeeze(np.real(T_jax)))
    plt.colorbar(clr, ax = axes[1])
    clr = axes[2].matshow(np.squeeze(np.imag(T_jax)))
    plt.colorbar(clr, ax = axes[2])

    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].set_title("Difference")
    clr = axes[0].matshow(np.squeeze(np.abs(T_jax-T)))
    plt.colorbar(clr, ax = axes[0])
    clr = axes[1].matshow(np.squeeze(np.real(T_jax-T)))
    plt.colorbar(clr, ax = axes[1])
    clr = axes[2].matshow(np.squeeze(np.imag(T_jax-T)))
    plt.colorbar(clr, ax = axes[2])
    plt.show()
    assert np.allclose(T, T_jax)


















def test_translation_of_shd_coeffs_does_not_affect_reconstructed_function():
    """The reconstructed function should be unaffected by translation
    """
    rng = np.random.default_rng()
    side_len = 0.3
    
    num_side = 20
    pos_eval = np.meshgrid(np.linspace(-side_len/2, side_len/2, num_side), np.linspace(-side_len/2, side_len/2, num_side), np.zeros(1,))
    pos_eval = np.concatenate(pos_eval, axis=-1)
    pos_eval = np.reshape(pos_eval, (-1,3))

    exp_center = rng.uniform(low=-0.1, high=0.1, size=(1,3))
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343
    max_order = 5
    max_order_output = max_order + 10
    shd_coeffs = _random_shd_coeffs(max_order, 1, rng)

    p = sph.reconstruct_pressure(shd_coeffs, pos_eval, exp_center, wave_num)
    pos_translate = rng.uniform(low = -0.2, high = 0.2, size=(1,3))
    new_exp_center = exp_center + pos_translate

    shd_coeffs_translated = sph.translate_shd_coeffs(shd_coeffs, pos_translate, wave_num, max_order_output)
    p_translated = sph.reconstruct_pressure(shd_coeffs_translated, pos_eval, new_exp_center, wave_num)
    p_translated_nocompensation = sph.reconstruct_pressure(shd_coeffs_translated, pos_eval, exp_center, wave_num)

    diff = p - p_translated

    # UNCOMMENT TO VIEW DIFFERENCES
    # p_all = {"original": p, "translated": p_translated, "difference": diff, "nocompensation" : p_translated_nocompensation, "diff nocompensation" : p - p_translated_nocompensation,}
    # plm.image_scatter_freq_response(p_all, freq, pos_eval)
    # plt.show()
    assert np.mean(np.abs(diff)) < 1e-8


def test_translation_of_shd_coeffs_gives_same_values_as_directly_estimated_shd_coeffs():
    sr = 1000
    reg = 1e-7
    max_order = 3
    center1 = np.zeros((1,3))
    center2 = np.ones((1,3)) * 0.2

    pos1, p1, freqs, sim_info = _generate_soundfield_data_omni(sr, center1)
    pos2, p2, freqs, sim_info = _generate_soundfield_data_omni(sr, center2)
    wave_num = 2 * np.pi * freqs / sim_info.c

    shd_coeffs1 = sph.inf_dimensional_shd_omni(p1, pos1, center1, max_order, wave_num, reg)
    shd_coeffs2 = sph.inf_dimensional_shd_omni(p2, pos2, center2, max_order, wave_num, reg)

    shd_coeffs2_est = sph.translate_shd_coeffs(shd_coeffs1, center2-center1, wave_num, max_order)

    fig, axes = plt.subplots(1, 4, figsize=(20,6))
    axes[0].plot(10*np.log10(np.mean(np.abs(shd_coeffs2 - np.squeeze(shd_coeffs2_est))**2, axis=-1)))
    axes[0].set_ylabel("Mean square error (dB)")
    axes[0].set_xlabel("Frequency (Hz)")

    axes[1].plot(10*np.log10(np.mean(np.abs(shd_coeffs2 - np.squeeze(shd_coeffs2_est))**2, axis=0) / np.mean(np.abs(shd_coeffs2)**2, axis=0)))
    axes[1].set_ylabel("Mean square error (dB)")
    axes[1].set_xlabel("Frequency (Hz)")

    axes[2].plot(np.real(shd_coeffs2_est[:,0]), label="estimated")
    axes[2].plot(np.real(shd_coeffs2[:,0]), label="recorded")
    axes[2].set_title("Real part")
    # axes[2].plot(np.imag(shd_coeffs2_est), label="estimated")
    # axes[2].plot(np.imag(shd_coeffs2), label="recorded") 
    # axes[2].set_title("Imag part")
    axes[3].plot(np.abs(shd_coeffs2_est[:,0]), label="estimated")
    axes[3].plot(np.abs(shd_coeffs2[:,0]), label="recorded")
    axes[3].set_title("Magnitude")
    for ax in axes:
        ax.legend()
    plt.show()
    


def test_translation_of_shd_coeffs_is_numerically_stable_for_small_arguments():
    """The reconstructed function should be unaffected by translation
    """
    rng = np.random.default_rng(1234567)
    side_len = 0.3
    
    num_side = 20
    pos_eval = np.meshgrid(np.linspace(-side_len/2, side_len/2, num_side), np.linspace(-side_len/2, side_len/2, num_side), np.zeros(1,))
    pos_eval = np.concatenate(pos_eval, axis=-1)
    pos_eval = np.reshape(pos_eval, (-1,3))

    exp_center = np.zeros((1,3)) #rng.uniform(low=-0.1, high=0.1, size=(1,3))
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343
    max_order = 1
    max_order_output = max_order + 10
    shd_coeffs = _random_shd_coeffs(max_order, 1, rng)

    p = sph.reconstruct_pressure(shd_coeffs, pos_eval, exp_center, wave_num)
    pos_translate = 1e-9 * np.array([[0,1,0]]) #rng.uniform(low = -0.2, high = 0.2, size=(1,3))
    new_exp_center = exp_center + pos_translate

    shd_coeffs_translated = sph.translate_shd_coeffs(shd_coeffs, pos_translate, wave_num, max_order_output)
    p_translated = sph.reconstruct_pressure(shd_coeffs_translated, pos_eval, new_exp_center, wave_num)
    p_translated_nocompensation = sph.reconstruct_pressure(shd_coeffs_translated, pos_eval, exp_center, wave_num)

    diff = p - p_translated

    # UNCOMMENT TO VIEW DIFFERENCES
    # p_all = {"original": p, "translated": p_translated, "difference": diff, "nocompensation" : p_translated_nocompensation, "diff nocompensation" : p - p_translated_nocompensation,}
    # plm.image_scatter_freq_response(p_all, freq, pos_eval)
    # plt.show()
    assert np.mean(np.abs(diff)) < 1e-8






 #====================== OTHER SHD FUNCTIONS ========================

def test_translated_inner_product_for_zero_order_is_sinc_function():
    rng = np.random.default_rng()
    num_pos = 10
    pos1 = rng.normal(size = (num_pos,3))
    pos2 = rng.normal(size = (num_pos,3))
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343
    val = sph.translated_inner_product(pos1, pos2, sph.directivity_omni(), sph.directivity_omni(), wave_num)
    val2 = ki.kernel_helmholtz_3d(pos1, pos2, wave_num)
    assert np.allclose(val, val2)

def test_measurement_conj_omni_times_measurement_omni_is_kernel_matrix():
    """
    Xi Xi^H = Psi where Psi is the matrix given in Ueno. For omni microphones as
    in this case, Psi equals K the kernel matrix from Ueno's kernel interpolation paper. 
    """
    rng = np.random.default_rng()
    num_mic1 = 3
    num_mic2 = 4
    pos1 = rng.normal(size=(num_mic1, 3))
    pos2 = rng.normal(size=(num_mic2, 3))
    exp_center = rng.normal(size = (1, 3)) * 0
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343

    input_vec = rng.normal(size=(1, num_mic2)) + 1j * rng.normal(size=(1, num_mic2))

    psi = ki.kernel_helmholtz_3d(pos1, pos2, wave_num)
    vec_ip = np.squeeze(psi @ input_vec[...,None], axis=-1)

    mse = []
    max_inner_order = 40
    for i in np.arange(5, max_inner_order, 5):
        shd_meas = sph.apply_measurement_conj_omni(input_vec, pos2, exp_center, i, wave_num)
        vec_meas = sph.apply_measurement_omni(shd_meas, pos1, exp_center, wave_num)
        mse.append(10 * np.log10(np.mean(np.abs(vec_meas - vec_ip)**2)))
    #plt.plot(np.arange(2, 35, 2), mse)
    #plt.show()
    assert all([mse[i] >= mse[i+1] or mse[i] < -200 for i in range(len(mse)-1)])
    assert np.allclose(vec_ip, vec_meas)

def test_measurement_conj_omni_times_measurement_omni_is_translated_inner_product():
    """
    Xi Xi^H = Psi where Psi is the matrix given in Ueno. For omni microphones as
    in this case, Psi equals K the kernel matrix from Ueno's kernel interpolation paper. 
    """
    rng = np.random.default_rng()
    num_mic1 = 3
    num_mic2 = 4
    pos1 = rng.normal(size=(num_mic1, 3))
    pos2 = rng.normal(size=(num_mic2, 3))
    exp_center = rng.normal(size = (1, 3)) * 0
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343

    input_vec = rng.normal(size=(1, num_mic2)) + 1j * rng.normal(size=(1, num_mic2))

    dir_coeffs1 = sph.directivity_omni()
    dir_coeffs2 = sph.directivity_omni()
    psi = sph.translated_inner_product(pos1, pos2, dir_coeffs1, dir_coeffs2, wave_num)
    vec_ip = np.squeeze(psi @ input_vec[...,None], axis=-1)

    mse = []
    max_inner_order = 40
    for i in np.arange(5, max_inner_order, 5):
        shd_meas = sph.apply_measurement_conj_omni(input_vec, pos2, exp_center, i, wave_num)
        vec_meas = sph.apply_measurement_omni(shd_meas, pos1, exp_center, wave_num)
        mse.append(10 * np.log10(np.mean(np.abs(vec_meas - vec_ip)**2)))
    #plt.plot(np.arange(2, 35, 2), mse)
    #plt.show()
    assert all([mse[i] >= mse[i+1] or mse[i] < -100 for i in range(len(mse)-1)])
    assert np.allclose(vec_ip, vec_meas)

def test_measurement_operator_times_conjugate_measurement_operator_equals_translated_inner_product():
    rng = np.random.default_rng()
    num_mic1 = 2
    num_mic2 = 3
    pos1 = rng.normal(size=(num_mic1, 3))
    pos2 = rng.normal(size=(num_mic2, 3))
    exp_center = rng.normal(size = (1, 3)) * 0
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343

    max_order1 = 2
    max_order2 = 2

    input_vec = rng.normal(size=(1, num_mic2))
    dir_coeffs1 = _random_shd_coeffs(max_order1, 1, rng)
    dir_coeffs2 = _random_shd_coeffs(max_order2, 1, rng)

    psi = sph.translated_inner_product(pos1, pos2, dir_coeffs1, dir_coeffs2, wave_num)
    vec_ip = np.squeeze(psi @ input_vec[...,None], axis=-1)

    mse = []
    max_inner_order = 41
    for inner_order in np.arange(10, max_inner_order, 10):
        shd_meas = sph.apply_measurement_conj(input_vec, pos2, exp_center, inner_order, wave_num, dir_coeffs2)
        vec_meas = sph.apply_measurement(shd_meas, pos1, exp_center, wave_num, dir_coeffs1)
        mse.append(10 * np.log10(np.mean(np.abs(vec_meas - vec_ip)**2 / np.mean(np.abs(vec_ip)**2))))

    #plt.plot(np.arange(10, max_inner_order, 10), mse)
    #plt.show()
    assert all([mse[i] >= mse[i+1] or mse[i] < -200 for i in range(len(mse)-1)])
    assert np.allclose(vec_ip, vec_meas)

# ============= SHD COEFFICIENTS AND RECONSTRUCTIONS =================
def test_good_reconstruction_using_shd_estimation():
    sr = 1000
    ds_freq = 20
    exp_center = np.zeros((1,3))
    pos_mic, rirs, freqs, sim_info = _generate_soundfield_data_omni(sr, exp_center)
    wave_num = 2 * np.pi * freqs / sim_info.c
    max_order = 9

    rirs = np.ascontiguousarray(rirs[::ds_freq,...])
    wave_num = np.ascontiguousarray(wave_num[::ds_freq])
    freqs = np.ascontiguousarray(freqs[::ds_freq])

    shd_coeffs = sph.inf_dimensional_shd_omni(rirs, pos_mic, exp_center, max_order, wave_num, 1e-8)

    p_est = sph.reconstruct_pressure(shd_coeffs, pos_mic, exp_center, wave_num)
    mse = 10 * np.log10(np.mean(np.abs(p_est - rirs)**2, axis=-1))

    # plt.plot(freqs, mse)
    # plt.show()
    assert np.mean(mse) < -50


def test_measurement_operator_omni_is_same_as_reconstruct_pressure():
    """Xi omni should be the same as the pressure reconstructed from the SHD coefficients
    """
    rng = np.random.default_rng()

    max_order = 10
    num_pos = 1
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    exp_center = rng.normal(size = (1, 3))

    shd_coeffs = _random_shd_coeffs(max_order, 1, rng)
    pos = rng.normal(size=(num_pos,3))
    
    p = sph.reconstruct_pressure(shd_coeffs, pos, exp_center, wave_num)
    p_xi = sph.apply_measurement_omni(shd_coeffs, pos, exp_center, wave_num)
    assert np.allclose(p, p_xi)


def test_measurement_operator_conj_omni_is_adjoint_to_measurement_operator_omni():
    rng = np.random.default_rng()
    num_mic = 10
    max_order = 5
    pos = rng.normal(size=(num_mic, 3))
    exp_center = rng.normal(size = (1, 3))
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343

    vec = rng.normal(size=(1, num_mic))
    shd_coeffs = _random_shd_coeffs(max_order, 1, rng)

    new_vec = sph.apply_measurement_omni(shd_coeffs, pos, exp_center, wave_num)
    val = np.sum(new_vec * np.conj(vec))

    new_shd_sequence = sph.apply_measurement_conj_omni(vec, pos, exp_center, max_order, wave_num)
    val_from_conj = np.sum(shd_coeffs * np.conj(new_shd_sequence))

    assert np.allclose(val, val_from_conj)

def test_measurement_operator_conj_is_adjoint_to_measurement_operator():
    rng = np.random.default_rng()
    num_mic = 10
    max_order = 5
    pos = rng.normal(size=(num_mic, 3))
    exp_center = rng.normal(size = (1, 3))
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343

    vec = rng.normal(size=(1, num_mic))
    shd_coeffs = _random_shd_coeffs(max_order, 1, rng)
    dir_coeffs = _random_shd_coeffs(max_order, 1, rng)

    new_vec = sph.apply_measurement(shd_coeffs, pos, exp_center, wave_num, dir_coeffs)
    val = np.sum(new_vec * np.conj(vec))

    new_shd_sequence = sph.apply_measurement_conj(vec, pos, exp_center, max_order, wave_num, dir_coeffs)
    val_from_conj = np.sum(shd_coeffs * np.conj(new_shd_sequence))

    assert np.allclose(val, val_from_conj)

def test_measurement_operator_conj_omni_is_hermitian_transpose_of_measurement_operator_omni():
    rng = np.random.default_rng()
    num_mic = 10
    max_order = 5
    pos = rng.normal(size=(num_mic, 3))
    exp_center = rng.normal(size = (1, 3))
    freq = rng.uniform(100, 1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343

    meas = sph.measurement_omni(pos, exp_center, max_order, wave_num)
    conj_meas = sph.measurement_conj_omni(pos, exp_center, max_order, wave_num)
    assert np.allclose(np.moveaxis(meas.conj(), -1, -2), conj_meas)









# ================== HELPER FUNCTIONS ==========================



def _random_shd_coeffs(max_order, num_freqs, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    shd_coeffs = rng.normal(size=(num_freqs, sph.shd_num_coeffs(max_order))) + \
        1j * rng.normal(size=(num_freqs, sph.shd_num_coeffs(max_order)))
    return shd_coeffs




def _generate_soundfield_data_omni(sr, exp_center = np.array([[0,0,0]])):
    rng = np.random.default_rng(10)
    side_len = 0.2
    num_mic = 100

    #pos_mic = np.zeros((num_mic, 3))
    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))
    pos_mic += exp_center

    pos_src = np.array([[3,0.05,-0.05]])

    setup = SimulatorSetup()
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 =  0.25
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 100
    setup.sim_info.plot_output = "none"

    setup.add_mics("omni", pos_mic)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = setup.sim_info.max_room_ir_length 
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], freqs, sim.sim_info




def _generate_soundfield_data_dir_center(sr, directivity_dir, rt60 = 0.25):
    rng = np.random.default_rng(10)
    side_len = 0.1
    num_mic = 30

    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))

    pos_src = np.array([[0,1.5,-0.05]])

    setup = SimulatorSetup()
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 = rt60
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "none"

    setup.add_mics("omni", pos_mic)
    setup.add_mics("dir", np.zeros((1,3)), directivity_type=["cardioid"], directivity_dir=directivity_dir)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 2048
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], fpaths["src"]["dir"][...,0], freqs, sim.sim_info



def _generate_soundfield_data_dir_surrounded_by_sources(sr, directivity_dir, num_src, rt60 = 0.25):
    rng = np.random.default_rng(10)
    side_len = 0.4
    num_mic = 30

    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))

    src_angles = np.linspace(0, 2*np.pi, num_src, endpoint=False)
    pos_src = utils.spherical2cart(np.ones((num_src,)), np.stack((src_angles, np.pi/2*np.ones((num_src,))), axis=-1))

    setup = SimulatorSetup()
    setup.sim_info.samplerate = sr
    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 = rt60
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "none"

    setup.add_mics("omni", pos_mic)
    setup.add_mics("dir", np.zeros((1,3)), directivity_type=["cardioid"], directivity_dir=directivity_dir)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 512
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)

    return sim.arrays["omni"].pos, sim.arrays["src"].pos, sim.arrays["dir"].pos, fpaths["src"]["omni"], fpaths["src"]["dir"], freqs, sim.sim_info


def _generate_soundfield_data_dir_and_omni(sr, directivity_dir, rt60 = 0.25):
    rng = np.random.default_rng(10)
    side_len = 1
    num_mic = 50

    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))
    pos_mic_dir = np.copy(pos_mic)

    pos_src = np.array([[0,1.5,-0.05]])

    setup = SimulatorSetup()
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 = rt60
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "none"

    setup.add_mics("omni", pos_mic)
    dir_type = num_mic*["cardioid"]
    dir_dir = np.tile(directivity_dir, (num_mic,1))
    setup.add_mics("dir", pos_mic_dir, directivity_type=dir_type, directivity_dir=dir_dir)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 512
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], fpaths["src"]["dir"][...,0], freqs, sim.sim_info


def __temp_gen_data():
    rng = np.random.default_rng(10)
    side_len = 1
    num_mic = 40
    num_eval = 800
    sr = 1000

    #pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))
    pos_mic = np.concatenate((rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 2)), np.zeros((num_mic,1))), axis=-1)
    pos_eval = np.concatenate((rng.uniform(low=-side_len/2, high=side_len/2, size=(num_eval, 2)), np.zeros((num_eval,1))), axis=-1)
    #pos_eval = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))

    pos_src = np.array([[3,0.05,-0.05]])

    setup = SimulatorSetup()
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 =  0.1
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 100
    setup.sim_info.plot_output = "none"

    setup.add_mics("omni", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = setup.sim_info.max_room_ir_length

    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)

    return sim.arrays["omni"].pos, sim.arrays["eval"].pos, fpaths["src"]["omni"][...,0], fpaths["src"]["eval"][...,0], freqs, sim.sim_info


def test_temp_ki():
    import aspcol.soundfieldestimation as sfe
    import plot_methods as plm
    import aspcol.plot as aspplot

    pos_mic, pos_eval, p_mic, p_eval, freqs, sim_info = __temp_gen_data()
    wave_num = 2 * np.pi *freqs / sim_info.c

    
    est = {}
    est["diffuse KRR"] = sfe.est_ki_diffuse_freq(p_mic, pos_mic, pos_eval, wave_num, 1e-5)
    est["nearest neighbour"] = sfe.nearest_neighbour_freq(p_mic, pos_mic, pos_eval)

    est_all = {"diffuse KRR": est["diffuse KRR"], "nearest neighbour": est["nearest neighbour"], "true": p_eval}
    plm.image_scatter_freq_response(est_all, freqs, pos_eval, dot_size = 25)

    fig, ax = plt.subplots(1,1)
    for est_name, p_est in est.items():
        mse = 10 * np.log10(np.mean(np.abs(p_est - p_eval)**2, axis=-1) / np.mean(np.abs(p_eval)**2, axis=-1))
        ax.plot(freqs, mse, label=est_name)
    ax.legend()
    aspplot.set_basic_plot_look(ax)
    plt.show()


def _get_freq_paths_correct_time_convention(arrays, num_freqs, samplerate):
    """Get the frequency domain response of the paths between all sources and microphones
    
    Parameters
    ----------
    arrays : ArrayCollection
        The array collection to get the frequency paths from
    num_freqs : int
        The number of frequencies to calculate the response for. The number refers to the total FFT
        length, so the number of real frequencies is num_freqs // 2 + 1, which is the number of 
        frequency bins in the output. 
    
    Returns
    -------
    freq_paths : dict of dicts of ndarrays
        freq_paths[src_name][mic_name] is a complex ndarray of shape (num_real_freqs, num_mic, num_src)
        representing the frequency response between a source and a microphone array
    freqs : ndarray of shape (num_real_freqs,)
        The real frequencies in Hz of the response

    Notes
    -----
    The frequencies returned are compatible with get_real_freqs of the package aspcol, as well as the
    output of np.fft.rfftfreq. 

    num_freqs can be safely chosen as a larger number than the number of samples in the impulse response,
    as the FFT will zero-pad the signal to the desired length. But if num_freqs is smaller, then the
    signal will be truncated.
    """
    if num_freqs % 2 == 1:
        raise NotImplementedError("Odd number of frequencies not supported yet")
    
    freqs = ft.get_real_freqs(num_freqs, samplerate)

    fpaths = {}
    for src, mic, path in arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        freq_domain_path = ft.rfft(path, n=num_freqs)
        fpaths[src.name][mic.name] = np.moveaxis(freq_domain_path, 1,2)
    return fpaths, freqs