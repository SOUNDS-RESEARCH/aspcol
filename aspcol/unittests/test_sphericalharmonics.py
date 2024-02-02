import numpy as np
import pathlib
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
import itertools as it
from sympy.physics.wigner import wigner_3j, gaunt

import scipy.special as special

import aspcol.sphericalharmonics as sph
import aspcol.utilities as utils
import aspcol.filterdesign as fd

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



# ============ TRANSLATION TESTS ============
def test_translated_shd_coeffs_are_similar_to_directly_estimated_shd_coeffs():
    sr = 1000
    reg = 1e-7
    max_order = 3
    center1 = np.zeros((1,3))
    center2 = np.ones((1,3)) * 0.5

    pos1, p1, freqs, sim_info = _generate_soundfield_data_omni(sr, center1)
    pos2, p2, freqs, sim_info = _generate_soundfield_data_omni(sr, center2)
    wave_num = 2 * np.pi * freqs / sim_info.c

    shd_coeffs1 = sph.inf_dimensional_shd_omni(p1, pos1, center1, max_order, wave_num, reg)
    shd_coeffs2 = sph.inf_dimensional_shd_omni(p2, pos2, center1, max_order, wave_num, reg)

    shd_coeffs2_est = sph.translate_shd_coeffs(shd_coeffs1, center2-center1, wave_num, max_order, max_order)

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
    



        
def test_translation_operator_against_uenos_implementation():
    """
    It looks like Ueno's implementation might not be correct. 
    It does not satisfy the additivity property well. 
    """
    rng = np.random.default_rng()
    pos = rng.normal(size = (1,3))
    wave_num = np.array([2 * np.pi * 1000 / 343])
    max_order_input = 1
    max_order_output = 1

    T_ref = sph.translation(pos, wave_num, max_order_input, max_order_output)
    T = sph.translation_operator(pos, wave_num, max_order_input, max_order_output)
    pass

def show_translation_operator_addition_property():
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


def test_translation_operator_addition_property_as_function_of_max_order_mid():
    rng = np.random.default_rng(123456)
    pos1 = rng.normal(size = (1,3))
    pos2 = rng.normal(size = (1,3))
    pos_added = pos1 + pos2
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order_input = 2
    max_order_output = 2
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


def test_hermitian_translation_operator_is_negative_argument():
    rng = np.random.default_rng()
    pos = rng.normal(size = (1,3))
    wave_num = np.array([2 * np.pi * 1000 / 343])
    max_order_input = 5
    max_order_output = 5

    T = np.squeeze(sph.translation_operator(pos, wave_num, max_order_input, max_order_output))
    T2 = np.squeeze(sph.translation_operator(-pos, wave_num, max_order_input, max_order_output))

    assert np.allclose(T, T2.conj().T)

def test_translation_operator_with_zero_arguments_is_identity():
    pos = np.zeros((1,3))
    wave_num = np.array([2 * np.pi * 1000 / 343])
    max_order_input = 5
    max_order_output = 5
    T = np.squeeze(sph.translation_operator(pos, wave_num, max_order_input, max_order_output))
    assert np.allclose(T, np.eye(T.shape[0]))

def test_translation_operator_with_one_zero_order_is_basis_function():
    """Identity given as (i) on page 88 in P. Martin's Multiple scattering: Interaction of 
    time-harmonic waves with N obstacles
    """
    rng = np.random.default_rng()
    pos = rng.normal(size=(1,3))
    freq = rng.uniform(100, 1000)
    wave_num = np.array([2 * np.pi * freq / 343])
    max_order_output = 5
    T = np.squeeze(sph.translation_operator(pos, wave_num, 0, max_order_output))

    orders, degrees = sph.shd_num_degrees_vector(max_order_output)
    basis_vals = np.squeeze(sph.shd_basis(pos, orders, degrees, wave_num))
    assert np.allclose(basis_vals, T)



# ============ DIRECTIVITY TESTS ============
def test_directivity_linear_is_cardioid_for_a_equals_one_half():
    dir_coeffs = sph.directivity_linear(0.5, np.array([[0,1,0]]))

    num_angles = 50
    angles = np.zeros((num_angles, 2))
    angles[:, 0] = np.linspace(0, 2*np.pi, num_angles)
    angles[:, 1] = np.pi / 2
    pos = utils.spherical2cart(np.ones((num_angles,)), angles)
    wave_num = np.ones((1,))

    p = sph.reconstruct_pressure(dir_coeffs, pos, np.zeros((1,3)), wave_num)

    fig, ax = plt.subplots(1,1, figsize=(8,6), subplot_kw={'projection': 'polar'})
    ax.plot(angles[:,0], np.abs(p[0,:])**2)
    plt.show()



def test_directivity_coefficients_times_harmonic_coefficients_is_microphone_signal():
    dir_dir = np.array([[0,1,0]])
    pos_omni, p_omni, p_dir, freqs, sim_info= _generate_soundfield_data_dir_center(1000, dir_dir)
    wave_num = 2 * np.pi * freqs / sim_info.c

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni, pos_omni, np.zeros((1,3)), 1, wave_num, 1e-9)

    #sph.apply_measurement_omni(shd_coeffs, np.zeros((1,3)), np.zeros((1,3)), 1, freqs)

    dir_coeffs = sph.directivity_linear(0.5, dir_dir)

    p_est = np.sum(np.conj(dir_coeffs) * shd_coeffs, axis=-1)

    # fig, axes = plt.subplots(1, 4, figsize=(20,6))
    # axes[0].plot(10*np.log10(np.abs(p_est - np.squeeze(p_dir))**2))
    # axes[0].set_title("Mean square error (dB)")
    # axes[1].plot(np.real(p_est), label="estimated")
    # axes[1].plot(np.real(p_dir), label="recorded")
    # axes[1].set_title("Real part")
    # axes[2].plot(np.imag(p_est), label="estimated")
    # axes[2].plot(np.imag(p_dir), label="recorded") 
    # axes[2].set_title("Imag part")
    # axes[3].plot(np.abs(p_est), label="estimated")
    # axes[3].plot(np.abs(p_dir), label="recorded")
    # axes[3].set_title("Magnitude")
    # for ax in axes:
    #     ax.legend()
    # plt.show()
    assert 10*np.log10(np.mean(np.abs(p_est - np.squeeze(p_dir))**2)) < -30



def test_estimated_shd_coeffs_from_omni_and_cardioid_are_similar():
    dir_dir = np.array([[0,1,0]])
    pos_omni, p_omni, p_dir, freqs, sim_info= _generate_soundfield_data_dir_center(1000, dir_dir)
    wave_num = 2 * np.pi * freqs / sim_info.c

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni, pos_omni, np.zeros((1,3)), 1, wave_num, 1e-9)

    #sph.apply_measurement_omni(shd_coeffs, np.zeros((1,3)), np.zeros((1,3)), 1, freqs)

    dir_coeffs = sph.directivity_linear(0.5, dir_dir)

    p_est = np.sum(np.conj(dir_coeffs) * shd_coeffs, axis=-1)
    assert False








def _generate_soundfield_data_omni(sr, exp_center = np.array([[0,0,0]])):
    rng = np.random.default_rng(10)
    side_len = 0.4
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
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "none"

    setup.add_mics("omni", pos_mic)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 512
    freqs = fd.get_real_freqs(num_freqs, sr)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(np.moveaxis(np.fft.fft(path, n=num_freqs), -1, 0),1,2)[:num_real_freqs,...]

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], freqs, sim.sim_info




def _generate_soundfield_data_dir_center(sr, directivity_dir):
    rng = np.random.default_rng(10)
    side_len = 0.4
    num_mic = 20

    #pos_mic = np.zeros((num_mic, 3))
    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))

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
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "none"

    setup.add_mics("omni", pos_mic)
    setup.add_mics("dir", np.zeros((1,3)), directivity_type=["cardioid"], directivity_dir=directivity_dir)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 512
    freqs = fd.get_real_freqs(num_freqs, sr)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(np.moveaxis(np.fft.fft(path, n=num_freqs), -1, 0),1,2)[:num_real_freqs,...]

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], fpaths["src"]["dir"][...,0], freqs, sim.sim_info
