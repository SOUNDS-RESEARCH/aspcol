import numpy as np
import pathlib
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
import itertools as it

import scipy.special as special

import aspcol.sphericalharmonics as sph
import aspcol.utilities as utils
import aspcol.filterdesign as fd
import aspcol.unittests.plot_methods as plm
import aspcol.kernelinterpolation as ki
import aspcol.planewaves as pw
import aspcol.montecarlo as mc

from aspsim.simulator import SimulatorSetup

import matplotlib.pyplot as plt

def show_plane_wave_response():
    freq = 400
    c = 343
    wave_num = 2 * np.pi * freq / c
    num_pos = 128
    direction = np.array([1,1,0])

    posx = np.linspace(-1, 1, num_pos)
    posy = np.linspace(-1, 1, num_pos)
    (posx, posy) = np.meshgrid(posx, posy)
    pos = np.stack((posx.flatten(), posy.flatten(), np.zeros(num_pos**2)), axis=-1)

    p = pw.plane_wave(pos, direction, wave_num)
    plm.image_scatter_freq_response(p, np.array([freq]), pos, dot_size=5)
    plt.show()





def _approx_dirac_planewave_coeffs(direction, width):
    """
    direction : ndarray of shape (3,)
        The direction of delta function. Must be a unit vector
    width : float
        The width of the approximate delta function, Measured as the distance from the direction vector. 
        The decision boundary is therefore a sphere-sphere intersection.
        The smaller the width, the more accuracte, but more prone to numerical errors.
    """

    #proj_len = np.sum(dir_vec * direction[None,:], axis=-1)
    height_r = np.sqrt(1 - width**2 / 4)
    triangle_area = height_r * width / 2
    #area2 = (width / 4) * np.sqrt(4 - width**2)

    height_w = 2 * triangle_area
    cap_height = np.sqrt(width**2 - height_w**2)

    cap_area = 2 * np.pi * cap_height
    delta_value = 1 / cap_area
    
    def dir_func(dir_vec):
        """
        dir_vec : ndarray of shape (num_directions, 3)
        input of the directional function
        """
        assert dir_vec.ndim == 2
        assert dir_vec.shape[1] == 3
        assert np.isclose(np.linalg.norm(dir_vec, axis=-1), 1).all()
        distance = np.linalg.norm(dir_vec - direction[None,:], axis=-1)
        val = delta_value * (distance < width).astype(float)
        return val
    return dir_func



def test_planewave_integral_of_dirac_coeff_function_returns_sound_pressure():
    """To get a really low error the dirac function must be very narrow. But then the monte
    carlo integration will need a LOT of samples. So this test will be very very slow if it
    is to be reliable. But with the default settings, it is easy to see that the two
    functions are very similar.
    """
    rng = np.random.default_rng()
    wave_dir = mc.uniform_random_on_sphere(1, rng)

    freq = 400
    c = 343
    wave_num = 2 * np.pi * freq / c
    exp_center = rng.uniform(-1, 1, size=(1,3))
    pos_eval = rng.uniform(-1, 1, size=(1,3)) #np.array([[0,1,0]])

    pw_coeff_func = _approx_dirac_planewave_coeffs(wave_dir, 1e-2)
    int_value = plane_wave_integral(pw_coeff_func, pos_eval, exp_center, wave_num, rng, int(1e8))
    pw_value = pw.plane_wave(pos_eval - exp_center, wave_dir, wave_num) 

    diff = int_value - pw_value
    mse = np.mean(np.abs(diff)**2)
    assert mse < 1e-2
    # num_pos = 32
    # posx = np.linspace(-1, 1, num_pos)
    # posy = np.linspace(-1, 1, num_pos)
    # (posx, posy) = np.meshgrid(posx, posy)
    # pos = np.stack((posx.flatten(), posy.flatten(), np.zeros(num_pos**2)), axis=-1)
    


def test_planewave_microphone_model_of_dirac_planewave_returns_directionality_function():
    """ Consider (7) in Brunnström et al 2024. If the sound field is a plane wave, the beta function is a
    dirac function. Then the result of (7) should be the directivity function for the plane wave angle. 

    This is quite slow since we use a monte-carlo integration with a ton of samples. This is because 
    the delta function requires mny samples to give a decent approximation of the value. 
    """

    rng = np.random.default_rng(12345)
    freq = 200
    c = 343
    wave_num = 2 * np.pi * freq / c

    mic_angle = mc.uniform_random_on_sphere(1, rng)
    directionality = pw.linear_directivity_function(0.5, mic_angle) #omni_directivity_function() 

    num_tries = 5
    error = np.zeros((num_tries,))
    for i in range(num_tries):
        pw_angle = mc.uniform_random_on_sphere(1, rng)
        dirac_func = _approx_dirac_planewave_coeffs(pw_angle, 1e-2)
        mic_val = microphone_model_plane_wave(dirac_func, directionality, np.zeros((1,3)), wave_num, rng, int(1e8))

        directionality_val = directionality(pw_angle)
        error[i] = np.abs(mic_val - directionality_val)

    assert np.mean(error) < 1e-2



def test_planewave_microphone_model_and_spherical_harmonics_are_identical_for_plane_wave_soundfield():
    rng = np.random.default_rng()
    max_order = 1
    num_pos = 10000
    pos_mic = np.zeros((1,3))
    exp_center = np.zeros((1,3))
    #pos = np.concatenate((rng.uniform(low = -1, high=1, size=(num_pos,2)), np.zeros((num_pos, 1))), axis=-1)
    pos = rng.uniform(low = -1, high=1, size=(num_pos,3))
    freq = rng.uniform(low=100, high=1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343
    
    planewave_direction = mc.uniform_random_on_sphere(1, rng)
    rad, angles = utils.cart2spherical(planewave_direction)
    orders, degrees = sph.shd_num_degrees_vector(max_order)
    neg_degrees = -degrees


    # harmonic coefficients for a plane wave
    const_factor = (-1.0)**(degrees) * (-1j)**orders * np.sqrt(4 * np.pi)
    sph_harm = special.sph_harm(neg_degrees, orders, angles[...,0], angles[...,1])
    shd_coeffs = const_factor * sph_harm
    shd_coeffs = shd_coeffs[None,:]
    #shd_plane_wave = np.squeeze(sph.reconstruct_pressure(shd_coeffs, pos, np.zeros((1, 3)), wave_num))

    # plane wave sound pressure
    #plane_wave = np.squeeze(pw.plane_wave(pos, planewave_direction, wave_num))

    mic_direction = mc.uniform_random_on_sphere(1, rng)
    dir_coeffs = sph.directivity_linear(0.5, mic_direction)
    #dir_coeffs = sph.directivity_omni()
    assert dir_coeffs.shape == shd_coeffs.shape
    p_shd = np.sum(np.conj(dir_coeffs) * shd_coeffs)
    #p_shd = sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, dir_coeffs)

    dir_func = pw.linear_directivity_function(0.5, mic_direction)
    #dir_func = pw.omni_directivity_function()
    pw_coeffs = _approx_dirac_planewave_coeffs(planewave_direction, 1e-2)
    p_pw = microphone_model_plane_wave(pw_coeffs, dir_func, rng, int(1e8))

    pass



def test_harmonic_microphone_model_applied_to_planewave_gives_back_directivity_function():
    rng = np.random.default_rng()
    max_order = 1
    #num_pos = 10000
    pos_mic = np.zeros((1,3))
    exp_center = np.zeros((1,3))
    #pos = np.concatenate((rng.uniform(low = -1, high=1, size=(num_pos,2)), np.zeros((num_pos, 1))), axis=-1)
    #pos = rng.uniform(low = -1, high=1, size=(num_pos,3))
    freq = rng.uniform(low=100, high=1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343
    
    num_dir = 30
    pw_azimuth = np.linspace(0, 2*np.pi, num_dir, endpoint=False)  
    pw_zenith = np.ones(num_dir) * np.pi / 2
    angles = np.stack((pw_azimuth, pw_zenith), axis=-1)
    #mc.uniform_random_on_sphere(1, rng)
    #rad, angles = utils.cart2spherical(planewave_direction)
    pw_direction = utils.spherical2cart(np.ones((num_dir,)), angles)
    


    # HARMONIC COEFFICIENTS FOR A PLANE WAVE
    shd_coeffs = _harmonic_coeffs_for_planewave(pw_direction, max_order)
    #shd_plane_wave = np.squeeze(sph.reconstruct_pressure(shd_coeffs, pos, np.zeros((1, 3)), wave_num))

    # plane wave sound pressure
    #plane_wave = np.squeeze(pw.plane_wave(pos, planewave_direction, wave_num))
    mic_azimuth = rng.uniform(0, 2*np.pi)
    mic_direction = utils.spherical2cart(np.ones((1,)), np.array([[mic_azimuth, np.pi/2]]))
    dir_coeffs = sph.directivity_linear(0.5, mic_direction)
    #dir_coeffs = sph.directivity_omni()
    #assert dir_coeffs.shape == shd_coeffs.shape
   
    #p_shd = np.sum(np.conj(dir_coeffs) * shd_coeffs, axis=-1)
    p_shd = np.concatenate([sph.apply_measurement(shd_coeffs[i:i+1], pos_mic, exp_center, wave_num, dir_coeffs) for i in range(num_dir)], axis=0)[:,0]

    dir_func = pw.linear_directivity_function(0.5, mic_direction)
    true_dir_func = dir_func(pw_direction)
    #pw_coeffs = _approx_dirac_planewave_coeffs(pw_direction, 1e-2)
    #p_pw = microphone_model_plane_wave(pw_coeffs, dir_func, rng, int(1e8))

    fig, ax = plt.subplots(1,1, figsize=(8,6), subplot_kw={'projection': 'polar'})
    ax.plot(pw_azimuth, np.abs(p_shd), label="SHD")
    ax.plot(pw_azimuth, true_dir_func, label="True directivity")
    ax.plot([mic_azimuth, mic_azimuth], [0, 1], label="Mic direction", linestyle="--")
    ax.legend()
    plt.show()


# def _harmonic_coeffs_for_planewave_old(pw_direction, max_order):
#     """Harmonic coefficients for a plane wave exp(-ikr^T d)
#     where r is the position and d is the direction of the plane wave.

#     The expansion center is assumed to be at the origin.

#     Parameters
#     ----------
#     pw_direction : ndarray of shape (num_pw, 3)
#         The direction of the plane wave. Must be a unit vector
#     max_order : int
#         The maximum order of the spherical harmonics expansion    
    
#     Returns
#     -------
#     shd_coeffs : ndarray of shape (num_pw, num_coeffs)
#         The spherical harmonic coefficients for the plane wave
#     """
#     assert pw_direction.ndim == 2
#     assert pw_direction.shape[1] == 3
#     assert np.allclose(np.linalg.norm(pw_direction, axis=-1), 1)

#     rad, angles = utils.cart2spherical(pw_direction)
#     orders, degrees = sph.shd_num_degrees_vector(max_order)
#     neg_degrees = -degrees
#     const_factor = (-1.0)**(degrees) * (-1j)**orders * np.sqrt(4 * np.pi)
#     sph_harm = special.sph_harm(neg_degrees[None,:], orders[None,:], angles[...,0,None], angles[...,1,None])
#     shd_coeffs = const_factor[None,:] * sph_harm
#     return shd_coeffs

def _harmonic_coeffs_for_planewave(pw_direction, max_order):
    """Harmonic coefficients for a plane wave exp(-ikr^T d)
    where r is the position and d is the direction of the plane wave.

    The expansion center is assumed to be at the origin.

    Parameters
    ----------
    pw_direction : ndarray of shape (num_pw, 3)
        The direction of the plane wave. Must be a unit vector
    max_order : int
        The maximum order of the spherical harmonics expansion    
    
    Returns
    -------
    shd_coeffs : ndarray of shape (num_pw, num_coeffs)
        The spherical harmonic coefficients for the plane wave
    """
    assert pw_direction.ndim == 2
    assert pw_direction.shape[1] == 3
    assert np.allclose(np.linalg.norm(pw_direction, axis=-1), 1)

    rad, angles = utils.cart2spherical(pw_direction)
    orders, degrees = sph.shd_num_degrees_vector(max_order)
    const_factor = (-1j)**orders * np.sqrt(4 * np.pi)
    sph_harm = special.sph_harm(degrees[None,:], orders[None,:], angles[...,0,None], angles[...,1,None])
    shd_coeffs = const_factor[None,:] * np.conj(sph_harm)
    return shd_coeffs

def _harmonic_coeffs_for_planewave_positive(pw_direction, max_order):
    """Harmonic coefficients for a plane wave exp(ikr^T d)
    where r is the position and d is the direction of the plane wave.

    The expansion center is assumed to be at the origin.

    Parameters
    ----------
    pw_direction : ndarray of shape (num_pw, 3)
        The direction of the plane wave. Must be a unit vector
    max_order : int
        The maximum order of the spherical harmonics expansion    
    
    Returns
    -------
    shd_coeffs : ndarray of shape (num_pw, num_coeffs)
        The spherical harmonic coefficients for the plane wave
    """
    assert pw_direction.ndim == 2
    assert pw_direction.shape[1] == 3
    assert np.allclose(np.linalg.norm(pw_direction, axis=-1), 1)

    rad, angles = utils.cart2spherical(pw_direction)
    orders, degrees = sph.shd_num_degrees_vector(max_order)

    # Calculate harmonic coefficients
    const_factor = 1j**orders * np.sqrt(4 * np.pi)
    sph_harm = special.sph_harm(degrees[None,:], orders[None,:], angles[...,0,None], angles[...,1,None])
    shd_coeffs = const_factor[None,:] * np.conj(sph_harm)
    return shd_coeffs


def microphone_model_plane_wave(pw_coeffs, dir_func, rng, num_samples):
    """Returns the signal that a microphone with directionality given by dir_func would record. 
    The microphone must be located at the expansion center for the plane wave coefficients.

    Defined as (7) in Brunnström et al 2024.

    Parameters
    ----------
    pw_coeffs : function
        the coefficients of the plane wave expansion defining the soundfield. This is a function that
        takes a direction unit vector and returns a complex value response.
    dir_func : function
        A function that takes direction unit vectors, ndarray of shape (num_points, 3)
        and returns a complex value response of shape (num_points)
    rng : numpy.random.Generator
        The random number generator to use
    num_samples : int
        The number of samples to use for the monte carlo integration
    """
    dir_vecs = mc.uniform_random_on_sphere(num_samples, rng)
    
    pw_vals = pw_coeffs(dir_vecs)
    dir_vals = dir_func(dir_vecs)
    mean_integrand = np.mean(pw_vals * dir_vals, axis=-1)
    
    sphere_area = 4 * np.pi
    est = sphere_area * mean_integrand
    return est


def plane_wave_integral(dir_func, pos, exp_center, wave_num, rng, num_samples):
    """Implements the integral of a function multiplied with a plane wave over a sphere.

    Defined according to (6) in Brunnström et al 2024.
    Same definition is (9) in Ribeiro 2023, but with a sign difference in the complex exponential,
    and without an expansion center. 

    Parameters
    ----------
    dir_func : function
        A function that takes direction unit vectors, ndarray of shape (num_points, 3)
        and returns a complex value response of shape (num_points)
    pos : ndarray of shape (num_pos, 3)
        The position where 
    exp_center : ndarray of shape (1,3)
        The center of the expansion
    wave_num : float
        The wave number of the plane wave
    rng : numpy.random.Generator
        The random number generator to use

    Returns 
    -------
    est : ndarray of shape (num_pos,)
        The estimated value of the integral evaluated at all the supplied positions
    """

    dir_vecs = mc.uniform_random_on_sphere(num_samples, rng)

    func_values = dir_func(dir_vecs)
    planewave_values = pw.plane_wave(pos - exp_center, dir_vecs, wave_num)
    mean_integrand = np.mean(func_values[None,:] * planewave_values, axis=-1)

    sphere_area = 4 * np.pi # must multiply by area of integration domain
    est = sphere_area * mean_integrand
    return est




















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
    pos_omni, p_omni, p_dir, freqs, sim_info = _generate_soundfield_data_dir_center(1000, dir_dir, 0)
    wave_num = 2 * np.pi * freqs / sim_info.c
    p_dir = np.squeeze(p_dir)

    exp_center = np.zeros((1,3))

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni, pos_omni, exp_center, 1, wave_num, 1e-8)
    dir_coeffs = sph.directivity_linear(0.5, dir_dir)

    p_est = np.sum(np.conj(dir_coeffs) * shd_coeffs, axis=-1)

    fig, axes = plt.subplots(1, 5, figsize=(20,6))
    axes[0].plot(10*np.log10(np.abs(p_est - np.squeeze(p_dir))**2))
    axes[0].set_title("Mean square error (dB)")
    axes[1].plot(np.real(p_est), label="estimated")
    axes[1].plot(np.real(p_dir), label="recorded")
    axes[1].set_title("Real part")
    axes[2].plot(np.imag(p_est), label="estimated")
    axes[2].plot(np.imag(p_dir), label="recorded") 
    axes[2].set_title("Imag part")
    axes[3].plot(np.abs(p_est), label="estimated")
    axes[3].plot(np.abs(p_dir), label="recorded")
    axes[3].set_title("Magnitude")
    axes[4].plot(np.fft.irfft(p_est), label="estimated")
    axes[4].plot(np.fft.irfft(p_dir), label="recorded")
    axes[4].set_title("Time domain")
    for ax in axes:
        ax.legend()
    plt.show()
    assert 10*np.log10(np.mean(np.abs(p_est - np.squeeze(p_dir))**2) / np.mean(np.abs(p_dir)**2)) < -30




def test_show_directional_microphone_model_as_function_of_angle():
    rng = np.random.default_rng(134532)
    num_src = 30
    exp_center = np.zeros((1,3))
    reg_param = 1e-10
    max_order = 8

    dir_dir = np.array([[0,1,0]])
    pos_omni, pos_src, pos_dir, p_omni, p_dir, freqs, sim_info = _generate_soundfield_data_dir_surrounded_by_sources(1000, dir_dir, num_src, 0)
    src_rad, src_angles = utils.cart2spherical(pos_src)
    p_dir = p_dir[:,0,:]
    wave_num = 2 * np.pi * freqs / sim_info.c

    shd_coeffs = [sph.inf_dimensional_shd_omni(p_omni[:,:,i], pos_omni, exp_center, max_order, wave_num, reg_param) for i in range(num_src)]
    shd_coeffs = np.stack(shd_coeffs, axis=-1)

    dir_func = pw.linear_directivity_function(0.5, dir_dir)
    dir_coeffs = _directionality_function_to_harmonic_coeffs(dir_func, 1, rng)
    #dir_coeffs = sph.directivity_linear(0.5, dir_dir)
    #p_est = np.sum(np.conj(dir_coeffs[:,:,None]) * shd_coeffs, axis=1)
    p_est = [sph.apply_measurement(shd_coeffs[:,:,i], pos_dir, exp_center, wave_num, dir_coeffs) for i in range(num_src)]
    p_est = np.concatenate(p_est, axis=-1)

   
    dir_coeffs_omni = sph.directivity_omni()
    #dir_coeffs = sph.directivity_linear(0.5, dir_dir)
    #p_est = np.sum(np.conj(dir_coeffs[:,:,None]) * shd_coeffs, axis=1)
    p_est_omni = [sph.apply_measurement(shd_coeffs[:,:,i], pos_dir, exp_center, wave_num, dir_coeffs_omni) for i in range(num_src)]
    p_est_omni = np.concatenate(p_est_omni, axis=-1)


    fig, axes = plt.subplots(1, 3, figsize=(12,6), subplot_kw={'projection': 'polar'})
    mse_per_angle = 10*np.log10(np.mean(np.abs(p_est - np.squeeze(p_dir))**2, axis=0) / np.mean(np.abs(p_dir)**2))
    axes[0].plot(src_angles[:,0], np.sum(np.abs(p_dir)**2, axis=0), label="Simulated power")
    axes[1].plot(src_angles[:,0], np.sum(np.abs(p_est)**2, axis=0), label="Microphone model power")
    axes[2].plot(src_angles[:,0], mse_per_angle)
    axes[0].set_title("Sound power")
    axes[1].set_title("Sound power")
    axes[2].set_title("MSE")
    for ax in axes:
        ax.legend()

    fig, axes = plt.subplots(1, 1, figsize=(12,6))
    mse_per_freq = 10*np.log10(np.mean(np.abs(p_est - np.squeeze(p_dir))**2, axis=-1) / np.mean(np.abs(p_dir)**2))
    axes.plot(freqs, mse_per_freq)
    #axes[1].plot(src_angles[:,0], np.sum(np.abs(p_est)**2, axis=0), label="Microphone model power")
    axes.set_title("MSE")

    num_freqs = 4
    freq_idxs = np.arange(len(freqs))[len(freqs) // 4::len(freqs) // 4]
    num_freqs = len(freq_idxs)
    fig, axes = plt.subplots(num_freqs, 3, figsize=(12, num_freqs*3), subplot_kw={'projection': 'polar'})

    for i, f_idx in enumerate(freq_idxs):
        axes[i,0].plot(src_angles[:,0], np.real(p_dir[f_idx,:]), label="Simulated")
        axes[i,0].plot(src_angles[:,0], np.real(p_est[f_idx,:]), label="Microphone model")

        axes[i,1].plot(src_angles[:,0], np.imag(p_dir[f_idx,:]), label="Simulated")
        axes[i,1].plot(src_angles[:,0], np.imag(p_est[f_idx,:]), label="Microphone model")
 
        axes[i,2].plot(src_angles[:,0], np.abs(p_dir[f_idx,:]), label="Simulated")
        axes[i,2].plot(src_angles[:,0], np.abs(p_est[f_idx,:]), label="Microphone model")

        axes[i,0].set_title(f"Real pressure {freqs[f_idx]}Hz")
        axes[i,1].set_title(f"Imag pressure {freqs[f_idx]}Hz")
        axes[i,2].set_title(f"Magnitude {freqs[f_idx]}Hz")

    for ax_row in axes:
        for ax in ax_row:
            ax.legend()
    plt.show()
    assert True
    #assert 10*np.log10(np.mean(np.abs(p_est - np.squeeze(p_dir))**2)) < -30




def test_directivity_coefficients_from_analytic_and_numerical_solver_is_identical():
    rng = np.random.default_rng(123453)
    max_order = 3
    mic_direction = mc.uniform_random_on_sphere(1, rng)
    #mic_direction = rng.normal(size=(1,3)) #np.array([[1,0,0]])
    #mic_direction = mic_direction / np.linalg.norm(mic_direction)
    dir_coeffs = sph.directivity_linear(0.5, mic_direction, max_order)
    dir_func = pw.linear_directivity_function(0.5, mic_direction)
    #dir_func = omni_directivity_function()

    dir_coeffs_estimated = _directionality_function_to_harmonic_coeffs(dir_func, max_order, rng, int(1e8))

    mse = np.mean(np.abs(dir_coeffs - dir_coeffs_estimated))**2 / np.mean(np.abs(dir_coeffs))**2
    assert mse < 1e-4


def _directionality_function_to_harmonic_coeffs(dir_func, max_order, rng=None, num_samples = 10**6):
    """Implements (10) from Brunnstroem et al 2024. 

    dir_func : function
        A function that takes direction unit vector and returns the microphone response
    
    Returns
    -------
    dir_coeffs : ndarray of shape (1, num_coeffs)
    """
    num_samples = 10**6
    if rng is None:
        rng = np.random.default_rng()

    dir_vecs = mc.uniform_random_on_sphere(num_samples, rng)
    rad, angles = utils.cart2spherical(dir_vecs)
    orders, degrees = sph.shd_num_degrees_vector(max_order)
    dir_val = dir_func(dir_vecs)

    angles = angles[None,...]
    Y = special.sph_harm(degrees[:,None], orders[:,None], angles[...,0], angles[...,1])

    sphere_area = 4 * np.pi # must multiply by area of integration domain
    scaling = ((-1j) ** orders) / np.sqrt(4 * np.pi)
    integral_value = np.mean(np.conj(Y) * np.conj(dir_val[None,:]), axis=-1)
    est = scaling * sphere_area * integral_value
    return est[None,:]

    #(-1j)**n / np.sqrt(4 * np.pi)


def test_estimated_shd_coeffs_for_plane_wave_sound_field_are_close_to_true_value():
    rng = np.random.default_rng(123456)
    c = 343
    exp_center = np.zeros((1,3))
    reg_param = 1e-8
    num_mic = 15
    max_order = 7
    

    mic_directions = mc.uniform_random_on_sphere(num_mic, rng)
    #dir_coeffs = [sph.directivity_linear(0.5, mic_direction[i,:]) for i in range(num_mic)]
    dir_coeffs = np.concatenate([sph.directivity_linear(0.5, mic_directions[i:i+1,:]) for i in range(num_mic)], axis=0)
    #dir_coeffs2 = sph.directivity_linear(0.5, mic_direction)
    pos_mic = rng.uniform(-0.5, 0.5, size=(num_mic,3))
    exp_center = np.zeros((1,3))

    freq = rng.uniform(low=100, high=1000, size=(1,))
    wave_num = 2 * np.pi * freq / c
    
    pw_direction = mc.uniform_random_on_sphere(1, rng)
    # SOUND PRESSURE FOR A PLANE WAVE
    sound_pressure = pw.plane_wave(pos_mic, pw_direction, wave_num)[None,:,0]

    # TRUE HARMONIC COEFFICIENTS FOR A PLANE WAVE
    shd_coeffs = _harmonic_coeffs_for_planewave(pw_direction, max_order)

    # RECORDED SOUND FOR A PLANE WAVE
    p_from_shd_omni = sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, sph.directivity_omni())
    #p_from_shd_dir = np.concatenate([sph.apply_measurement(shd_coeffs, pos_mic[i:i+1], exp_center, wave_num, dir_coeffs[i:i+1,:]) for i in range(num_mic)], axis=-1)


    #assert np.mean(np.abs(p_from_shd_omni - sound_pressure)**2) / np.mean(np.abs(sound_pressure)**2) < 1e-5

    
    #shd_coeffs_omni = sph.inf_dimensional_shd_omni(p_from_shd_omni, pos_mic, exp_center, max_order, wave_num, reg_param)
    #shd_coeffs_dir = sph.inf_dimensional_shd(p_from_shd_dir, pos_mic, exp_center, max_order, wave_num, reg_param, dir_coeffs=dir_coeffs)
    shd_coeffs_dir = sph.inf_dimensional_shd(p_from_shd_omni, pos_mic, exp_center, max_order, wave_num, reg_param, dir_coeffs=dir_coeffs[None,:,:])


    mse_incr_order_omni = [np.mean(np.abs(shd_coeffs_omni[:,:i] - shd_coeffs[:,:i])**2) / np.mean(np.abs(shd_coeffs[:,:i])**2) for i in range(1, shd_coeffs.shape[-1])]
    mse_incr_order_dir = [np.mean(np.abs(shd_coeffs_dir[:,:i] - shd_coeffs[:,:i])**2) / np.mean(np.abs(shd_coeffs[:,:i])**2) for i in range(1, shd_coeffs.shape[-1])]
    #mse = np.mean(np.abs(shd_coeffs - shd_coeffs)**2) / np.mean(np.abs(shd_coeffs)**2)

    plt.plot(mse_incr_order_omni, label="omni")
    plt.plot(mse_incr_order_dir, label="dir")
    plt.legend()
    plt.show()
    pass
    #shd_coeffs_dir = sph.inf_dimensional_shd(p_dir, pos_omni, exp_center, 1, wave_num, reg_param, dir_coeffs=sph.directivity_linear(0.5, dir_dir))
    #p_est_dir = sph.reconstruct_pressure(shd_coeffs_dir, pos_omni, exp_center, wave_num)
    #sph.apply_measurement_omni(shd_coeffs, np.zeros((1,3)), np.zeros((1,3)), 1, freqs)

    #dir_coeffs = sph.directivity_linear(0.5, dir_dir)
    #p_est = np.sum(np.conj(dir_coeffs) * shd_coeffs, axis=-1)

    # diff = shd_coeffs_true - shd_coeffs_dir
    # mse_per_freq = 10 * np.log10(np.mean(np.abs(diff)**2, axis=-1) / np.mean(np.abs(shd_coeffs_true)**2, axis=-1))
    # mse_per_shd = 10 * np.log10(np.mean(np.abs(diff)**2, axis=0) / np.mean(np.abs(shd_coeffs_true)**2, axis=0))
    # fig, axes = plt.subplots(1, 2, figsize=(15,6))
    # axes[0].plot(freqs, mse_per_freq)
    # axes[0].set_title("Mean square error per frequency (dB)")
    # axes[1].plot(mse_per_shd)
    # axes[1].set_title("Mean square error per SHD coefficient (dB)")

    # fig, axes = plt.subplots(1, 3, figsize=(15,6))
    # axes[0].plot(np.real(shd_coeffs_dir[:,0]), label="dir")
    # axes[0].plot(np.real(shd_coeffs_true[:,0]), label="omni")
    # axes[0].set_title("Real part")
    # axes[1].plot(np.imag(shd_coeffs_dir[:,0]), label="dir")
    # axes[1].plot(np.imag(shd_coeffs_true[:,0]), label="omni")
    # axes[1].set_title("Imag part")
    # axes[2].plot(np.abs(shd_coeffs_dir[:,0]), label="dir")
    # axes[2].plot(np.abs(shd_coeffs_true[:,0]), label="omni")
    # axes[2].set_title("Magnitude")
    # for ax in axes:
    #     ax.legend()

    # plm.image_scatter_freq_response({"omni" : p_est_omni, "dir" : p_est_dir}, freqs, pos_omni)
    # plt.show()
    # assert False


def test_estimated_shd_coeffs_are_similar_with_omni_and_cardiod_microphones():
    dir_dir = np.array([[0,1,0]])
    pos_omni, p_omni, p_dir, freqs, sim_info= _generate_soundfield_data_dir_and_omni(1000, dir_dir, 0)
    wave_num = 2 * np.pi * freqs / sim_info.c
    exp_center = np.zeros((1,3))
    reg_param = 1e-6

    ds_freq = 20
    p_omni = np.ascontiguousarray(p_omni[::ds_freq,...])
    p_dir = np.ascontiguousarray(p_dir[::ds_freq,...])
    wave_num = np.ascontiguousarray(wave_num[::ds_freq])
    freqs = np.ascontiguousarray(freqs[::ds_freq])

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni, pos_omni, exp_center, 1, wave_num, reg_param)
    shd_coeffs_dir = sph.inf_dimensional_shd(p_dir, pos_omni, exp_center, 1, wave_num, reg_param, dir_coeffs=sph.directivity_linear(0.5, dir_dir))

    p_est_omni = sph.reconstruct_pressure(shd_coeffs, pos_omni, exp_center, wave_num)
    p_est_dir = sph.reconstruct_pressure(shd_coeffs_dir, pos_omni, exp_center, wave_num)
    #sph.apply_measurement_omni(shd_coeffs, np.zeros((1,3)), np.zeros((1,3)), 1, freqs)

    #dir_coeffs = sph.directivity_linear(0.5, dir_dir)
    #p_est = np.sum(np.conj(dir_coeffs) * shd_coeffs, axis=-1)

    diff = shd_coeffs - shd_coeffs_dir
    mse_per_freq = 10 * np.log10(np.mean(np.abs(diff)**2, axis=-1) / np.mean(np.abs(shd_coeffs)**2, axis=-1))
    mse_per_shd = 10 * np.log10(np.mean(np.abs(diff)**2, axis=0) / np.mean(np.abs(shd_coeffs)**2, axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    axes[0].plot(freqs, mse_per_freq)
    axes[0].set_title("Mean square error per frequency (dB)")
    axes[1].plot(mse_per_shd)
    axes[1].set_title("Mean square error per SHD coefficient (dB)")

    fig, axes = plt.subplots(1, 3, figsize=(15,6))
    axes[0].plot(np.real(shd_coeffs_dir[:,0]), label="dir")
    axes[0].plot(np.real(shd_coeffs[:,0]), label="omni")
    axes[0].set_title("Real part")
    axes[1].plot(np.imag(shd_coeffs_dir[:,0]), label="dir")
    axes[1].plot(np.imag(shd_coeffs[:,0]), label="omni")
    axes[1].set_title("Imag part")
    axes[2].plot(np.abs(shd_coeffs_dir[:,0]), label="dir")
    axes[2].plot(np.abs(shd_coeffs[:,0]), label="omni")
    axes[2].set_title("Magnitude")
    for ax in axes:
        ax.legend()

    plm.image_scatter_freq_response({"omni" : p_est_omni, "dir" : p_est_dir}, freqs, pos_omni)
    plt.show()
    assert False








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
    freqs = fd.get_real_freqs(num_freqs, sr)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(np.moveaxis(np.fft.fft(path, n=num_freqs), -1, 0),1,2)[:num_real_freqs,...]

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
    freqs = fd.get_real_freqs(num_freqs, sr)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(np.moveaxis(np.fft.fft(path, n=num_freqs), -1, 0),1,2)[:num_real_freqs,...]

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
    freqs = fd.get_real_freqs(num_freqs, sr)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(np.moveaxis(np.fft.fft(path, n=num_freqs), -1, 0),1,2)[:num_real_freqs,...]

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
    freqs = fd.get_real_freqs(num_freqs, sr)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(np.moveaxis(np.fft.fft(path, n=num_freqs), -1, 0),1,2)[:num_real_freqs,...]

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], fpaths["src"]["dir"][...,0], freqs, sim.sim_info

