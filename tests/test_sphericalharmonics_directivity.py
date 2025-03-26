import numpy as np
import scipy.special as special
import scipy.signal as spsig
import pytest

import aspcol.sphericalharmonics as sph
import aspcol.utilities as utils
import plot_methods as plm
import aspcol.planewaves as pw

import aspcol.plot as aspplot


from aspsim.simulator import SimulatorSetup
import aspsim.room.region as region
import aspsim.signal.sources as sources
import aspsim.diagnostics.diagnostics as dg

import aspcore.filter as fc
import aspcore.montecarlo as mc
import aspcore.fouriertransform as ft
import aspcore.filterdesign as fd
import aspcore.pseq as pseq

import matplotlib.pyplot as plt


@pytest.fixture(scope="session")
def fig_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("figs")




def test_harmonic_microphone_model_applied_to_planewave_gives_back_directivity_function():
    rng = np.random.default_rng()
    max_order = 1
    pos_mic = np.zeros((1,3))
    exp_center = np.zeros((1,3))
    freq = rng.uniform(low=100, high=1000, size=(1,))
    wave_num = 2 * np.pi * freq / 343
    
    num_dir = 30
    pw_direction = mc.uniform_random_on_sphere(num_dir, rng)

    # DATA TO PLOT IN POLAR COORDINATES
    # mic_azimuth = rng.uniform(0, 2*np.pi)
    # mic_direction = utils.spherical2cart(np.ones((1,)), np.array([[mic_azimuth, np.pi/2]]))
    # pw_azimuth = np.linspace(0, 2*np.pi, num_dir, endpoint=False)  
    # pw_zenith = np.ones(num_dir) * np.pi / 2
    # angles = np.stack((pw_azimuth, pw_zenith), axis=-1)
    # pw_direction = utils.spherical2cart(np.ones((num_dir,)), angles)

    # HARMONIC COEFFICIENTS FOR A PLANE WAVE
    shd_coeffs = pw.shd_coeffs_for_planewave(pw_direction, max_order)
    mic_direction = mc.uniform_random_on_sphere(1, rng)

    A = 0.5
    dir_coeffs = sph.directivity_linear(A, mic_direction)
    p_shd = np.concatenate([sph.apply_measurement(shd_coeffs[i:i+1], pos_mic, exp_center, wave_num, dir_coeffs) for i in range(num_dir)], axis=0)[:,0]

    dir_func = pw.linear_directivity_function(A, mic_direction)
    true_dir_func = dir_func(pw_direction)

    # fig, ax = plt.subplots(1,1, figsize=(8,6), subplot_kw={'projection': 'polar'})
    # ax.plot(pw_azimuth, np.abs(p_shd), label="SHD")
    # ax.plot(pw_azimuth, true_dir_func, label="True directivity")
    # #ax.plot([mic_azimuth, mic_azimuth], [0, 1], label="Mic direction", linestyle="--")
    # ax.legend()
    # plt.show()

    assert np.mean(np.abs(true_dir_func - p_shd)**2) / np.mean(np.abs(true_dir_func)**2) < 1e-12


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
    mic_direction = np.array([[0,1,0]])
    pos_dir = np.zeros((1, 3))
    pos_omni, p_omni, p_dir, freqs, sim_info = _generate_soundfield_data_dir(2000, mic_direction, 0, pos_dir, 0.5, 100)
    wave_num = 2 * np.pi * freqs / sim_info.c
    p_dir = np.squeeze(p_dir)

    exp_center = np.zeros((1,3))

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni, pos_omni, exp_center, 1, wave_num, 1e-8)
    dir_coeffs = sph.directivity_linear(0.5, mic_direction)
    p_est = sph.apply_measurement(shd_coeffs, pos_dir, exp_center, wave_num, dir_coeffs)[:,0]

    #p_est = np.sum(np.conj(dir_coeffs) * shd_coeffs, axis=-1)

    fig, axes = plt.subplots(1, 5, figsize=(20,6))
    axes[0].plot(10*np.log10(np.abs(p_est - np.squeeze(p_dir))**2))
    axes[0].set_title("Mean square error (dB)")
    axes[1].plot(np.real(p_est), label="microphone model")
    axes[1].plot(np.real(p_dir), label="simulation")
    axes[1].set_title("Real part")
    axes[2].plot(np.imag(p_est), label="microphone model")
    axes[2].plot(np.imag(p_dir), label="simulation") 
    axes[2].set_title("Imag part")
    axes[3].plot(np.abs(p_est), label="microphone model")
    axes[3].plot(np.abs(p_dir), label="simulation")
    axes[3].set_title("Magnitude")
    axes[4].plot(ft.irfft(p_est), label= "microphone model")
    axes[4].plot(ft.irfft(p_dir), label="simulation")
    axes[4].set_title("Time domain")
    for ax in axes:
        ax.legend()
    plt.show()
    #assert 10*np.log10(np.mean(np.abs(p_est - np.squeeze(p_dir))**2) / np.mean(np.abs(p_dir)**2)) < -30


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
    rng = np.random.default_rng()
    max_order = 1
    mic_direction = mc.uniform_random_on_sphere(1, rng)
    #mic_direction = rng.normal(size=(1,3)) #np.array([[1,0,0]])
    #mic_direction = mic_direction / np.linalg.norm(mic_direction)
    A = rng.uniform(0, 1)
    dir_coeffs = sph.directivity_linear(A, mic_direction, max_order)
    dir_func = pw.linear_directivity_function(A, mic_direction)
    #dir_func = omni_directivity_function()

    dir_coeffs_estimated = _directionality_function_to_harmonic_coeffs(dir_func, max_order, rng, int(1e6))

    mse = np.mean(np.abs(dir_coeffs - dir_coeffs_estimated))**2 / np.mean(np.abs(dir_coeffs))**2
    assert mse < 1e-5


def _directionality_function_to_harmonic_coeffs(dir_func, max_order, rng=None, num_samples = 10**6):
    """Implements (10) from Brunnstroem et al 2024. 

    dir_func : function
        A function that takes direction unit vector and returns the microphone response
    
    Returns
    -------
    dir_coeffs : ndarray of shape (1, num_coeffs)
    """
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
    rng = np.random.default_rng()
    c = 343
    exp_center = np.zeros((1,3))
    reg_param = 1e-8
    num_mic = 120
    max_order = 10
    dir_constant = rng.uniform(0, 1)

    mic_directions = mc.uniform_random_on_sphere(num_mic, rng)
    dir_coeffs = np.concatenate([sph.directivity_linear(dir_constant, mic_directions[i:i+1,:]) for i in range(num_mic)], axis=0)
    pos_mic = rng.uniform(-0.5, 0.5, size=(num_mic,3))
    exp_center = np.zeros((1,3))

    freq = rng.uniform(low=100, high=800, size=(1,))
    wave_num = 2 * np.pi * freq / c
    
    pw_direction = mc.uniform_random_on_sphere(1, rng)
    # SOUND PRESSURE FOR A PLANE WAVE
    #sound_pressure = pw.plane_wave(pos_mic, pw_direction, wave_num)[None,:,0]

    # TRUE HARMONIC COEFFICIENTS FOR A PLANE WAVE
    shd_coeffs = pw.shd_coeffs_for_planewave(pw_direction, max_order)

    # RECORDED SOUND FOR A PLANE WAVE
    p_from_shd_omni = sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, sph.directivity_omni())
    p_from_shd_dir = np.concatenate([sph.apply_measurement(shd_coeffs, pos_mic[i:i+1], exp_center, wave_num, dir_coeffs[i:i+1,:]) for i in range(num_mic)], axis=-1)

    shd_coeffs_omni = sph.inf_dimensional_shd_omni(p_from_shd_omni, pos_mic, exp_center, max_order, wave_num, reg_param)
    shd_coeffs_dir = sph.inf_dimensional_shd(p_from_shd_dir, pos_mic, exp_center, max_order, wave_num, reg_param, dir_coeffs=dir_coeffs[None,:,:])

    mse_incr_order_omni = [np.mean(np.abs(shd_coeffs_omni[:,:i] - shd_coeffs[:,:i])**2) / np.mean(np.abs(shd_coeffs[:,:i])**2) for i in range(1, shd_coeffs.shape[-1])]
    mse_incr_order_dir = [np.mean(np.abs(shd_coeffs_dir[:,:i] - shd_coeffs[:,:i])**2) / np.mean(np.abs(shd_coeffs[:,:i])**2) for i in range(1, shd_coeffs.shape[-1])]

    # plt.plot(mse_incr_order_omni, label="omni")
    # plt.plot(mse_incr_order_dir, label="dir")
    # plt.legend()
    # plt.show()

    assert np.sum(mse_incr_order_dir) < 3 * np.sum(mse_incr_order_omni) # The directivity model should not be much worse than the omni model
    assert np.mean(mse_incr_order_omni[:4]) < 1e-4 #the lowest orders should be well estimated
    assert np.mean(mse_incr_order_dir[:4]) < 1e-4 #the lowest orders should be well estimated



def test_estimated_sound_pressure_for_plane_wave_sound_field_are_close_to_true_value_with_directional_microphones():
    rng = np.random.default_rng(2345678)
    c = 343
    exp_center = rng.uniform(-0.2, 0.2, size=(1,3)) #np.array([[0, 0, 0]]) #np.zeros((1,3))
    reg_param = 1e-8
    num_mic = 80
    max_order = 15
    dir_constant = rng.uniform(0, 1)

    mic_directions = mc.uniform_random_on_sphere(num_mic, rng)
    #dir_coeffs = np.concatenate([sph.directivity_linear(dir_constant, mic_directions[i:i+1,:]) for i in range(num_mic)], axis=0)
    dir_coeffs = sph.directivity_linear(dir_constant, mic_directions)
    pos_mic = rng.uniform(-0.5, 0.5, size=(num_mic,3))
    exp_center = np.zeros((1,3))

    #freq = rng.uniform(low=100, high=500, size=(1,))
    freq = np.array([400]) 
    wave_num = 2 * np.pi * freq / c
    pw_direction = mc.uniform_random_on_sphere(1, rng)

    # TRUE HARMONIC COEFFICIENTS FOR A PLANE WAVE
    shd_coeffs = pw.shd_coeffs_for_planewave(pw_direction, max_order)

    # RECORDED SOUND FOR A PLANE WAVE
    p_from_shd_omni = sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, sph.directivity_omni())
    #p_from_shd_dir = np.concatenate([sph.apply_measurement(shd_coeffs, pos_mic[i:i+1], exp_center, wave_num, dir_coeffs[i:i+1,:]) for i in range(num_mic)], axis=-1)
    p_from_shd_dir = sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, dir_coeffs[None,:,:])

    shd_coeffs_omni = sph.inf_dimensional_shd_omni(p_from_shd_omni, pos_mic, exp_center, max_order, wave_num, reg_param)
    shd_coeffs_dir = sph.inf_dimensional_shd(p_from_shd_dir, pos_mic, exp_center, max_order, wave_num, reg_param, dir_coeffs=dir_coeffs[None,:,:])

    num_eval = 300
    pos_eval = rng.uniform(-0.45, 0.45, size=(num_eval,3))
    sound_pressure = pw.plane_wave(pos_eval, pw_direction, wave_num)[None,:,0]

    p_omni = sph.reconstruct_pressure(shd_coeffs_omni, pos_eval, exp_center, wave_num)
    p_dir = sph.reconstruct_pressure(shd_coeffs_dir, pos_eval, exp_center, wave_num)

    mse_omni = np.mean(np.abs(sound_pressure - p_omni)**2) / np.mean(np.abs(sound_pressure)**2)
    mse_dir = np.mean(np.abs(sound_pressure - p_dir)**2) / np.mean(np.abs(sound_pressure)**2)

    assert mse_dir < 10 * mse_omni
    assert mse_dir < 1e-4
    assert mse_omni < 1e-4


def test_estimated_shd_coeffs_from_ism_sound_field_are_similar_with_omni_and_cardiod_microphones():
    rng = np.random.default_rng(12345)

    num_mic = 50
    side_len = 0.4

    pos_omni = np.concatenate((rng.uniform(-side_len / 2, side_len / 2, size=(num_mic,2)), np.zeros((num_mic, 1))), axis=-1)
    mic_directivity = mc.uniform_random_on_sphere(num_mic, rng)

    pos_omni, pos_eval, p_omni, p_dir, p_eval, freqs, sim_info = _generate_soundfield_data_dir_and_omni(1000, mic_directivity, rt60=0, side_line=side_len, pos_omni=pos_omni, rng=rng)
    wave_num = 2 * np.pi * freqs / sim_info.c
    exp_center = np.zeros((1,3))
    reg_param = 1e-6
    max_order = 8

    ds_freq = 20
    p_omni = np.ascontiguousarray(p_omni[ds_freq//2::ds_freq,...])
    p_dir = np.ascontiguousarray(p_dir[ds_freq//2::ds_freq,...])
    wave_num = np.ascontiguousarray(wave_num[ds_freq//2::ds_freq])
    freqs = np.ascontiguousarray(freqs[ds_freq//2::ds_freq])
    dir_coeffs = np.tile(sph.directivity_linear(0.5, mic_directivity), (freqs.shape[0], 1, 1))

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni, pos_omni, exp_center, max_order, wave_num, reg_param)
    shd_coeffs_dir = sph.inf_dimensional_shd(p_dir, pos_omni, exp_center, max_order, wave_num, reg_param, dir_coeffs)

    diff = shd_coeffs - shd_coeffs_dir
    mse_per_freq = 10 * np.log10(np.mean(np.abs(diff)**2, axis=-1) / np.mean(np.abs(shd_coeffs)**2, axis=-1))
    mse_per_shd = 10 * np.log10(np.mean(np.abs(diff)**2, axis=0) / np.mean(np.abs(shd_coeffs)**2, axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    axes[0].plot(freqs, mse_per_freq)
    axes[0].set_title("MSE for shd_coeffs per frequency (dB)")
    axes[1].plot(mse_per_shd)
    axes[1].set_title("MSE for shd_coeffs per coefficient (dB)")
    for ax in axes:
        ax.set_ylabel("Magnitude")

    fig, axes = plt.subplots(1, 3, figsize=(15,6))
    axes[0].plot(freqs, np.real(shd_coeffs_dir[:,0]), label="dir")
    axes[0].plot(freqs, np.real(shd_coeffs[:,0]), label="omni")
    axes[0].set_title("Order 0: Real part")
    axes[1].plot(freqs, np.imag(shd_coeffs_dir[:,0]), label="dir")
    axes[1].plot(freqs, np.imag(shd_coeffs[:,0]), label="omni")
    axes[1].set_title("Order 0: Imag part")
    axes[2].plot(freqs, np.abs(shd_coeffs_dir[:,0]), label="dir")
    axes[2].plot(freqs, np.abs(shd_coeffs[:,0]), label="omni")
    axes[2].set_title("Order 0: Magnitude")
    for ax in axes:
        ax.legend()
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
    plt.show()



def test_estimated_sound_pressure_with_cardioid_microphone_gives_same_value_as_omni_microphones():
    """
    """
    rng = np.random.default_rng(123456)
    sr = 1000
    max_order = 16
    exp_center = np.zeros((1,3))
    reg_param = 1e-8
    #num_rotations = 30
    side_len = 0.7

    #pos_mic = np.zeros((1,3))#rng.uniform(-0.5, 0.5, size=(1,3))
    #pos_cardioid = np.ones((num_rotations,1)) * pos_mic

    num_mic = 24
    cuboid = region.Cuboid((side_len, side_len, side_len), np.zeros(3), rng=rng)
    pos_mic = cuboid.sample_points(num_mic)
    
    rect = region.Rectangle((side_len, side_len), np.zeros(3), (0.04, 0.04))
    pos_eval = rect.equally_spaced_points()

    mic_angle = np.linspace(0, 2*np.pi, num_mic, endpoint=False)
    mic_dir = utils.spherical2cart(np.ones(num_mic), np.stack((mic_angle, np.pi/2 * np.ones(num_mic)),axis=-1))
    #mic_dir = mc.uniform_random_on_sphere(num_rotations, rng)

    num_src = 25
    src_direction = mc.uniform_random_on_sphere(num_src, rng)
    src_distance = rng.uniform(2, 4, size=(num_src,))
    pos_src = src_direction * src_distance[:,None]

    pos_omni, pos_cardioid, pos_eval, p_omni_sim, p_cardioid_sim, p_eval_sim, freqs, sim_info = _generate_soundfield_data_dir_and_omni(sr, mic_dir, rt60=0, side_len=side_len, pos_omni=pos_mic, pos_cardioid=pos_mic, pos_eval=pos_eval, pos_src = pos_src, rng=rng)
    fi = int(p_omni_sim.shape[0] * 0.8)
    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_omni_sim = np.sum(p_omni_sim[fi:fi+1,...], axis=-1)
    p_cardioid_sim = np.sum(p_cardioid_sim[fi:fi+1,...], axis=-1)
    p_eval_sim = np.sum(p_eval_sim[fi:fi+1,...], axis=-1)

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni_sim, pos_mic, exp_center, max_order, wave_num, reg_param)
    p_eval_shd = sph.apply_measurement_omni(shd_coeffs, pos_eval, exp_center, wave_num)
    
    dir_coeffs = sph.directivity_linear(0.5, mic_dir)[None,...]
    shd_coeffs_cardioid = sph.inf_dimensional_shd(p_cardioid_sim, pos_cardioid, exp_center, max_order, wave_num, reg_param, dir_coeffs=dir_coeffs)
    p_eval_shd_cardioid = sph.apply_measurement_omni(shd_coeffs_cardioid, pos_eval, exp_center, wave_num)

    mse_omni = np.mean(np.abs(p_eval_sim - p_eval_shd)**2, axis=-1) / np.mean(np.abs(p_eval_sim)**2, axis=-1)
    mse_cardioid = np.mean(np.abs(p_eval_sim - p_eval_shd_cardioid)**2, axis=-1) / np.mean(np.abs(p_eval_sim)**2, axis=-1)

    # fig, ax = plt.subplots(1,1, figsize=(8,6))
    # ax.plot(freqs, 10*np.log10(mse_omni), label="omni")
    # ax.plot(freqs, 10*np.log10(mse_cardioid), label="dir")
    # ax.set_title("MSE between omni and dir microphone model (dB)")
    # ax.legend()
    # plt.show()

    plm.image_scatter_freq_response({"omni error" : p_eval_shd - p_eval_sim, "cardioid error" : p_eval_shd_cardioid - p_eval_sim}, freq, pos_eval)

    plm.image_scatter_freq_response({"omni" : p_eval_shd, "cardioid" : p_eval_shd_cardioid}, freq, pos_eval)
    plt.show()


def test_estimated_sound_pressure_for_image_source_method_sound_field_are_close_to_true_value_with_directional_microphones():
    rng = np.random.default_rng(12345)

    num_mic = 50
    side_len = 0.4
    pos_omni = np.concatenate((rng.uniform(-side_len / 2, side_len / 2, size=(num_mic,2)), np.zeros((num_mic, 1))), axis=-1)

    mic_directivity = mc.uniform_random_on_sphere(num_mic, rng)
    #mic_directivity = mc.uniform_random_on_sphere(num_mic, rng)
    pos_omni, pos_eval, p_omni, p_dir, p_eval, freqs, sim_info = _generate_soundfield_data_dir_and_omni(1000, mic_directivity, rt60=0, side_line=side_len, pos_omni=pos_omni, rng=rng)
    wave_num = 2 * np.pi * freqs / sim_info.c
    exp_center = np.zeros((1,3))
    reg_param = 1e-8
    max_order = 10

    ds_freq = 20
    p_omni = np.ascontiguousarray(p_omni[ds_freq//2::ds_freq,...])
    p_dir = np.ascontiguousarray(p_dir[ds_freq//2::ds_freq,...])
    p_eval = np.ascontiguousarray(p_eval[ds_freq//2::ds_freq,...])
    wave_num = np.ascontiguousarray(wave_num[ds_freq//2::ds_freq])
    freqs = np.ascontiguousarray(freqs[ds_freq//2::ds_freq])
    dir_coeffs = np.tile(sph.directivity_linear(0.5, mic_directivity), (freqs.shape[0], 1, 1))

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni, pos_omni, exp_center, max_order, wave_num, reg_param)
    shd_coeffs_dir = sph.inf_dimensional_shd(p_dir, pos_omni, exp_center, max_order, wave_num, reg_param, dir_coeffs)

    #num_eval = 500
    #pos_eval = np.concatenate((rng.uniform(-side_len / 2, side_len / 2, size=(num_eval,2)), np.zeros((num_eval,1))), axis=-1)
    p_est_omni = sph.reconstruct_pressure(shd_coeffs, pos_eval, exp_center, wave_num)
    p_est_dir = sph.reconstruct_pressure(shd_coeffs_dir, pos_eval, exp_center, wave_num)
    #sph.apply_measurement_omni(shd_coeffs, np.zeros((1,3)), np.zeros((1,3)), 1, freqs)

    mse_diff = np.mean(np.abs(p_est_omni - p_est_dir)**2, axis=-1) / np.mean(np.abs(p_est_omni)**2, axis=-1)
    mse_omni = np.mean(np.abs(p_eval - p_est_omni)**2, axis=-1) / np.mean(np.abs(p_eval)**2, axis=-1)
    mse_dir = np.mean(np.abs(p_eval - p_est_dir)**2, axis=-1) / np.mean(np.abs(p_eval)**2, axis=-1)

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(freqs, 10*np.log10(mse_diff), label="diff")
    ax.plot(freqs, 10*np.log10(mse_omni), label="omni")
    ax.plot(freqs, 10*np.log10(mse_dir), label="dir")
    ax.set_title("MSE between omni and dir microphone model (dB)")
    ax.legend()
    plt.show()


    plm.image_scatter_freq_response({"omni" : p_est_omni, "dir" : p_est_dir}, freqs, pos_eval)
    plt.show()
    #dir_coeffs = sph.directivity_linear(0.5, dir_dir)
    #p_est = np.sum(np.conj(dir_coeffs) * shd_coeffs, axis=-1)

    # diff = shd_coeffs - shd_coeffs_dir
    # mse_per_freq = 10 * np.log10(np.mean(np.abs(diff)**2, axis=-1) / np.mean(np.abs(shd_coeffs)**2, axis=-1))
    # mse_per_shd = 10 * np.log10(np.mean(np.abs(diff)**2, axis=0) / np.mean(np.abs(shd_coeffs)**2, axis=0))
    # fig, axes = plt.subplots(1, 2, figsize=(15,6))
    # axes[0].plot(freqs, mse_per_freq)
    # axes[0].set_title("MSE for shd_coeffs per frequency (dB)")
    # axes[1].plot(mse_per_shd)
    # axes[1].set_title("MSE for shd_coeffs per coefficient (dB)")

    # fig, axes = plt.subplots(1, 3, figsize=(15,6))
    # axes[0].plot(np.real(shd_coeffs_dir[:,0]), label="dir")
    # axes[0].plot(np.real(shd_coeffs[:,0]), label="omni")
    # axes[0].set_title("Real part")
    # axes[1].plot(np.imag(shd_coeffs_dir[:,0]), label="dir")
    # axes[1].plot(np.imag(shd_coeffs[:,0]), label="omni")
    # axes[1].set_title("Imag part")
    # axes[2].plot(np.abs(shd_coeffs_dir[:,0]), label="dir")
    # axes[2].plot(np.abs(shd_coeffs[:,0]), label="omni")
    # axes[2].set_title("Magnitude")
    # for ax in axes:
    #     ax.legend()

    # plm.image_scatter_freq_response({"omni" : p_est_omni, "dir" : p_est_dir}, freqs, pos_omni)
    # plt.show()
    # assert False


def test_cardoid_directivity_function_is_same_as_pyroomacoustics_cardoid_function():
    import pyroomacoustics as pra
    rng = np.random.default_rng()
    mic_direction = mc.uniform_random_on_sphere(1, rng)
    radius, angles = utils.cart2spherical(mic_direction)
    dir_obj = pra.directivities.CardioidFamily(
        orientation=pra.directivities.DirectionVector(azimuth=angles[0,0], colatitude=angles[0,1], degrees=False),
        pattern_enum=pra.directivities.DirectivityPattern.CARDIOID,
    )
    test_directions = mc.uniform_random_on_sphere(100, rng)
    _, angles = utils.cart2spherical(test_directions)

    response_pra = dir_obj.get_response(angles[:,0], angles[:,1], degrees = False)
    dir_func = pw.linear_directivity_function(0.5, mic_direction)
    response_pw = dir_func(test_directions)

    # fig, ax = plt.subplots(1,1, figsize=(8,6), subplot_kw={'projection': 'polar'})
    # ax.plot(azimuth, response_pra, label = "pra")
    # ax.plot(azimuth, response_pw, label = "pw")
    # ax.legend()
    # plt.show()
    assert np.allclose(response_pra, response_pw)





def test_measurement_model_with_cardioid_microphone_gives_same_value_as_simulated_cardioid_microphone():
    """This is intended to be the most general test for the measurement model. For a generic simulated sound field, 
    the shd model should return the same result. 
    
    """
    rng = np.random.default_rng(123456)
    sr = 1000
    max_order = 16
    exp_center = np.zeros((1,3))
    reg_param = 1e-8
    num_rotations = 30
    side_len = 0.7

    pos_mic = np.zeros((1,3))#rng.uniform(-0.5, 0.5, size=(1,3))
    pos_cardioid = np.ones((num_rotations,1)) * pos_mic

    cuboid = region.Cuboid((side_len, side_len, side_len), np.zeros(3), rng=rng)
    pos_omni = cuboid.sample_points(128)
    
    rect = region.Rectangle((side_len, side_len), np.zeros(3), (0.04, 0.04))
    pos_eval = rect.equally_spaced_points()

    mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    mic_dir = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))
    #mic_dir = mc.uniform_random_on_sphere(num_rotations, rng)

    pos_src = np.array([[0,100,0]])

    pos_omni, pos_cardioid, pos_eval, p_omni_sim, p_cardioid_sim, p_eval_sim, freqs, sim_info = _generate_soundfield_data_dir_and_omni(sr, mic_dir, rt60=0, side_len=side_len, pos_omni=pos_omni, pos_cardioid=pos_cardioid, pos_eval=pos_eval, pos_src = pos_src, rng=rng)
    fi = int(p_omni_sim.shape[0] * 0.8)
    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_omni_sim = p_omni_sim[fi:fi+1,...]
    p_eval_sim = p_eval_sim[fi:fi+1,...]

    shd_coeffs = sph.inf_dimensional_shd_omni(p_omni_sim, pos_omni, exp_center, max_order, wave_num, reg_param)
    p_eval_shd = sph.apply_measurement_omni(shd_coeffs, pos_eval, exp_center, wave_num)

    mse_omni_reconstruction = np.mean(np.abs(p_eval_shd - p_eval_sim)**2) / np.mean(np.abs(p_eval_sim)**2)
    plm.image_scatter_freq_response({"omni_sim" : p_eval_sim, "omni_shd" : p_eval_shd}, freq, pos_eval, dot_size = 60)
    plt.show()
    assert mse_omni_reconstruction < 1e-5, f"MSE: {mse_omni_reconstruction}" # Check that the shd coefficients matches the simulated sound field

    # ====== Run the actual test on the cardioid microphone ======
    p_cardioid = {}
    p_cardioid["sim"] = p_cardioid_sim[fi,...]
    
    dir_coeffs_cardioid = sph.directivity_linear(0.5, mic_dir)[None,...]
    p_cardioid["shd"] = sph.apply_measurement(shd_coeffs, pos_cardioid, exp_center, wave_num, dir_coeffs_cardioid)[0,...]

    mse_cardioid = np.mean(np.abs(p_cardioid["shd"] - p_cardioid["sim"])**2) / np.mean(np.abs(p_cardioid["sim"])**2)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    for name, val in p_cardioid.items():
        axes[0].plot(mic_angle, np.real(val), label=name)
        axes[1].plot(mic_angle, np.imag(val), label=name)
        axes[2].plot(mic_angle, np.abs(val), label=name)
    axes[0].set_title("Real part")
    axes[1].set_title("Imaginary part")
    axes[2].set_title("Magnitude")
    axes[0].legend()
    for ax in axes:
        aspplot.set_basic_plot_look(ax)
        ylim = ax.get_ylim()
        ylim_max = np.max(np.abs(ylim))
        ax.set_ylim(-ylim_max, ylim_max)
    fig.suptitle(f"Total MSE: {mse_cardioid}")
    plt.show()

    assert mse_cardioid < 1e-5, f"MSE: {mse_cardioid}"



def test_measurement_model_with_cardioid_microphone_as_a_function_of_the_distance_to_the_source():
    """Tests have indicated that the measurement model works well for sound fields that are very close to
    plane waves, i.e. the source is very far away. But then the simulated and shd model becomes more different
    as the source comes closer and the sound field becomes less plane wave like. 
    """
    rng = np.random.default_rng(123456)
    sr = 1000
    max_order = 16
    exp_center = np.zeros((1,3))
    reg_param = 1e-8
    num_rotations = 60
    side_len = 0.8

    source_distances = np.array([1, 3, 10, 30, 100])
    num_src = source_distances.shape[0]
    source_direction = np.array([[1,0,0]])
    pos_src = source_distances[:,None] * source_direction

    pos_mic = np.zeros((1,3))#rng.uniform(-0.5, 0.5, size=(1,3))
    pos_cardioid = np.ones((num_rotations,1)) * pos_mic

    cuboid = region.Cuboid((side_len, side_len, side_len), np.zeros(3), rng=rng)
    pos_omni = cuboid.sample_points(128)
    
    rect = region.Rectangle((side_len, side_len), np.zeros(3), (side_len/10, side_len/10))
    pos_eval = rect.equally_spaced_points()

    mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    mic_dir = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))
    #mic_dir = mc.uniform_random_on_sphere(num_rotations, rng)

    pos_omni, pos_cardioid, pos_eval, p_omni_sim, p_cardioid_sim, p_eval_sim, freqs, sim_info = _generate_soundfield_data_dir_and_omni(sr, mic_dir, rt60=0, side_len=side_len, pos_omni=pos_omni, pos_cardioid=pos_cardioid, pos_eval=pos_eval, rng=rng, pos_src=pos_src)
    fi = int(p_omni_sim.shape[0] * 0.8)
    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_omni_sim = p_omni_sim[fi:fi+1,...]
    p_eval_sim = p_eval_sim[fi:fi+1,...]

    shd_coeffs = [sph.inf_dimensional_shd_omni(p_omni_sim[...,i], pos_omni, exp_center, max_order, wave_num, reg_param) for i in range(num_src)]
    p_eval_shd = np.stack([sph.apply_measurement_omni(sc, pos_eval, exp_center, wave_num) for sc in shd_coeffs], axis=-1)

    mse_omni_reconstruction = np.mean(np.abs(p_eval_shd - p_eval_sim)**2) / np.mean(np.abs(p_eval_sim)**2)
    p_all = {f"omni_sim_{i}" : p_eval_sim[...,i] for i in range(num_src)}
    p_all.update({f"omni_shd_{i}" : p_eval_shd[...,i] for i in range(num_src)})
    plm.image_scatter_freq_response(p_all, freq, pos_eval, dot_size = 60)
    plt.show()
    assert mse_omni_reconstruction < 1e-5, f"MSE: {mse_omni_reconstruction}" # Check that the shd coefficients matches the simulated sound field

    # ====== Run the actual test on the cardioid microphone ======
    p_cardioid = {}
    p_cardioid["sim"] = p_cardioid_sim[fi,...]
    
    dir_coeffs_cardioid = sph.directivity_linear(0.5, mic_dir)[None,...]
    p_cardioid["shd"] = np.stack([sph.apply_measurement(sc, pos_cardioid, exp_center, wave_num, dir_coeffs_cardioid) for sc in shd_coeffs], axis=-1)[0,...]

    mse_cardioid = [np.mean(np.abs(p_cardioid["shd"][...,i] - p_cardioid["sim"][...,i])**2) / np.mean(np.abs(p_cardioid["sim"][...,i])**2) for i in range(num_src)]
    
    fig, axes = plt.subplots(num_src, 3, figsize=(10, 10))
    for i in range(num_src):
        for name, val in p_cardioid.items():
            axes[i,0].plot(mic_angle, np.real(val[...,i]), label=name)
            axes[i,1].plot(mic_angle, np.imag(val[...,i]), label=name)
            axes[i,2].plot(mic_angle, np.abs(val[...,i]), label=name)
        axes[i,0].set_title("Real part")
        axes[i,1].set_title("Imaginary part")
        axes[i,2].set_title("Magnitude")
        axes[i,0].legend()
    for ax_row in axes:
        for ax in ax_row:
            aspplot.set_basic_plot_look(ax)
            ylim = ax.get_ylim()
            ylim_max = np.max(np.abs(ylim))
            ax.set_ylim(-ylim_max, ylim_max)
    fig.suptitle(f"Cardioid response for source distance {source_distances} m")

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(source_distances, 10 * np.log10(mse_cardioid))
    ax.set_xlabel("Distance to source")
    ax.set_ylabel("MSE (dB)")
    aspplot.set_basic_plot_look(ax)

    fig, axes = plt.subplots(2, 1, figsize=(8,6))
    for i in range(num_src):
        axes[0].plot(mic_angle, np.abs(p_cardioid["sim"][...,i]) / np.linalg.norm(p_cardioid["sim"][...,i]), label=f"distance_{source_distances[i]}")
        axes[1].plot(mic_angle, np.abs(p_cardioid["shd"][...,i]) / np.linalg.norm(p_cardioid["shd"][...,i]) , label=f"distance_{source_distances[i]}")
    axes[0].set_title("Simulated")
    axes[1].set_title("SHD")
    for ax in axes:
        ax.legend()
        aspplot.set_basic_plot_look(ax)
    fig.suptitle("Magnitude of cardioid response")

    plt.show()

    assert np.all([mse_c < 1e-5 for mse_c in mse_cardioid]), f"MSE: {mse_cardioid}"


def test_sound_pressure_estimates_with_cardioid_microphones_as_a_function_of_the_distance_to_the_source():
    """Tests have indicated that the measurement model works well for sound fields that are very close to
    plane waves, i.e. the source is very far away. But then the simulated and shd model becomes more different
    as the source comes closer and the sound field becomes less plane wave like. 
    """
    rng = np.random.default_rng(123456)
    sr = 1000
    max_order = 16
    exp_center = np.zeros((1,3))
    reg_param = 1e-8
    #num_rotations = 60
    side_len = 0.8

    source_distances = np.array([1, 3, 10, 30, 100])
    num_src = source_distances.shape[0]
    source_direction = np.array([[1,0,0]])
    pos_src = source_distances[:,None] * source_direction

    #pos_mic = np.zeros((1,3))#rng.uniform(-0.5, 0.5, size=(1,3))
    #pos_cardioid = np.ones((num_rotations,1)) * pos_mic

    num_mic = 64
    cuboid = region.Cuboid((side_len, side_len, side_len), np.zeros(3), rng=rng)
    pos_omni = cuboid.sample_points(num_mic)
    pos_cardioid = np.copy(pos_omni)
    
    rect = region.Rectangle((side_len, side_len), np.zeros(3), (side_len/20, side_len/20))
    pos_eval = rect.equally_spaced_points()

    #mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    #mic_dir = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))
    mic_dir = mc.uniform_random_on_sphere(num_mic, rng)
    dir_coeffs = sph.directivity_linear(0.5, mic_dir)[None,...]

    pos_omni, pos_cardioid, pos_eval, p_omni_sim, p_cardioid_sim, p_eval_sim, freqs, sim_info = _generate_soundfield_data_dir_and_omni(sr, mic_dir, rt60=0, side_len=side_len, pos_omni=pos_omni, pos_cardioid=pos_cardioid, pos_eval=pos_eval, rng=rng, pos_src=pos_src)
    fi = int(p_omni_sim.shape[0] * 0.8)
    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_omni_sim = p_omni_sim[fi:fi+1,...]
    p_cardioid_sim = p_cardioid_sim[fi:fi+1,...]
    p_eval_sim = p_eval_sim[fi:fi+1,...]

    shd_coeffs_omni = [sph.inf_dimensional_shd_omni(p_omni_sim[...,i], pos_omni, exp_center, max_order, wave_num, reg_param) for i in range(num_src)]
    shd_coeffs_cardioid = [sph.inf_dimensional_shd(p_cardioid_sim[...,i], pos_cardioid, exp_center, max_order, wave_num, reg_param, dir_coeffs) for i in range(num_src)]
    p_eval_omni = np.stack([sph.apply_measurement_omni(sc, pos_eval, exp_center, wave_num) for sc in shd_coeffs_omni], axis=-1)
    p_eval_cardioid = np.stack([sph.apply_measurement_omni(sc, pos_eval, exp_center, wave_num) for sc in shd_coeffs_cardioid], axis=-1)

    # mse_omni_reconstruction = np.mean(np.abs(p_eval_omni - p_eval_sim)**2) / np.mean(np.abs(p_eval_sim)**2)
    # p_all = {f"omni_sim_{i}" : p_eval_sim[...,i] for i in range(num_src)}
    # p_all.update({f"omni_shd_{i}" : p_eval_omni[...,i] for i in range(num_src)})
    # 
    # assert mse_omni_reconstruction < 1e-5, f"MSE: {mse_omni_reconstruction}" # Check that the omni estimates are good enough
    mse_omni = [np.mean(np.abs(p_eval_omni[...,i] - p_eval_sim[...,i])**2) / np.mean(np.abs(p_eval_sim[...,i])**2) for i in range(num_src)]
    mse_cardioid = [np.mean(np.abs(p_eval_cardioid[...,i] - p_eval_sim[...,i])**2) / np.mean(np.abs(p_eval_sim[...,i])**2) for i in range(num_src)]

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(source_distances, 10 * np.log10(mse_omni), "-o", label="omni") 
    ax.plot(source_distances, 10 * np.log10(mse_cardioid), "-x", label="cardioid")
    ax.set_xlabel("Distance to source (m)")
    ax.set_ylabel("MSE (dB)")
    aspplot.set_basic_plot_look(ax)

    plm.image_scatter_freq_response({"omni close" : p_eval_omni[...,0], "cardioid close" : p_eval_cardioid[...,0], "sim close" : p_eval_sim[...,0]}, freq, pos_eval, dot_size = 60)
    plm.image_scatter_freq_response({"omni far" : p_eval_omni[...,-1], "cardioid far" : p_eval_cardioid[...,-1], "sim far" : p_eval_sim[...,-1]}, freq, pos_eval, dot_size = 60)
    # plt.show()

    plt.show()

    assert np.all([mse_c < 1e-5 for mse_c in mse_cardioid]), f"MSE: {mse_cardioid}"









def test_measurement_model_with_differential_cardioid_microphones_as_a_function_of_the_distance_to_the_source():
    """Tests have indicated that the measurement model works well for sound fields that are very close to
    plane waves, i.e. the source is very far away. But then the simulated and shd model becomes more different
    as the source comes closer and the sound field becomes less plane wave like. 
    """
    rng = np.random.default_rng(123456)
    sr = 4000
    max_order = 16
    exp_center = np.zeros((1,3))
    reg_param = 1e-8
    num_rotations = 60
    side_len = 0.4

    source_distances = np.array([1, 3, 10, 30, 100])
    num_src = source_distances.shape[0]
    source_direction = np.array([[0,1,0]])
    pos_src = source_distances[:,None] * source_direction

    pos_mic = np.zeros((1,3))#rng.uniform(-0.5, 0.5, size=(1,3))
    pos_cardioid = np.ones((num_rotations,1)) * pos_mic

    cuboid = region.Cuboid((side_len, side_len, side_len), np.zeros(3), rng=rng)
    pos_omni = cuboid.sample_points(256)
    
    rect = region.Rectangle((side_len, side_len), np.zeros(3), (side_len/10, side_len/10))
    pos_eval = rect.equally_spaced_points()

    mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    mic_dir = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))
    #mic_dir = mc.uniform_random_on_sphere(num_rotations, rng)

    pos_omni, pos_cardioid, pos_eval, p_omni_sim, p_cardioid_sim, p_eval_sim, freqs, sim_info = _generate_soundfield_differential_cardioids(sr, mic_dir, rt60=0, side_len=side_len, pos_omni=pos_omni, pos_cardioid=pos_cardioid, pos_eval=pos_eval, rng=rng, pos_src=pos_src)
    fi = int(p_omni_sim.shape[0] * 0.2)
    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_omni_sim = p_omni_sim[fi:fi+1,...]
    p_eval_sim = p_eval_sim[fi:fi+1,...]

    shd_coeffs = [sph.inf_dimensional_shd_omni(p_omni_sim[...,i], pos_omni, exp_center, max_order, wave_num, reg_param) for i in range(num_src)]
    p_eval_shd = np.stack([sph.apply_measurement_omni(sc, pos_eval, exp_center, wave_num) for sc in shd_coeffs], axis=-1)

    mse_omni_reconstruction = np.mean(np.abs(p_eval_shd - p_eval_sim)**2) / np.mean(np.abs(p_eval_sim)**2)
    p_all = {f"omni_sim_{i}" : p_eval_sim[...,i] for i in range(num_src)}
    p_all.update({f"omni_shd_{i}" : p_eval_shd[...,i] for i in range(num_src)})
    #plm.image_scatter_freq_response(p_all, freq, pos_eval, dot_size = 60)
    #plt.show()
    assert mse_omni_reconstruction < 1e-5, f"MSE: {mse_omni_reconstruction}" # Check that the shd coefficients matches the simulated sound field

    # ====== Run the actual test on the cardioid microphone ======
    p_cardioid = {}
    p_cardioid["sim"] = p_cardioid_sim[fi,...]
    
    dir_coeffs_cardioid = sph.directivity_linear(0.5, mic_dir)[None,...]
    p_cardioid["shd"] = np.stack([sph.apply_measurement(sc, pos_cardioid, exp_center, wave_num, dir_coeffs_cardioid) for sc in shd_coeffs], axis=-1)[0,...]

    mse_cardioid = [np.mean(np.abs(p_cardioid["shd"][...,i] - p_cardioid["sim"][...,i])**2) / np.mean(np.abs(p_cardioid["sim"][...,i])**2) for i in range(num_src)]
    
    fig, axes = plt.subplots(num_src, 3, figsize=(10, 10))
    for i in range(num_src):
        for name, val in p_cardioid.items():
            axes[i,0].plot(mic_angle, np.real(val[...,i]), label=name)
            axes[i,1].plot(mic_angle, np.imag(val[...,i]), label=name)
            axes[i,2].plot(mic_angle, np.abs(val[...,i]), label=name)
        axes[i,0].set_title("Real part")
        axes[i,1].set_title("Imaginary part")
        axes[i,2].set_title("Magnitude")
        axes[i,0].legend()
    for ax_row in axes:
        for ax in ax_row:
            aspplot.set_basic_plot_look(ax)
            ylim = ax.get_ylim()
            ylim_max = np.max(np.abs(ylim))
            ax.set_ylim(-ylim_max, ylim_max)
    fig.suptitle(f"Cardioid response for source distance {source_distances} m")

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(source_distances, 10 * np.log10(mse_cardioid))
    ax.set_xlabel("Distance to source")
    ax.set_ylabel("MSE (dB)")
    aspplot.set_basic_plot_look(ax)

    fig, axes = plt.subplots(2, 1, figsize=(8,6))
    for i in range(num_src):
        axes[0].plot(mic_angle, np.abs(p_cardioid["sim"][...,i]) / np.linalg.norm(p_cardioid["sim"][...,i]), label=f"distance_{source_distances[i]}")
        axes[1].plot(mic_angle, np.abs(p_cardioid["shd"][...,i]) / np.linalg.norm(p_cardioid["shd"][...,i]) , label=f"distance_{source_distances[i]}")
    axes[0].set_title("Simulated")
    axes[1].set_title("SHD")
    for ax in axes:
        ax.legend()
        aspplot.set_basic_plot_look(ax)
    fig.suptitle("Magnitude of cardioid response")

    plt.show()

    assert np.all([mse_c < 1e-5 for mse_c in mse_cardioid]), f"MSE: {mse_cardioid}"













def test_measurement_model_with_differential_cardioid_microphones_calculated_in_time_domain_equivalent_to_ideal_model_for_plane_waves():
    """
    """
    rng = np.random.default_rng(123456)
    sr = 1000
    exp_center = np.zeros((1,3))
    num_rotations = 60
    side_len = 0.4
    c = 343

    num_freqs = 2 * sr

    wave_direction = np.array([[1,0,0]])
    wave_num = ft.get_real_wavenum(num_freqs, sr, c)

    pos_mic = np.array([[0.1,0.1,0.1]])#np.zeros((1,3))#rng.uniform(-0.5, 0.5, size=(1,3))
    pos_cardioid = np.ones((num_rotations,1)) * pos_mic
    
    #rect = region.Rectangle((side_len, side_len), np.zeros(3), (side_len/10, side_len/10))
    #pos_eval = rect.equally_spaced_points()

    mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    mic_direction = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))
    #mic_dir = mc.uniform_random_on_sphere(num_rotations, rng)

    differential_distance = 1e-2


    pos_cardioid = pos_cardioid #+ differential_distance * mic_direction / 2
    pos_cardioid_diff = pos_cardioid - differential_distance * mic_direction

    p_mic = np.squeeze(pw.plane_wave(pos_cardioid, wave_direction, wave_num, exp_center), axis=-1)
    p_diff = np.squeeze(pw.plane_wave(pos_cardioid_diff, wave_direction, wave_num, exp_center), axis=-1)

    sig_mic = ft.irfft(p_mic)
    sig_diff = ft.irfft(p_diff)

    filter_order = num_freqs 
    sig_diff_cardioid = fd.differential_cardioid_microphone(sig_mic, sig_diff, differential_distance, filter_order, c, sr)

    dir_func = pw.linear_directivity_function(0.5, mic_direction)
    cardioid_factor = np.real_if_close(dir_func(wave_direction))
    sig_pw_cardioid = cardioid_factor[:,None] * sig_mic

    num_examples = 5
    fig, axes = plt.subplots(num_examples, 1, figsize=(8,10))
    for i in range(num_examples):
        axes[i].plot(sig_diff_cardioid[i,:], label="Differential")
        axes[i].plot(sig_pw_cardioid[i,:], label="True")
        axes[i].legend()
        aspplot.set_basic_plot_look(axes[i])

    freq_diff_cardioid = ft.rfft(sig_diff_cardioid)
    freq_pw_cardioid = ft.rfft(sig_pw_cardioid)

    num_examples = 3
    fig, axes = plt.subplots(num_examples, 3, figsize=(14,10))
    for i in range(num_examples):
        axes[i,0].plot(np.real(freq_diff_cardioid[:,i]), label="Differential")
        axes[i,0].plot(np.real(freq_pw_cardioid[:,i]), label="True")
        axes[i,1].plot(20 * np.log10(np.abs(freq_diff_cardioid[:,i])), label="Differential")
        axes[i,1].plot(20 * np.log10(np.abs(freq_pw_cardioid[:,i])), label="True")

        mse = np.abs(freq_diff_cardioid[:,i] - freq_pw_cardioid[:,i])**2 / np.mean(np.abs(freq_pw_cardioid[:,i])**2)
        axes[i,2].plot(10 * np.log10(mse), label="MSE")

    axes[0,0].set_title("Real part")
    axes[0,1].set_title("Magnitude (dB)")
    axes[0,2].set_title("Mean square error (dB)")

    for ax_row in axes:
        for ax in ax_row:
            aspplot.set_basic_plot_look(ax)
            ax.legend()

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    mse = np.mean(np.abs(freq_diff_cardioid - freq_pw_cardioid)**2, axis=-1) / np.mean(np.abs(freq_pw_cardioid)**2, axis=-1)
    ax.plot(10 * np.log10(mse))
    aspplot.set_basic_plot_look(ax)
    ax.set_title("Mean square error")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("MSE (dB)")

    plt.show()









def test_measurement_model_with_differential_cardioid_microphones_calculated_in_time_domain_equivalent_to_pyroomacoustics_for_far_away_sources(fig_folder):
    """
    """
    rng = np.random.default_rng(123456)
    sr = 2000
    num_rotations = 20
    c = 343

    pos_src = np.array([[0,100,0]])
    pos_mic = np.array([[0.1,0.1,0.1]])#np.zeros((1,3))#rng.uniform(-0.5, 0.5, size=(1,3))
    pos_cardioid = np.ones((num_rotations,1)) * pos_mic
    

    mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    mic_direction = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))
    #mic_dir = mc.uniform_random_on_sphere(num_rotations, rng)

    differential_distance = 1e-3
    pos_cardioid = pos_cardioid #+ differential_distance * mic_direction / 2
    pos_diff = pos_cardioid - differential_distance * mic_direction


    setup = _get_default_simulator_setup(sr, fig_folder)
    setup.sim_info.start_sources_before_0 = False
    setup.sim_info.rt60 = 0
    setup.sim_info.max_room_ir_length = sr
    setup.sim_info.tot_samples = 16 * setup.sim_info.max_room_ir_length
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.room_size = [1000, 1000, 1000]
    setup.sim_info.extra_delay = 128

    seq_len = setup.sim_info.max_room_ir_length
    sequence = pseq.create_pseq(seq_len)
    sequence_src = sources.Sequence(1e3 * sequence)

    setup.add_mics("omni", pos_mic)
    setup.add_mics("cardioid", pos_cardioid, directivity_type=num_rotations*["cardioid"], directivity_dir=mic_direction)
    setup.add_mics("cardioid_diff", pos_diff)
    setup.add_free_source("src", pos_src, sequence_src)

    sim = setup.create_simulator()
    for mic_array in sim.arrays.mics():
        sim.diag.add_diagnostic(mic_array.name, dg.RecordSignal(mic_array.name, sim.sim_info, num_channels=mic_array.num, export_func="npz"))

    sim.run_simulation()

    sig_omni = np.load(sim.folder_path / f"omni_{setup.sim_info.tot_samples}.npz")["omni"]
    sig_cardioid = np.load(sim.folder_path / f"cardioid_{setup.sim_info.tot_samples}.npz")["cardioid"]
    sig_diff = np.load(sim.folder_path / f"cardioid_diff_{setup.sim_info.tot_samples}.npz")["cardioid_diff"]

    sig_omni = sig_omni[:,seq_len:]
    sig_cardioid = sig_cardioid[:,seq_len:]
    sig_diff = sig_diff[:,seq_len:]    
    
    filter_order = 3*seq_len
    filter_freq = 50
    sig_diff_cardioid = fd.differential_cardioid_microphone(sig_omni, sig_diff, differential_distance, filter_order, c, sr)

    highpass_order = seq_len

    #highpass_ir = spsig.firls(2 * highpass_order + 1, np.array([[0, filter_freq / 2], [filter_freq, sr/2]]), np.array([[0, 0], [1, 1]]), fs=sr)
    #spsig.butter(highpass_order, filter_freq, btype="high", fs=sr, output="sos")

    sig_cardioid = _highpass_microphone_signal(sig_cardioid, filter_order, sr, filter_freq)

    sig_cardioid = sig_cardioid[:,2*filter_order:-2*filter_order]
    sig_diff_cardioid = sig_diff_cardioid[:,2*filter_order:-2*filter_order]

    num_examples = 5
    fig, axes = plt.subplots(num_examples, 1, figsize=(8,10))
    for i in range(num_examples):
        axes[i].plot(sig_diff_cardioid[i,:], label="Differential")
        axes[i].plot(sig_cardioid[i,:], label="True")
        axes[i].legend()
        aspplot.set_basic_plot_look(axes[i])

    fig, axes = plt.subplots(num_examples, 2, figsize=(8,10))
    for i in range(num_examples):
        axes[i,0].plot(sim.arrays.paths["src"]["cardioid_diff"][0,i,:], label="Differential")
        axes[i,0].plot(sim.arrays.paths["src"]["cardioid"][0,i,:], label="Cardioid")
        
        axes[i,1].plot(20*np.log10(np.abs(sim.arrays.paths["src"]["cardioid_diff"][0,i,:])), label="Differential")
        axes[i,1].plot(20 * np.log10(np.abs(sim.arrays.paths["src"]["cardioid"][0,i,:])), label="Cardioid")

    for ax_row in axes:
        for ax in ax_row:
            ax.legend()
            aspplot.set_basic_plot_look(ax)
    axes[0,0].set_title("Room impulse responses")
    axes[0,1].set_title("Room impulse responses (dB)")

    #freq_diff_cardioid = ft.rfft(sig_diff_cardioid[:,2*filter_order:2*filter_order + seq_len])
    #freq_cardioid = ft.rfft(sig_cardioid[:,2*filter_order:2*filter_order + seq_len])

    freq_diff_cardioid = ft.rfft(sig_diff_cardioid[:,:seq_len])
    freq_cardioid = ft.rfft(sig_cardioid[:,:seq_len])

    num_examples = 3
    fig, axes = plt.subplots(num_examples, 3, figsize=(14,10))
    for i in range(num_examples):
        axes[i,0].plot(np.real(freq_diff_cardioid[:,i]), label="Differential")
        axes[i,0].plot(np.real(freq_cardioid[:,i]), label="True")
        axes[i,1].plot(20 * np.log10(np.abs(freq_diff_cardioid[:,i])), label="Differential")
        axes[i,1].plot(20 * np.log10(np.abs(freq_cardioid[:,i])), label="True")

        mse = np.abs(freq_diff_cardioid[:,i] - freq_cardioid[:,i])**2 / np.mean(np.abs(freq_cardioid[:,i])**2)
        axes[i,2].plot(10 * np.log10(mse), label="MSE")

    axes[0,0].set_title("Real part")
    axes[0,1].set_title("Magnitude (dB)")
    axes[0,2].set_title("Mean square error (dB)")

    for ax_row in axes:
        for ax in ax_row:
            aspplot.set_basic_plot_look(ax)
            ax.legend()

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    mse = np.mean(np.abs(freq_diff_cardioid - freq_cardioid)**2, axis=-1) / np.mean(np.abs(freq_cardioid)**2, axis=-1)
    ax.plot(10 * np.log10(mse))
    aspplot.set_basic_plot_look(ax)
    ax.set_title("Mean square error")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("MSE (dB)")

    plt.show()



def _highpass_microphone_signal(mic_sig, filter_order, samplerate, filter_below):
    ir_len = 2*filter_order + 1

    def freq_resp_func(f):
        filter_response = np.ones_like(f)
        num_freqs_to_filter = np.sum(f <= filter_below)
        scaling = np.linspace(0, 1, num_freqs_to_filter) ** 4
        filter_response[:num_freqs_to_filter] *= scaling
        return filter_response
    
    ir = fd.fir_from_frequency_function(freq_resp_func, ir_len, samplerate, window="hamming")

    signal_filtered = spsig.fftconvolve(ir[None,:], mic_sig, axes=-1, mode="full")

    signal_filtered = signal_filtered[...,filter_order:filter_order+mic_sig.shape[-1]]
    return signal_filtered








def test_measurement_model_with_differential_cardioid_microphones_calculated_in_time_domain_equivalent_to_spherical_harmonic_model(fig_folder):
    """
    """
    rng = np.random.default_rng(123456)
    sr = 2000
    num_rotations = 20
    c = 343

    num_shd_mics = 256

    pos_src = np.array([[3,0,0]])
    pos_mic = np.zeros((1,3)) #np.array([[0.1,0.1,0.1]])#np.zeros((1,3))#rng.uniform(-0.5, 0.5, size=(1,3))
    pos_shd_mics = rng.uniform(-0.3, 0.3, size=(num_shd_mics,3))
    
    mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    mic_direction = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))

    differential_distance = 3e-3 #1e-2 #1e-3
    pos_diff = pos_mic - differential_distance * mic_direction

    setup = _get_default_simulator_setup(sr, fig_folder)
    setup.sim_info.start_sources_before_0 = False
    setup.sim_info.rt60 = 0.31
    setup.sim_info.max_room_ir_length = int((sr // 2))
    setup.sim_info.tot_samples = 32 * setup.sim_info.max_room_ir_length
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.room_size = [5.4, 3.2, 2.1] #[4, 5,3]
    setup.sim_info.room_center = [1.2, 0.1, 0.1]
    setup.sim_info.extra_delay = 128 #40 #128

    seq_len = setup.sim_info.max_room_ir_length
    sequence = pseq.create_pseq(seq_len)
    sequence_src = sources.Sequence(sequence)

    setup.add_mics("shd_mics", pos_shd_mics)
    setup.add_mics("omni", pos_mic)
    setup.add_mics("diff", pos_diff)
    setup.add_free_source("src", pos_src, sequence_src)

    sim = setup.create_simulator()
    for mic_array in sim.arrays.mics():
        sim.diag.add_diagnostic(mic_array.name, dg.RecordSignal(mic_array.name, sim.sim_info, num_channels=mic_array.num, export_func="npz"))

    sim.run_simulation()

    sig_omni = np.load(sim.folder_path / f"omni_{setup.sim_info.tot_samples}.npz")["omni"]
    sig_diff = np.load(sim.folder_path / f"diff_{setup.sim_info.tot_samples}.npz")["diff"]
    sig_shd_mics = np.load(sim.folder_path / f"shd_mics_{setup.sim_info.tot_samples}.npz")["shd_mics"]

    sig_omni = sig_omni[:,seq_len:]
    sig_diff = sig_diff[:,seq_len:]
    sig_shd_mics = sig_shd_mics[:,seq_len:]
    

    # CONSTRUCT SHD RESPONSE
    max_order = 20
    reg_param = 1e-8
    exp_center = np.zeros((1,3))
    wave_num = ft.get_real_wavenum(seq_len, sr, c)
    rir_freq_shd_mics = ft.rfft(sim.arrays.paths["src"]["shd_mics"][0,...])
    shd_coeffs = sph.inf_dimensional_shd_omni(rir_freq_shd_mics, pos_shd_mics, exp_center, max_order, wave_num, reg_param)
    reconstructed_pressure = sph.apply_measurement_omni(shd_coeffs, pos_shd_mics, exp_center, wave_num)
    reconstruction_error = np.mean(np.abs(reconstructed_pressure - rir_freq_shd_mics)**2) / np.mean(np.abs(rir_freq_shd_mics)**2)
    assert reconstruction_error < 1e-5, f"Reconstruction error: {reconstruction_error}"

    dir_coeffs = sph.directivity_linear(0.5, mic_direction)[None,...]
    rir_freq_shd_cardioid = np.concatenate([sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, dir_coeffs[:,i:i+1,:]) for i in range(num_rotations)], axis=-1)
    rir_shd_cardioid = ft.irfft(rir_freq_shd_cardioid)

    # CONSTRUCT DIFFERENTIAL RESPONSE
    filter_order = 5*seq_len
    filter_freq = 50
    sig_diff_cardioid = fd.differential_cardioid_microphone(sig_omni, sig_diff, differential_distance, filter_order, c, sr)
    sig_diff_cardioid = sig_diff_cardioid[:,2*filter_order:-2*filter_order]
    sig_diff_cardioid = sig_diff_cardioid[:,:seq_len]
    rir_diff_cardioid = pseq.decorrelate(sig_diff_cardioid, sequence)


    num_examples = 5
    fig, axes = plt.subplots(num_examples, 1, figsize=(8,10))
    for i in range(num_examples):
        axes[i].plot(sig_diff_cardioid[i,:], label="Differential")
        #axes[i].plot(sig_cardioid[i,:], label="True")
        axes[i].legend()
        aspplot.set_basic_plot_look(axes[i])

    fig, axes = plt.subplots(num_examples, 2, figsize=(8,10))
    for i in range(num_examples):
        axes[i,0].plot(rir_diff_cardioid[i,:], label="Differential")
        axes[i,0].plot(rir_shd_cardioid[i,:], label="SHD")
        
        axes[i,1].plot(20 * np.log10(np.abs(rir_diff_cardioid[i,:])), label="Differential")
        axes[i,1].plot(20 * np.log10(np.abs(rir_shd_cardioid[i,:])), label="SHD")

    for ax_row in axes:
        for ax in ax_row:
            ax.legend()
            aspplot.set_basic_plot_look(ax)
    axes[0,0].set_title("Room impulse responses")
    axes[0,1].set_title("Room impulse responses (dB)")

    #freq_diff_cardioid = ft.rfft(sig_diff_cardioid[:,2*filter_order:2*filter_order + seq_len])
    #freq_cardioid = ft.rfft(sig_cardioid[:,2*filter_order:2*filter_order + seq_len])

    freq_diff_cardioid = ft.rfft(rir_diff_cardioid)
    freq_shd_cardioid = ft.rfft(rir_shd_cardioid)

    num_examples = 3
    fig, axes = plt.subplots(num_examples, 3, figsize=(14,10))
    for i in range(num_examples):
        axes[i,0].plot(np.real(freq_diff_cardioid[:,i]), label="Differential")
        axes[i,0].plot(np.real(freq_shd_cardioid[:,i]), label="True")
        axes[i,1].plot(20 * np.log10(np.abs(freq_diff_cardioid[:,i])), label="Differential")
        axes[i,1].plot(20 * np.log10(np.abs(freq_shd_cardioid[:,i])), label="True")

        mse = np.abs(freq_diff_cardioid[:,i] - freq_shd_cardioid[:,i])**2 / np.mean(np.abs(freq_shd_cardioid[:,i])**2)
        axes[i,2].plot(10 * np.log10(mse), label="MSE")

    axes[0,0].set_title("Real part")
    axes[0,1].set_title("Magnitude (dB)")
    axes[0,2].set_title("Mean square error (dB)")

    for ax_row in axes:
        for ax in ax_row:
            aspplot.set_basic_plot_look(ax)
            ax.legend()


    fig, ax = plt.subplots(1,1, figsize=(8,6))
    mse = np.mean(np.abs(freq_diff_cardioid - freq_shd_cardioid)**2, axis=-1) / np.mean(np.abs(freq_shd_cardioid)**2, axis=-1)
    freqs = ft.get_real_freqs(seq_len, sr)

    mean_mse = np.mean(mse[freqs > 50])
    ax.plot(10 * np.log10(mse))
    aspplot.set_basic_plot_look(ax)
    ax.set_title(f"Mean square error, mean: {10 * np.log10(mean_mse)}")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("MSE (dB)")

    plt.show()





















def _generate_soundfield_data_omni(sr, exp_center = np.array([[0,0,0]])):
    rng = np.random.default_rng(10)
    side_len = 0.2
    num_mic = 100

    #pos_mic = np.zeros((num_mic, 3))
    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))
    pos_mic += exp_center

    pos_src = np.array([[3,0.05,-0.05]])

    setup = _get_default_simulator_setup(sr)

    setup.add_mics("omni", pos_mic)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = setup.sim_info.max_room_ir_length
    #fpaths, freqs = sim.arrays.get_freq_paths(num_freqs, sr)
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], freqs, sim.sim_info




def _generate_soundfield_data_dir(sr, mic_direction, rt60 = 0.25, dir_pos = np.zeros((1, 3)), side_len = 0.1, num_mic = 30):
    rng = np.random.default_rng(10)

    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))
    pos_src = np.array([[0,1.5,-0.05]])

    setup = _get_default_simulator_setup(sr)
    setup.sim_info.rt60 = rt60

    setup.add_mics("omni", pos_mic)
    setup.add_mics("dir", dir_pos, directivity_type=["cardioid"], directivity_dir=mic_direction)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = sr // 2
    #fpaths, freqs = sim.arrays.get_freq_paths(num_freqs, sr)
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], fpaths["src"]["dir"][...,0], freqs, sim.sim_info



def _generate_soundfield_data_dir_surrounded_by_sources(sr, directivity_dir, num_src, rt60 = 0.25):
    rng = np.random.default_rng(10)
    side_len = 0.4
    num_mic = 30

    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))

    src_angles = np.linspace(0, 2*np.pi, num_src, endpoint=False)
    pos_src = utils.spherical2cart(np.ones((num_src,)), np.stack((src_angles, np.pi/2*np.ones((num_src,))), axis=-1))

    setup = _get_default_simulator_setup(sr)
    setup.sim_info.rt60 = rt60

    setup.add_mics("omni", pos_mic)
    setup.add_mics("dir", np.zeros((1,3)), directivity_type=["cardioid"], directivity_dir=directivity_dir)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 512
    #fpaths, freqs = sim.arrays.get_freq_paths(num_freqs, sr)
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)
    return sim.arrays["omni"].pos, sim.arrays["src"].pos, sim.arrays["dir"].pos, fpaths["src"]["omni"], fpaths["src"]["dir"], freqs, sim.sim_info


def _generate_soundfield_data_dir_and_omni(sr, mic_direction, rt60 = 0.25, side_len = 0.5, pos_omni=None, pos_cardioid=None, pos_eval=None, rng = None, pos_src=None):
    if rng is None:
        rng = np.random.default_rng()

    if pos_omni is None:
        num_omni = 40
        pos_omni = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_omni, 3))
        #pos_mic = np.concatenate((pos_mic, np.zeros((num_mic, 1))), axis=-1)
        #pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))
    num_omni = pos_omni.shape[0]
    if pos_cardioid is None:
        pos_cardioid = np.copy(pos_omni)
    num_cardioid = pos_cardioid.shape[0]
    if pos_eval is None:
        pos_eval = region.Cuboid((side_len, side_len, side_len), (0,0,0), point_spacing=(0.1, 0.1, 0.1)).equally_spaced_points()

    if pos_src is None:
        pos_src = np.array([[0,2,0]])

    setup = _get_default_simulator_setup(sr)
    setup.sim_info.rt60 = rt60
    setup.sim_info.room_size = [1000, 1000, 1000]

    setup.add_mics("omni", pos_omni)
    setup.add_mics("eval", pos_eval)
    dir_type = num_cardioid*["cardioid"]
    if mic_direction.shape[0] == 1:
        mic_direction = np.tile(mic_direction, (num_cardioid,1))
    setup.add_mics("cardioid", pos_cardioid, directivity_type=dir_type, directivity_dir=mic_direction)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = setup.sim_info.max_room_ir_length

    #fpaths, freqs = sim.arrays.get_freq_paths(num_freqs, sr)
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)
    if pos_src.shape[0] == 1:
        fpaths["src"]["omni"] = fpaths["src"]["omni"][...,0]
        fpaths["src"]["cardioid"] = fpaths["src"]["cardioid"][...,0]
        fpaths["src"]["eval"] = fpaths["src"]["eval"][...,0]

    return sim.arrays["omni"].pos, sim.arrays["cardioid"].pos, sim.arrays["eval"].pos, fpaths["src"]["omni"], fpaths["src"]["cardioid"], fpaths["src"]["eval"], freqs, sim.sim_info


def _generate_soundfield_differential_cardioids(sr, mic_direction, rt60 = 0.25, side_len = 0.5, pos_omni=None, pos_cardioid=None, pos_eval=None, rng = None, pos_src=None):
    if rng is None:
        rng = np.random.default_rng()
    if pos_eval is None:
        pos_eval = region.Cuboid((side_len, side_len, side_len), (0,0,0), point_spacing=(0.1, 0.1, 0.1)).equally_spaced_points()
    if pos_src is None:
        pos_src = np.array([[0,2,0]])

    if pos_omni is None:
        num_omni = 40
        pos_omni = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_omni, 3))
        #pos_mic = np.concatenate((pos_mic, np.zeros((num_mic, 1))), axis=-1)
        #pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))
    num_omni = pos_omni.shape[0]

    if pos_cardioid is None:
        pos_cardioid = np.copy(pos_omni)
    differential_distance = 1e-3

    if mic_direction.shape[0] == 1:
        mic_direction = np.tile(mic_direction, (num_cardioid,1))

    pos_cardioid = pos_cardioid #+ differential_distance * mic_direction / 2
    pos_cardioid_diff = pos_cardioid - differential_distance * mic_direction

    num_cardioid = pos_cardioid.shape[0]



    setup = _get_default_simulator_setup(sr)
    setup.sim_info.rt60 = rt60
    setup.sim_info.room_size = [1000, 1000, 1000]

    setup.add_mics("omni", pos_omni)
    setup.add_mics("eval", pos_eval)
    setup.add_mics("cardioid", pos_cardioid)
    setup.add_mics("cardioid_diff", pos_cardioid_diff)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 2*setup.sim_info.max_room_ir_length

    # TIME DOMAIN SOLUTION
    samples_delay = sr * differential_distance / sim.sim_info.c
    filter_order = 512
    delay_filter, int_delay = fd.fractional_delay_filter(samples_delay, filter_order)
    delayed_cardioid = spsig.fftconvolve(delay_filter[None,None,:], sim.arrays.paths["src"]["cardioid_diff"], axes=-1)
    delayed_cardioid = delayed_cardioid[...,int_delay:]
    delayed_cardioid = delayed_cardioid[...,:sim.arrays.paths["src"]["cardioid"].shape[-1]]

    sim.arrays.paths["src"]["cardioid_diff"] = delayed_cardioid

    # filt = fc.create_filter(ir = delay_filter[None,None,:], broadcast_dim=(sim.arrays["cardioid_2"].num), sum_over_input=False)
    # filt.ir[...] = delay_filter[None, None, :]
    # delayed_cardioid = []
    # for src_idx in range(sim.arrays["src"].num):
    #     delayed_cardioid.append(np.convolve(sim.arrays.paths["src"]["cardioid_2"][src_idx, mic_idx,:]))
    # delayed_cardioid = delayed_cardioid[...,int_delay:]

    #fpaths, freqs = sim.arrays.get_freq_paths(num_freqs, sr)
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)
    if pos_src.shape[0] == 1:
        fpaths["src"]["omni"] = fpaths["src"]["omni"][...,0]
        fpaths["src"]["cardioid"] = fpaths["src"]["cardioid"][...,0]
        fpaths["src"]["cardioid_diff"] = fpaths["src"]["cardioid_diff"][...,0]
        fpaths["src"]["eval"] = fpaths["src"]["eval"][...,0]


    angular_freq = 2 * np.pi * freqs
    wave_num = angular_freq / sim.sim_info.c


    # EXACT SOLUTION
    # both_mic_factor = 1 / (1 - np.exp(-1j * 2 * wave_num * differential_distance))
    # mic1_factor = np.conj(both_mic_factor)
    # mic2_factor = - np.conj(both_mic_factor * np.exp(-1j * wave_num * differential_distance))

    # # APPROXIMATE SOLUTION
    # common_factor = 1j / (2 * wave_num * differential_distance)
    # mic1_factor = np.conj(-common_factor)
    # mic2_factor = np.conj(common_factor  * np.exp(-1j * wave_num * differential_distance))

    #cardioid_response = fpaths["src"]["cardioid"] * mic1_factor[:,None,None] + fpaths["src"]["cardioid_2"] * mic2_factor[:,None,None]

    # UNCOMMENT FOR TIME DOMAIN SOLUTION, MUST ALREADY BE DELAYEED
    common_factor = np.conj(1 / (1 - np.exp(-1j * 2 * wave_num * differential_distance)))
    fpaths["src"]["cardioid"] = common_factor[:,None,None] * (fpaths["src"]["cardioid"] - fpaths["src"]["cardioid_2"])

    return sim.arrays["omni"].pos, sim.arrays["cardioid"].pos, sim.arrays["eval"].pos, fpaths["src"]["omni"], fpaths["src"]["cardioid"], fpaths["src"]["eval"], freqs, sim.sim_info




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


def _get_default_simulator_setup(sr, folder = None):
    if folder is not None:
        output_method = "pdf"
    else:
        output_method = "none"
    setup = SimulatorSetup(folder)
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 =  0.25
    setup.sim_info.max_room_ir_length = sr
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = sr // 8
    setup.sim_info.plot_output = output_method
    return setup