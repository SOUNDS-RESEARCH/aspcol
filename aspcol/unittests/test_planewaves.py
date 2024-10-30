import numpy as np

import aspcore.montecarlo as mc
import aspcore.fouriertransform as ft

import aspcore.utilities as coreutils

import aspcol.sphericalharmonics as sph
import aspcol.utilities as utils
import aspcol.unittests.plot_methods as plm
import aspcol.planewaves as pw

from aspsim.simulator import SimulatorSetup
import aspsim.room.region as region

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
    int_value = pw.plane_wave_integral(pw_coeff_func, pos_eval, exp_center, wave_num, rng, int(1e8))
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
    the delta function requires many samples to give a decent approximation of the value. 
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
        mic_val = pw.apply_measurement(dirac_func, directionality, rng, int(1e8))

        directionality_val = directionality(pw_angle)
        error[i] = np.mean(np.abs(mic_val - directionality_val))

    assert np.mean(error) < 1e-2




def tst_planewave_microphone_model_and_spherical_harmonics_are_identical_for_plane_wave_soundfield():
    """Don't remember what this is supposed to do, so could probably be removed. 
    
    """
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
    p_pw = pw.apply_measurement(pw_coeffs, dir_func, rng, int(1e8))

    pass









# ============= COMPARE MODEL WITH IMAGE SOURCE MODEL =================
def tst_measurement_model_for_analytic_plane_wave_gives_same_value_as_simulated_plane_wave():
    """
    Does not really test anything different than 
    test_measurement_model_with_cardioid_for_analytic_plane_wave_gives_same_value_as_simulated_plane_wave
    so it could probably be removed soon. 
    
    """
    rng = np.random.default_rng()
    sr = 1000
    side_len = 1
    max_order = 16
    
    pw_dir = mc.uniform_random_on_sphere(1, rng)
    mic_dir = mc.uniform_random_on_sphere(1, rng)

    rect = region.Rectangle((side_len, side_len), np.zeros(3), (0.04, 0.04))
    pos_mic = rect.equally_spaced_points()

    p = {}

    pos_mic, p_mic_sim, ir_mic, pos_cardioid, p_cardioid_sim, freqs, sim_info = _generate_plane_wave_with_ism(sr, pw_dir, pos_mic, pos_mic, mic_dir)
    fi = int(p_mic_sim.shape[0] * 0.8)
    p_mic_sim = p_mic_sim[fi,...]
    p_cardioid_sim = p_cardioid_sim[fi,...]

    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_mic_analytic, exp_center, gain_adjustment = pw.find_matching_plane_wave(p_mic_sim, freq, pw_dir, pos_mic, sim_info.c)
    shd_coeffs = gain_adjustment * pw.shd_coeffs_for_planewave(pw_dir, max_order)

    #dir_coeffs_omni = sph.directivity_linear(0, pw_dir)
    #p_omni_shd = sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, dir_coeffs_omni)

    dir_coeffs_cardioid = sph.directivity_linear(0.5, mic_dir)
    p["cardioid_shd"] = sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, dir_coeffs_cardioid)

    p["cardioid_sim"] = p_cardioid_sim


    #p_dir_shd = sph.apply_measurement(shd_coeffs, pos_mic, exp_center, wave_num, dir_coeffs)

    p["omni_sim"] = p_mic_sim
    plm.image_scatter_freq_response(p, freq, pos_mic, dot_size = 60)
    plt.show()

    plm.image_scatter_freq_response({"difference" : p["cardioid_shd"] - p["cardioid_sim"]}, freq, pos_mic, dot_size = 60)
    plt.show()


    #assert np.mean(np.abs(p_dir_shd - p_mic_sim)**2) / np.mean(np.abs(p_mic_sim)**2) < 1e-5


def test_measurement_model_with_cardioid_for_analytic_plane_wave_gives_same_value_as_simulated_plane_wave():
    """
    
    As the test test_matched_analytic_plane_wave_is_very_similar_to_simulated_plane_wave shows, the data
    from the simulator and the analytic representation only matches up to an MSE of around 1e-6. Therefore the
    directional measurement model will not match better than that. 
    """
    rng = np.random.default_rng(123456)
    sr = 1000
    max_order = 24
    num_rotations = 30
    
    pw_dir = mc.uniform_random_on_sphere(1, rng)

    pos_mic = rng.uniform(-0.5, 0.5, size=(1,3)) # if chosen too far out (compared to exp_center and max_order) then the shd representation will not be good
    pos_dir = np.ones((num_rotations,1)) * pos_mic

    rect = region.Cuboid((1, 1, 1), np.zeros(3), (0.1, 0.1, 0.1), rng=rng)
    #rect = region.Rectangle((1, 1), np.zeros(3), (0.04, 0.04))
    #pos_omni = rect.equally_spaced_points()
    pos_omni = rect.sample_points(512)

    #mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    #mic_dir = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))
    mic_dir = mc.uniform_random_on_sphere(num_rotations, rng)

    p = {}

    pos_omni, p_omni_sim, ir_mic, pos_cardioid, p_cardioid_sim, freqs, sim_info = _generate_plane_wave_with_ism(sr, pw_dir, pos_omni, pos_dir, mic_dir)
    fi = int(p_omni_sim.shape[0] * 0.8)
    p_omni_sim = p_omni_sim[fi,...]
    p_cardioid_sim = p_cardioid_sim[fi,...]
    p["cardioid_sim"] = p_cardioid_sim

    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_mic_analytic, exp_center, gain_adjustment = pw.find_matching_plane_wave(p_omni_sim, freq, pw_dir, pos_omni, sim_info.c)
    shd_coeffs = gain_adjustment * pw.shd_coeffs_for_planewave(pw_dir, max_order)

    p_omni_shd = sph.apply_measurement_omni(shd_coeffs, pos_omni, exp_center, wave_num)
    mse_plane_wave = np.mean(np.abs(p_omni_shd - p_omni_sim)**2) / np.mean(np.abs(p_omni_sim)**2)
    #plm.image_scatter_freq_response(p, freq, pos_omni, dot_size = 60)
    #plt.show()
    assert mse_plane_wave < 1e-5, "The data used to check the model is not correct"

    dir_coeffs_cardioid = sph.directivity_linear(0.5, mic_dir)[None,...]
    p["cardioid_shd"] = sph.apply_measurement(shd_coeffs, pos_dir, exp_center, wave_num, dir_coeffs_cardioid)

    mse = np.mean(np.abs(p["cardioid_shd"] - p["cardioid_sim"])**2) / np.mean(np.abs(p["cardioid_sim"])**2)

    for name in p:
        p[name] = np.squeeze(p[name])

    # fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    # for name, val in p.items():
    #     axes[0].plot(mic_angle, np.real(val), label=name)
    #     axes[1].plot(mic_angle, np.imag(val), label=name)
    #     axes[2].plot(mic_angle, np.abs(val), label=name)
    # axes[0].set_title("Real part")
    # axes[1].set_title("Imaginary part")
    # axes[2].set_title("Magnitude")
    # axes[0].legend()
    # for ax in axes:
    #     aspplot.set_basic_plot_look(ax)
    #     ylim = ax.get_ylim()
    #     ylim_max = np.max(np.abs(ylim))
    #     ax.set_ylim(-ylim_max, ylim_max)
    # fig.suptitle(f"Total MSE: {mse}")
    # plt.show()

    assert mse < 1e-5, f"MSE: {mse}"



def test_measurement_model_with_cardioid_for_multiple_analytic_plane_waves_gives_same_value_as_simulated_plane_wave():
    """
    This is meant to test whether the cardioid model breaks down when there are multiple plane waves, as it breaks
    down for spherically spreading waves.
    
    As the test test_matched_analytic_plane_wave_is_very_similar_to_simulated_plane_wave shows, the data
    from the simulator and the analytic representation only matches up to an MSE of around 1e-6. Therefore the
    directional measurement model will not match better than that. 
    """
    rng = np.random.default_rng()
    sr = 1000
    max_order = 14
    num_rotations = 30
    num_plane_waves = 5
    
    pw_dir = mc.uniform_random_on_sphere(num_plane_waves, rng)

    pos_mic = rng.uniform(-0.5, 0.5, size=(1,3)) # if chosen too far out (compared to exp_center and max_order) then the shd representation will not be good
    pos_dir = np.ones((num_rotations,1)) * pos_mic

    rect = region.Cuboid((1, 1, 1), np.zeros(3), (0.1, 0.1, 0.1), rng=rng)
    #rect = region.Rectangle((1, 1), np.zeros(3), (0.04, 0.04))
    #pos_omni = rect.equally_spaced_points()
    pos_omni = rect.sample_points(128)

    #mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    #mic_dir = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))
    mic_dir = mc.uniform_random_on_sphere(num_rotations, rng)

    p_cardioid = {}
    p_omni = {}
    shd = {}

    for i in range(num_plane_waves):
        pos_omni, p_omni_sim, ir_mic, pos_cardioid, p_cardioid_sim, freqs, sim_info = _generate_plane_wave_with_ism(sr, pw_dir[i:i+1,:], pos_omni, pos_dir, mic_dir, rng.uniform(300, 900))
        fi = int(p_omni_sim.shape[0] * 0.8)
        p_omni_sim = p_omni_sim[fi,...]
        p_cardioid_sim = p_cardioid_sim[fi,...]
        if not "cardioid_sim" in p_cardioid:
            p_cardioid["cardioid_sim"] = np.zeros_like(p_cardioid_sim)
        if not "omni_sim" in p_omni:
            p_omni["omni_sim"] = np.zeros_like(p_omni_sim)
        p_cardioid["cardioid_sim"] += p_cardioid_sim
        p_omni["omni_sim"] += p_omni_sim

        freq = freqs[fi:fi+1]
        wave_num = 2 * np.pi * freq / sim_info.c
        p_mic_analytic, exp_center, gain_adjustment = pw.find_matching_plane_wave(p_omni_sim, freq, pw_dir[i:i+1,:], pos_omni, sim_info.c)
        shd_coeffs = gain_adjustment * pw.shd_coeffs_for_planewave(pw_dir[i:i+1,:], max_order)
        new_exp_center = np.zeros((1,3))
        shd_coeffs = sph.translate_shd_coeffs(shd_coeffs, new_exp_center-exp_center, wave_num, max_order)

        if not "shd_coeffs" in shd:
            shd["shd_coeffs"] = np.zeros_like(shd_coeffs)
        shd["shd_coeffs"] += shd_coeffs
        
    p_omni_shd = sph.apply_measurement_omni(shd["shd_coeffs"], pos_omni, new_exp_center, wave_num)
    mse_plane_wave = np.mean(np.abs(p_omni_shd - p_omni["omni_sim"])**2) / np.mean(np.abs(p_omni["omni_sim"])**2)
    #plm.image_scatter_freq_response(p, freq, pos_omni, dot_size = 60)
    #plt.show()
    assert mse_plane_wave < 1e-5, "The data used to check the model is not correct"

    dir_coeffs_cardioid = sph.directivity_linear(0.5, mic_dir)[None,...]
    p_cardioid["cardioid_shd"] = sph.apply_measurement(shd["shd_coeffs"], pos_dir, new_exp_center, wave_num, dir_coeffs_cardioid)

    mse = np.mean(np.abs(p_cardioid["cardioid_shd"] - p_cardioid["cardioid_sim"])**2) / np.mean(np.abs(p_cardioid["cardioid_sim"])**2)

    for name in p_cardioid:
        p_cardioid[name] = np.squeeze(p_cardioid[name])

    # fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    # for name, val in p_cardioid.items():
    #     axes[0].plot(mic_angle, np.real(val), label=name)
    #     axes[1].plot(mic_angle, np.imag(val), label=name)
    #     axes[2].plot(mic_angle, np.abs(val), label=name)
    # axes[0].set_title("Real part")
    # axes[1].set_title("Imaginary part")
    # axes[2].set_title("Magnitude")
    # axes[0].legend()
    # for ax in axes:
    #     aspplot.set_basic_plot_look(ax)
    #     ylim = ax.get_ylim()
    #     ylim_max = np.max(np.abs(ylim))
    #     ax.set_ylim(-ylim_max, ylim_max)
    # fig.suptitle(f"Total MSE: {mse}")
    # plt.show()

    assert mse < 1e-5, f"MSE: {mse}"







def test_matched_analytic_plane_wave_is_very_similar_to_simulated_plane_wave():
    rng = np.random.default_rng()
    sr = 2000
    side_len = 1

    pw_dir = mc.uniform_random_on_sphere(1, rng)

    #rect = region.Rectangle((side_len, side_len), np.zeros(3), (0.04, 0.04))
    #pos_mic = rect.equally_spaced_points()
    rect = region.Cuboid((side_len, side_len, side_len), np.zeros(3), rng=rng)
    pos_mic = rect.sample_points(1000)
    

    pos_mic, p_mic_sim, ir_mic_sim, _, _, freqs, sim_info = _generate_plane_wave_with_ism(sr, pw_dir, pos_mic)
    fi = int(p_mic_sim.shape[0] * 0.8)
    p_mic_sim = p_mic_sim[fi,...]

    freq = freqs[fi:fi+1]
    p_mic_analytic, _, _ = pw.find_matching_plane_wave(p_mic_sim, freq, pw_dir, pos_mic, sim_info.c)

    #plm.image_scatter_freq_response({"sim" : p_mic_sim, "analytic" : p_mic_analytic}, freq, pos_mic, dot_size = 60)
    #plt.show()

    mse = np.mean(np.abs(p_mic_sim - p_mic_analytic)**2) / np.mean(np.abs(p_mic_sim)**2)
    ##plt.plot(np.squeeze(ir_mic_sim).T)
    #plt.title(mse)
    #plt.show()
    assert mse < 1e-5

def test_matched_analytic_plane_wave_is_identical_to_original_analytic_plane_wave():
    rng = np.random.default_rng()
    freq = np.ones(1) * 400
    c = 343
    wave_num = 2 * np.pi * freq / c
    side_len = 1

    pw_dir = mc.uniform_random_on_sphere(1, rng)
    rect = region.Cuboid((side_len, side_len, side_len), np.zeros(3), rng=rng)
    pos_mic = rect.sample_points(1000)

    p_mic_orig = np.squeeze(pw.plane_wave(pos_mic, pw_dir, wave_num), axis=-1)
    p_mic_analytic, _, _ = pw.find_matching_plane_wave(p_mic_orig, freq, pw_dir, pos_mic, c)
    assert np.mean(np.abs(p_mic_orig - p_mic_analytic)**2) / np.mean(np.abs(p_mic_orig)**2) < 1e-20

def test_matched_analytic_plane_wave_pressure_and_reconstructed_analytic_plane_wave_are_identical():
    rng = np.random.default_rng()
    sr = 1000
    pw_dir = mc.uniform_random_on_sphere(1, rng)

    pos_mic, p_mic_sim, _, _, _, freqs, sim_info = _generate_plane_wave_with_ism(sr, pw_dir)
    fi = int(p_mic_sim.shape[0] * 0.8)
    p_mic_sim = p_mic_sim[fi,...]

    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_mic_matched, exp_center, gain_adjustment = pw.find_matching_plane_wave(p_mic_sim, freq, pw_dir, pos_mic, sim_info.c)

    pw_reconstructed = np.squeeze(pw.plane_wave(pos_mic, pw_dir, wave_num, exp_center), axis=-1)
    pw_reconstructed = pw_reconstructed * gain_adjustment

    assert np.mean(np.abs(pw_reconstructed - p_mic_matched)**2) / np.mean(np.abs(p_mic_matched)**2) < 1e-20
    
def test_matched_analytic_plane_wave_reconstruction_and_spherical_harmonic_reconstruction_are_identical():
    rng = np.random.default_rng()
    sr = 1000
    max_order = 30
    pw_dir = mc.uniform_random_on_sphere(1, rng)

    side_len = 1
    rect = region.Cuboid((side_len, side_len, side_len), np.zeros(3), rng=rng)
    pos_mic = rect.sample_points(1000)

    pos_mic, p_mic_sim, _, _, _, freqs, sim_info = _generate_plane_wave_with_ism(sr, pw_dir, pos_mic)
    fi = int(p_mic_sim.shape[0] * 0.8)
    p_mic_sim = p_mic_sim[fi,...]

    freq = freqs[fi:fi+1]
    wave_num = 2 * np.pi * freq / sim_info.c
    p_mic_matched, exp_center, gain_adjustment = pw.find_matching_plane_wave(p_mic_sim, freq, pw_dir, pos_mic, sim_info.c)

    shd_coeffs = gain_adjustment * pw.shd_coeffs_for_planewave(pw_dir, max_order)
    p_mic_shd = sph.reconstruct_pressure(shd_coeffs, pos_mic, exp_center, wave_num)
    assert np.mean(np.abs(p_mic_shd - p_mic_matched)**2) / np.mean(np.abs(p_mic_matched)**2) < 1e-15



def _generate_plane_wave_with_ism(sr, pw_dir, pos_mic=None, pos_dir=None, mic_dir=None, src_distance=1000):
    rng = np.random.default_rng(10)
    side_len = 0.5
    num_mic = 40

    assert np.allclose(np.linalg.norm(pw_dir), 1)

    #pos_mic = np.zeros((num_mic, 3))
    if pos_mic is None:
        pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))

    if pos_dir is None:
        pos_dir = np.zeros((1,3))
        mic_dir = np.array([[1,0,0]])
    
    #src_distance = 1000
    pos_src = pw_dir * src_distance 

    setup = _get_default_simulator_setup(sr)
    prop_delay = np.linalg.norm(pos_src) / setup.sim_info.c
    extra_delay_sec = 0.1
    rir_len = coreutils.next_power_of_two((1.5 * prop_delay + extra_delay_sec) * sr)
    extra_delay_samples = int(extra_delay_sec * sr)

    setup.sim_info.room_size = [3 * src_distance, 3*src_distance, 3*src_distance]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 =  0
    setup.sim_info.max_room_ir_length = rir_len
    setup.sim_info.sim_buffer = rir_len
    setup.sim_info.extra_delay = extra_delay_samples

    setup.add_mics("omni", pos_mic)
    if mic_dir.shape[0] == 1:
        mic_dir = np.ones((pos_dir.shape[0], 1)) * mic_dir
    setup.add_mics("cardioid", pos_dir, directivity_type=pos_dir.shape[0]*["cardioid"], directivity_dir=mic_dir)
    setup.add_controllable_source("src", pos_src)
    sim = setup.create_simulator()

    #sim.arrays.paths["src"]["omni"] = 10000 * sim.arrays.paths["src"]["omni"]
    #sim.arrays.paths["src"]["cardioid"] = 10000 * sim.arrays.paths["src"]["cardioid"]

    num_freqs = setup.sim_info.max_room_ir_length
    #fpaths, freqs = sim.arrays.get_freq_paths(num_freqs, sr)
    fpaths, freqs = _get_freq_paths_correct_time_convention(sim.arrays, num_freqs, sr)

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], sim.arrays.paths["src"]["omni"], \
            sim.arrays["cardioid"].pos, fpaths["src"]["cardioid"][...,0], freqs, sim.sim_info



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
    
    freqs = (samplerate / (num_freqs)) * np.arange(num_freqs // 2 + 1)
    num_real_freqs = ft.get_real_freqs(num_freqs, samplerate).shape[-1]

    fpaths = {}
    for src, mic, path in arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        freq_domain_path = ft.rfft(path, n=num_freqs)
        fpaths[src.name][mic.name] = np.moveaxis(freq_domain_path, 1,2)
    return fpaths, freqs




def _get_default_simulator_setup(sr):
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
    setup.sim_info.extra_delay = 60
    setup.sim_info.plot_output = "none"
    return setup


