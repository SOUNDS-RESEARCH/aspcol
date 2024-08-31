import numpy as np
import scipy.signal as spsig

import aspcol.sphericalharmonics as sph
import aspcol.utilities as utils
import aspcol.fouriertransform as ft
import aspcol.filterdesign as fd
import aspcol.plot as aspplot

from aspsim.simulator import SimulatorSetup
import aspsim.room.region as region

import matplotlib.pyplot as plt

import hypothesis as hyp
import hypothesis.strategies as st

@hyp.settings(deadline=None)
@hyp.given(half_ir_len = st.integers(min_value=16, max_value=256))
def test_fir_from_frequency_response_always_gives_real_valued_filter(half_ir_len):
    rng = np.random.default_rng(123456)
    def freq_resp(f):
        return rng.uniform(-1, 1, size=f.shape) + 1j * rng.uniform(-1, 1, size=f.shape)

    ir_len = half_ir_len * 2 + 1
    ir = fd.fir_from_frequency_function(freq_resp, ir_len)

    # fig, ax = plt.subplots(1,1)
    # ax.plot(np.real(ir))
    # ax.plot(np.imag(ir))
    # plt.show()

    assert np.allclose(0, np.imag(ir)), "The impulse response should be real valued"

@hyp.settings(deadline=None)
@hyp.given(half_ir_len = st.integers(min_value=16, max_value=256))
def test_fir_from_frequency_response_gives_correct_group_delay(half_ir_len):
    ir_len = 2 * half_ir_len + 1
    def freq_resp(f):
       return np.ones_like(f)

    ir = fd.fir_from_frequency_function(freq_resp, ir_len)

    # fig, ax = plt.subplots(1,1)
    # ax.plot(ir)
    
    w, gd = spsig.group_delay((ir, 1), 512)
    # fig, ax = plt.subplots(1,1)
    # ax.plot(w, gd)
    # plt.show()

    assert np.allclose(np.median(gd), half_ir_len)


def test_differential_microphone_filter_gives_same_result_as_pyroomacoustics_for_approximate_plane_wave_sound_field():
    sr = 1000
    num_rotations = 30

    pos_mic = np.zeros((1,3))
    pos_mic_tiled = np.tile(pos_mic, (num_rotations,1))
    mic_angle = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    mic_direction = utils.spherical2cart(np.ones(num_rotations), np.stack((mic_angle, np.pi/2 * np.ones(num_rotations)),axis=-1))

    diff_mic_distance = 0.01
    pos_mic_secondary = pos_mic - diff_mic_distance * mic_direction

    pos_src = np.array([[0,90,0]])

    setup = SimulatorSetup()
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [200, 200, 200]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.max_room_ir_length = 2*sr
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = sr // 2
    setup.sim_info.plot_output = "none"
    setup.sim_info.rt60 = 0

    setup.add_mics("omni", pos_mic)
    setup.add_mics("secondary_mics", pos_mic_secondary)
    dir_type = num_rotations*["cardioid"]
    setup.add_mics("cardioid", pos_mic_tiled, directivity_type=dir_type, directivity_dir=mic_direction)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    signal_main = np.tile(sim.arrays.paths["src"]["omni"][0,:,:], (num_rotations,1))
    signal_secondary = sim.arrays.paths["src"]["secondary_mics"][0,:,:]
    filter_below = 200
    cardioid_signal = fd.differential_cardioid_microphone(signal_main, signal_secondary, diff_mic_distance, 500, sim.sim_info.c, sim.sim_info.samplerate, filter_below=filter_below)
    
    power_per_angle_pra = np.mean(np.abs(sim.arrays.paths["src"]["cardioid"][0,:,:])**2, axis=-1)
    power_per_angle_diff = np.mean(np.abs(cardioid_signal)**2, axis=-1)
    power_error = np.mean(np.abs(power_per_angle_pra - power_per_angle_diff))

    fig, ax = plt.subplots(1,1)
    ax.set_title(f"Mean power error: {power_error}")
    ax.set_ylabel("Response power")
    ax.set_xlabel("Microphone angle")
    ax.plot(mic_angle, power_per_angle_pra, label="Pyroomacoustics")
    ax.plot(mic_angle, power_per_angle_diff, label="Differential")
    ax.legend()

    fig, ax = plt.subplots(1,1)
    example_idx = 5
    mse_time_response = np.mean(np.abs(cardioid_signal - sim.arrays.paths["src"]["cardioid"][0,:,:])**2) / np.mean(np.abs(sim.arrays.paths["src"]["cardioid"][0,:,:])**2)
    ax.set_title(f"Mean square error: {mse_time_response}")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    ax.plot(sim.arrays.paths["src"]["cardioid"][0,example_idx,:], label="Pyroomacoustics")
    ax.plot(cardioid_signal[example_idx,:], label="Differential")
    ax.plot(sim.arrays.paths["src"]["omni"][0,0,:], label="Pra omni")
    ax.legend()

    fig, axes = plt.subplots(3,1, sharex=True, figsize=(6,8))

    axes[2].set_xlabel("Frequency")
    axes[0].set_ylabel("Absolute")
    axes[1].set_ylabel("Real")
    axes[2].set_ylabel("Imag")
    freqs = ft.get_real_freqs(cardioid_signal.shape[-1], sim.sim_info.samplerate)
    freq_response_pra = ft.rfft(sim.arrays.paths["src"]["cardioid"][0,example_idx,:])
    freq_response_diff = ft.rfft(cardioid_signal[example_idx,:])
    freq_mask = freqs > filter_below
    freq_mask[-5:] = False
    mse = np.mean(np.abs(freq_response_pra[freq_mask] - freq_response_diff[freq_mask])**2) / np.mean(np.abs(freq_response_pra[freq_mask])**2)
    axes[0].set_title(f"Mean square error: {mse}")

    axes[0].plot(freqs, np.abs(freq_response_pra), label="Pyroomacoustics")
    axes[0].plot(freqs, np.abs(freq_response_diff), label="Differential")
    axes[0].plot(freqs, np.abs(ft.rfft(sim.arrays.paths["src"]["omni"][0,0,:])), label="Pra omni")

    axes[1].plot(freqs, np.real(freq_response_pra), label="Pyroomacoustics")
    axes[1].plot(freqs, np.real(freq_response_diff), label="Differential")
    #axes[1].plot(freqs, np.real(ft.rfft(sim.arrays.paths["src"]["omni"][0,0,:])), label="Pra omni")

    axes[2].plot(freqs, np.imag(freq_response_pra), label="Pyroomacoustics")
    axes[2].plot(freqs, np.imag(freq_response_diff), label="Differential")
    #axes[2].plot(freqs, np.imag(ft.rfft(sim.arrays.paths["src"]["omni"][0,0,:])), label="Pra omni")

    for ax in axes:
        ax.legend()
        aspplot.set_basic_plot_look(ax)

    plt.show()







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

    pos_cardioid_1 = pos_cardioid #+ differential_distance * mic_direction / 2
    pos_cardioid_2 = pos_cardioid - differential_distance * mic_direction

    num_cardioid = pos_cardioid.shape[0]



    setup = _get_default_simulator_setup(sr)
    setup.sim_info.rt60 = rt60
    setup.sim_info.room_size = [1000, 1000, 1000]

    setup.add_mics("omni", pos_omni)
    setup.add_mics("eval", pos_eval)
    setup.add_mics("cardioid", pos_cardioid_1)
    setup.add_mics("cardioid_2", pos_cardioid_2)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()



    return sim.arrays["omni"].pos, sim.arrays["cardioid"].pos, sim.arrays["eval"].pos, fpaths["src"]["omni"], fpaths["src"]["cardioid"], fpaths["src"]["eval"], freqs, sim.sim_info

























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

