import numpy as np
import pathlib
import copy
import matplotlib.pyplot as plt

import aspcore.fouriertransform as ft
import aspcore.montecarlo as mc

from aspsim.simulator import SimulatorSetup
import aspsim.room.region as reg
import aspsim.array as ar
import aspsim.room.generatepoints as gp

import aspcol.utilities as aspcolutil
import aspcol.plot as aspplot




def generate_data(num_mic, 
                  num_src, 
                  num_dense, 
                  rt60s, 
                  snr, 
                  c = 343,
                  different_mic_pos = False, 
                  output_method = "pdf", 
                  main_fig_path=pathlib.Path(__file__).parent.joinpath("figs"),
                  eval_spacing = 0.05,
                  rng= None):
    if not isinstance(rt60s, (tuple, list, np.ndarray)):
        rt60s = [rt60s]
    if not isinstance(c, (tuple, list, np.ndarray)):
        c = [c]

    if len(c) > 1 and len(rt60s) == 1:
        rt60s = rt60s * len(c)
    if len(rt60s) > 1 and len(c) == 1:
        c = c * len(rt60s)
    assert len(c) == len(rt60s)

    if rng is None:
        rng = np.random.default_rng()
    #assert len(rt60s) == 2

    side_len = 0.5
    z_len = 0.15
    center = (0.5, 0, 0)
    eval_spacing_all_dims = (eval_spacing,eval_spacing, eval_spacing)
    target_region = reg.Cuboid((side_len, side_len, z_len), center, eval_spacing_all_dims,rng)
    #target_region = reg.Cylinder(side_len, z_len, center, eval_spacing_all_dims, rng)

    assert num_mic % 2 == 0
    side_xtra = 0.05
    pos_mic = target_region.center + np.concatenate([gp.equidistant_rectangle(num_mic//2, (side_len, side_len), offset=0.75, z=0), 
                                        gp.equidistant_rectangle(num_mic//2, (side_len+side_xtra, side_len+side_xtra), offset=0.25, z=0)], axis=0)
    #pos_mic = target_region.sample_points(num_mic)
    pos_dense =  target_region.sample_points(num_dense)
    pos_src = gp.equiangular_circle(num_src, 1.5, start_angle = 0, z = 0, rng=rng)

    image_spacing = eval_spacing / 2
    image_region = reg.Rectangle((1.2*side_len, 1.2*side_len), center, (image_spacing, image_spacing))
    pos_image = image_region.equally_spaced_points()

    # if different_mic_pos:
    #     pos_mic_prev = target_region.sample_points(num_mic)
    # else:
    #     pos_mic_prev = pos_mic

    sim = []
    for i, (rt60, c_val) in enumerate(zip(rt60s, c)):
        #if different_mic_pos:
        #    pos_mic = target_region.sample_points(num_mic)
        #rng_single_dataset = np.random.default_rng(6543456)
        if i == len(rt60s) - 1:
            pos_image_to_use = pos_image
        else:
            pos_image_to_use = None

        ir = {
            "data" : [],
            "eval" : [],
            "dense" : [],
            "image" : [],
        }
        freqs  =[]
        wave_num = []
        sim_, ir_data_, ir_eval_, ir_dense_, ir_image_, wave_num_, freqs_ = generate_single_dataset(copy.deepcopy(target_region), pos_mic, pos_src, pos_dense, rt60, snr, c_val, rng, output_method, main_fig_path=main_fig_path, pos_image = pos_image_to_use)
        sim.append(sim_)
        ir["data"].append(ir_data_)
        ir["eval"].append(ir_eval_)
        ir["dense"].append(ir_dense_)
        ir["image"].append(ir_image_)
        freqs.append(freqs_)
        wave_num.append(wave_num_)


    # if rt60s[0] != rt60s[1]:
    #     _, ir_data_prev, _, ir_dense_prev, _, _ = generate_single_dataset(copy.deepcopy(target_region), pos_mic_prev, num_src, rt60s[0], snr, rng, output_method, main_fig_path=main_fig_path)
    # else:
    #     ir_data_prev = ir_data
    #     ir_dense_prev = ir_dense
    

    return sim, ir, wave_num, freqs


def generate_single_dataset(target_region, pos_mic, pos_src, pos_dense, rt60, snr, c, rng, output_method = "pdf", main_fig_path = pathlib.Path(__file__).parent.joinpath("figs"), pos_image = None): #pos_mic.shape[0]
    #c = 343
    samplerate = 2000
    ir_len = int(samplerate*0.8)
    pos_eval = target_region.equally_spaced_points()

    setup = SimulatorSetup(main_fig_path)

    setup.sim_info.tot_samples = ir_len
    setup.sim_info.samplerate = samplerate
    setup.sim_info.c = c
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [5.4, 4.3, 3.2]
    setup.sim_info.room_center = [0.5, 0.1, 0.1]
    setup.sim_info.rt60 = rt60
    setup.sim_info.max_room_ir_length = ir_len
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = ir_len
    setup.sim_info.extra_delay = 64
    setup.sim_info.plot_output = output_method
    setup.sim_info.start_sources_before_0 = True

    setup.add_mics("mic", pos_mic)
    setup.add_array(ar.RegionArray("eval", target_region, pos_eval))
    setup.add_mics("dense", pos_dense)
    setup.add_controllable_source("src", pos_src)
    if pos_image is not None:
        setup.add_array(ar.MicArray("image", pos_image))

    sim = setup.create_simulator()

    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    freqs = ft.get_real_freqs(ir_len, samplerate)

    #ir_data = sim.arrays.paths["src"]["mic"] 
    ir_data = ft.rfft(sim.arrays.paths["src"]["mic"] )
    ir_eval = ft.rfft(sim.arrays.paths["src"]["eval"])
    ir_dense = ft.rfft(sim.arrays.paths["src"]["dense"])

    #ir_data += rng.normal(0, noise_power, ir_data.shape)

    sig_power = np.mean(np.abs(ir_data)**2, axis=(1,2))
    noise_power = sig_power / aspcolutil.db2pow(snr)
    noise_power = noise_power[:,None,None]
    scale = np.sqrt(noise_power / 2)
    noise = rng.normal(0, scale, ir_data.shape) + 1j * rng.normal(0, scale, ir_data.shape)
    ir_data += noise

    fig, ax = plt.subplots(1,1)
    measured_snr = np.mean(np.abs(ir_data)**2, axis=(1,2)) / np.mean(np.abs(noise)**2, axis=(1,2))
    measured_snr_db = 10 * np.log10(measured_snr)   
    ax.plot(measured_snr_db)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("SNR [dB]")
    ax.set_title("Measured SNR")
    aspplot.save_plot(output_method, sim.folder_path, "measured_snr")
    #noise_pow_factor = current_snr / aspcolutil.db2pow(snr)
    #noise_data = noise_data * np.sqrt(noise_pow_factor)

    # REDUCE NUMBER OF FREQUENCIES
    freq_idxs = slice(20, None, 10)
    ir_data = np.copy(ir_data[freq_idxs,...])
    ir_eval = np.copy(ir_eval[freq_idxs,...])
    ir_dense = np.copy(ir_dense[freq_idxs,...])
    wave_num = np.copy(wave_num[freq_idxs])
    freqs = np.copy(freqs[freq_idxs])
    if "image" in sim.arrays:
        ir_image = ft.rfft(sim.arrays.paths["src"]["image"])
        ir_image = np.copy(ir_image[freq_idxs,...])
    else:
        ir_image = None

    return sim, ir_data, ir_eval, ir_dense, ir_image, wave_num, freqs







def generate_covariance_noise(cov, snr, rng):
    """

    Parameters
    ----------
    cov : ndarray of shape (num_freq, M, M)
        covariance matrix
    snr : float or ndarray of shape (num_freq)
        signal to noise ratio in dB
    """
    if np.isinf(snr):
        return np.zeros_like(cov)
    dim = cov.shape[-1]
    num_freq = cov.shape[0]
    noise_vectors = np.stack([mc.sample_complex_gaussian(np.zeros(dim), np.eye(dim), rng, 3 * dim) for i in range(num_freq)], axis=0)

    noise_cov = noise_vectors @ np.moveaxis(noise_vectors.conj(), -1, -2)

    if np.isscalar(snr):
        snr = np.ones((num_freq)) * snr

    noise_scaling = np.trace(cov, axis1=-2, axis2=-1) / (np.trace(noise_cov, axis1=-2, axis2=-1) * aspcolutil.db2pow(snr))
    noise_cov = noise_cov * noise_scaling[:,None,None]
    return noise_cov

def add_noise_to_ir(ir_data, snr_db, rng):
    sig_power = np.mean(np.abs(ir_data)**2, axis=(1,2))
    noise_power = sig_power / aspcolutil.db2pow(snr_db)
    noise_power = noise_power[:,None,None]
    noise = rng.normal(0, noise_power/2, ir_data.shape) + 1j * rng.normal(0, noise_power/2, ir_data.shape)
    ir_data = ir_data + noise
    return ir_data


def generate_sound_zone_data(num_mic, num_src, num_dense, samplerate, rt60s, snr, c, output_method = "pdf", eval_spacing = 0.1, main_fig_path=pathlib.Path(__file__).parent.joinpath("figs")):
    assert isinstance(rt60s, (tuple, list, np.ndarray))
    #    rt60s = [rt60s]
    rng = np.random.default_rng(6543456)

    ir_len = samplerate//2  #// 2 

    setup = SimulatorSetup(main_fig_path)

    setup.sim_info.tot_samples = 8 * samplerate
    setup.sim_info.samplerate = samplerate
    setup.sim_info.c = c[0]
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [5.4, 4.3, 3.2]
    setup.sim_info.room_center = [0.5, 0.1, 0.1]
    setup.sim_info.rt60 = rt60s[0]
    setup.sim_info.max_room_ir_length = ir_len
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = 2 * samplerate
    setup.sim_info.extra_delay = 64
    setup.sim_info.output_smoothing = 128
    setup.sim_info.plot_output = output_method
    setup.sim_info.start_sources_before_0 = True

    #num_dense = 64
    side_len = 0.5
    z_len = 0.15
    #eval_spacing = 0.15
    #z_len = eval_spacing
    eval_spacing_all = (eval_spacing, eval_spacing, eval_spacing)
    bright_zone = reg.Cuboid((side_len, side_len, z_len), (-0.5, 0, 0), eval_spacing_all, rng)
    dark_zone = reg.Cuboid((side_len, side_len, z_len), (0.5, 0, 0), eval_spacing_all, rng)
    #pos_mic = bright_zone.sample_points(num_mic)

    setup.add_array(ar.RegionArray("bright_eval", bright_zone, bright_zone.equally_spaced_points()))
    setup.add_array(ar.MicArray("bright_dense", bright_zone.sample_points(num_dense)))
    setup.add_array(ar.RegionArray("dark_eval", dark_zone, dark_zone.equally_spaced_points()))
    setup.add_array(ar.MicArray("dark_dense", dark_zone.sample_points(num_dense)))


    #assert num_mic % 2 == 0
    side_xtra = 0.05
    bright_mic_pos = bright_zone.center + gp.equidistant_rectangle(num_mic, (side_len, side_len), extra_side_lengths=(side_xtra, side_xtra), offset=0.5, z = 0)
    #+ np.concatenate([gp.equidistant_rectangle(num_mic//2, (side_len, side_len), offset=0.75, z=0), 
                                        #gp.equidistant_rectangle(num_mic//2, (side_len+side_xtra, side_len+side_xtra), offset=0.25, z=0)], axis=0)
    dark_mic_pos = dark_zone.center + gp.equidistant_rectangle(num_mic, (side_len, side_len), extra_side_lengths=(side_xtra, side_xtra), offset=0.5, z = 0)
    #bright_mic_pos = bright_zone.sample_points(num_mic)
    #dark_mic_pos = dark_zone.sample_points(num_mic)
    setup.add_array(ar.MicArray("bright_mic", bright_mic_pos))
    setup.add_array(ar.MicArray("dark_mic", dark_mic_pos))

    pos_src = gp.equiangular_circle(num_src, 1.5, start_angle = 0, z = 0)
    setup.add_controllable_source("src", pos_src)

    sims = [setup.create_simulator()]
    if rt60s[0] == rt60s[1] and c[0] == c[1]:
        sims.append(sims[0])
    else:
        setup.sim_info.rt60 = rt60s[1]
        setup.sim_info.c = c[1]
        sim = setup.create_simulator()
        sims.append(sim)
    ir_data_bright = add_noise_to_ir(ft.rfft(sims[-1].arrays.paths["src"]["bright_mic"]), snr, rng)
    ir_data_dark = add_noise_to_ir(ft.rfft(sims[-1].arrays.paths["src"]["dark_mic"]), snr, rng)

    return sims, setup, ir_data_bright, ir_data_dark






def generate_sound_zone_data_test(num_mic, num_src, num_dense, samplerate, rt60s, snr, c, output_method = "pdf", eval_spacing = 0.1, main_fig_path=pathlib.Path(__file__).parent.joinpath("figs")):
    assert isinstance(rt60s, (tuple, list, np.ndarray))
    #    rt60s = [rt60s]
    rng = np.random.default_rng(6543456)

    ir_len = samplerate

    setup = SimulatorSetup(main_fig_path)

    setup.sim_info.tot_samples = 8 * samplerate
    setup.sim_info.samplerate = samplerate
    setup.sim_info.c = c[0]
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [5.4, 4.3, 3.2]
    setup.sim_info.room_center = [0.5, 0.1, 0.1]
    setup.sim_info.rt60 = rt60s[0]
    setup.sim_info.max_room_ir_length = ir_len
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = 2 * samplerate
    setup.sim_info.extra_delay = 64
    setup.sim_info.output_smoothing = 128
    setup.sim_info.plot_output = output_method
    setup.sim_info.start_sources_before_0 = True

    #num_dense = 64
    side_len = 0.5
    z_len = 0.15
    #eval_spacing = 0.15
    #z_len = eval_spacing
    eval_spacing_all = (eval_spacing, eval_spacing, eval_spacing)
    bright_zone = reg.Cuboid((side_len, side_len, z_len), (-0.5, 0, 0), eval_spacing_all, rng)
    dark_zone = reg.Cuboid((side_len, side_len, z_len), (0.5, 0, 0), eval_spacing_all, rng)
    #pos_mic = bright_zone.sample_points(num_mic)

    setup.add_array(ar.RegionArray("bright_eval", bright_zone, bright_zone.equally_spaced_points()))
    setup.add_array(ar.MicArray("bright_dense", bright_zone.sample_points(num_dense)))
    setup.add_array(ar.RegionArray("dark_eval", dark_zone, dark_zone.equally_spaced_points()))
    setup.add_array(ar.MicArray("dark_dense", dark_zone.sample_points(num_dense)))


    #assert num_mic % 2 == 0
    side_xtra = 0.05
    bright_mic_pos = bright_zone.center + gp.equidistant_rectangle(num_mic, (side_len, side_len), extra_side_lengths=(side_xtra, side_xtra), offset=0.5, z = 0)
    #+ np.concatenate([gp.equidistant_rectangle(num_mic//2, (side_len, side_len), offset=0.75, z=0), 
                                        #gp.equidistant_rectangle(num_mic//2, (side_len+side_xtra, side_len+side_xtra), offset=0.25, z=0)], axis=0)
    dark_mic_pos = dark_zone.center + gp.equidistant_rectangle(num_mic, (side_len, side_len), extra_side_lengths=(side_xtra, side_xtra), offset=0.5, z = 0)
    #bright_mic_pos = bright_zone.sample_points(num_mic)
    #dark_mic_pos = dark_zone.sample_points(num_mic)
    setup.add_array(ar.MicArray("bright_mic", bright_mic_pos))
    setup.add_array(ar.MicArray("dark_mic", dark_mic_pos))

    pos_src = gp.equiangular_circle(num_src, 1.4, start_angle = 0, z = 0)#previously 1.5m
    #pos_src[::2,:2] *= 1.1
    setup.add_controllable_source("src", pos_src) #

    sims = [setup.create_simulator()]
    if rt60s[0] == rt60s[1] and c[0] == c[1]:
        sims.append(sims[0])
    else:
        setup.sim_info.rt60 = rt60s[1]
        setup.sim_info.c = c[1]
        sim = setup.create_simulator()
        sims.append(sim)
    ir_data_bright = add_noise_to_ir(ft.rfft(sims[-1].arrays.paths["src"]["bright_mic"]), snr, rng)
    ir_data_dark = add_noise_to_ir(ft.rfft(sims[-1].arrays.paths["src"]["dark_mic"]), snr, rng)

    return sims, setup, ir_data_bright, ir_data_dark


