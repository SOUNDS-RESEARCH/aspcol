import matplotlib.pyplot as plt
import numpy as np
import pathlib

from aspsim.simulator import SimulatorSetup
import aspsim.room.region as reg


import aspcore.fouriertransform as ft
import aspcore.utilities as utils

import aspcol.kernelinterpolation as ki
import aspcol.soundfieldestimation as sfe

import aspcol.soundfieldestimation_jax as sfe_jax


def run_exp():
    samplerate = 1000
    pos_mic, p_mic, pos_eval, p_eval, freqs, fig_folder, sim_info = generate_data(samplerate)
    wave_num = ft.freqs_to_wavenum(freqs, sim_info.c)
    #wave_num = freqs / sim_info.c
    
    reg_param = 1e-4

    # Should be directed in the preferred propagation direction
    direction = np.array([-1, 0, 0])
    beta = 2.0

    estimates_numpy = {}
    estimates_jax = {}

    estimates_numpy["ki"] = sfe.est_ki_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param)
    estimates_numpy["ki_dir"] = sfe.est_ki_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param, ki.kernel_directional_3d, [direction, beta])

    estimates_numpy["rff"] = sfe.est_ki_freq_rff(p_mic, pos_mic, pos_eval, wave_num, reg_param, num_basis = 512)
    estimates_numpy["rff_dir"] = sfe.est_ki_freq_rff(p_mic, pos_mic, pos_eval, wave_num, reg_param, num_basis = 512, direction=direction, beta=beta)

    estimates_jax["ki"] = sfe_jax.est_ki_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param)
    estimates_jax["ki_dir"] = sfe_jax.est_ki_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param, direction=direction, beta=beta)

    estimates_jax["rff"] = sfe_jax.est_ki_freq_rff(p_mic, pos_mic, pos_eval, wave_num, reg_param, num_basis=512)
    estimates_jax["rff_dir"] = sfe_jax.est_ki_freq_rff(p_mic, pos_mic, pos_eval, wave_num, reg_param, num_basis=512, direction=direction, beta=beta)    

    for est_name, est in estimates_numpy.items():
        print(f"MSE Numpy {est_name}: {10*np.log10(np.mean(np.abs(est - p_eval)**2))} dB")
    for est_name, est in estimates_jax.items():
        print(f"MSE JAX {est_name}: {10*np.log10(np.mean(np.abs(est - p_eval)**2))} dB")

    fig, axes = plt.subplots(1,2, figsize=(12, 6))
    axes[0].set_title("Numpy implementations")
    axes[1].set_title("JAX implementations")
    for est_name, est in estimates_numpy.items():
        axes[0].plot(freqs, 10*np.log10(np.mean(np.abs(est - p_eval)**2, axis=-1)), label=est_name)
    for est_name, est in estimates_jax.items():
        axes[1].plot(freqs, 10*np.log10(np.mean(np.abs(est - p_eval)**2, axis=-1)), label=est_name)

    for ax in axes:
        utils.set_basic_plot_look(ax)
        ax.legend()
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("MSE [dB]")
    plt.show()


def generate_data(sr):
    rng = np.random.default_rng(10)
    side_len = 1.5
    num_mic = 12
    
    eval_region = reg.Rectangle((side_len, side_len), (0,0,0), (0.1, 0.1))
    pos_eval = eval_region.equally_spaced_points()

    pos_mic = np.zeros((num_mic, 3))
    pos_mic[:,:2] = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 2))
    pos_src = np.array([[3,0.05,-0.05]])

    setup = SimulatorSetup(pathlib.Path(__file__).parent.joinpath("figs"))
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 =  0.10
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "none"

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 512
    freqs = ft.get_real_freqs(num_freqs, sr)

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(ft.rfft(path, n=num_freqs), 1, 2)
        
    return sim.arrays["mic"].pos, fpaths["src"]["mic"][...,0], sim.arrays["eval"].pos, fpaths["src"]["eval"][...,0], freqs, sim.folder_path, sim.sim_info



if __name__ == "__main__":
    run_exp()