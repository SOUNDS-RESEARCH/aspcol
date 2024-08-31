import matplotlib.pyplot as plt
import numpy as np
import pathlib

from aspsim.simulator import SimulatorSetup
import aspsim.room.region as reg

import aspcol.kernelinterpolation as ki
import aspcol.filterdesign as fd
import aspcol.fouriertransform as ft
import aspcol.soundfieldestimation as sfe


def run_exp():
    samplerate = 1000
    pos_mic, p_mic, pos_eval, p_eval, freqs, fig_folder, sim_info = generate_data(samplerate)
    wave_num = freqs / sim_info.c
    
    reg_param = 1e-5

    est_ki = sfe.est_ki_diffuse_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param)
    est_ki_dir = est_ki_directional_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param, np.array([[1,0,0]]), 5)

    print(f"MSE kernel interpolation: {10*np.log10(np.mean(np.abs(est_ki - p_eval)**2))} dB")
    print(f"MSE directional kernel interpolation: {10*np.log10(np.mean(np.abs(est_ki_dir - p_eval)**2))} dB")

    fig, ax = plt.subplots(1,1)
    ax.plot(freqs, 10*np.log10(np.mean(np.abs(est_ki - p_eval)**2, axis=-1)), label="diffuse KI")
    ax.plot(freqs, 10*np.log10(np.mean(np.abs(est_ki_dir - p_eval)**2, axis=-1)), label="directional KI")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("MSE [dB]")
    plt.legend()
    plt.show()

def est_ki_directional_freq(p_freq, pos_mic, pos_eval, wave_num, reg_param, direction, direction_param):
    est_filt = ki.get_krr_parameters(ki.kernel_directional_3d, reg_param, pos_eval, pos_mic, wave_num, direction, direction_param)[:,0,:,:]
    p_ki = est_filt @ p_freq[:,:,None]
    return np.squeeze(p_ki, axis=-1)




def generate_data(sr):
    rng = np.random.default_rng(10)
    side_len = 1
    num_mic = 20
    
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
    setup.sim_info.rt60 =  0.25
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
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(ft.rfft(path, n=num_freqs), 1, 2)
        
    return sim.arrays["mic"].pos, fpaths["src"]["mic"][...,0], sim.arrays["eval"].pos, fpaths["src"]["eval"][...,0], freqs, sim.folder_path, sim.sim_info



if __name__ == "__main__":
    run_exp()