import matplotlib.pyplot as plt
import numpy as np
import pathlib

from aspsim.simulator import SimulatorSetup
import aspsim.room.region as reg

import aspcol.kernelinterpolation as ki
import aspcol.filterdesign as fd
import aspcol.soundfieldestimation as sfe
import aspcol.sphericalharmonics as sph
import aspcol.fouriertransform as ft


def run_exp():
    samplerate = 1000
    pos_mic, p_mic, pos_eval, p_eval, freqs, fig_folder, sim_info = generate_data(samplerate)
    wave_num = freqs / sim_info.c
    
    reg_param = 1e-4
    max_order = 6
    exp_center = np.zeros((1,3))
    omni_dir = sph.directivity_omni(max_order=1)

    harmonics = {}
    harmonics["omni"] = sph.inf_dimensional_shd_omni(p_mic, pos_mic, exp_center, max_order, wave_num, reg_param)
    harmonics["omni-general"] = sph.inf_dimensional_shd(p_mic, pos_mic, exp_center, max_order, wave_num, reg_param, dir_coeffs = omni_dir)

    pressure_est = {}
    pressure_est["ki"] = sfe.est_ki_diffuse_freq(p_mic, pos_mic, pos_eval, wave_num, reg_param)
    for harm in harmonics:
        pressure_est[harm] = sph.reconstruct_pressure(harmonics[harm], pos_eval, exp_center, max_order, wave_num)


    for est_name, est in pressure_est.items():
        print(f"MSE {est_name}: {10*np.log10(np.mean(np.abs(est - p_eval)**2))} dB")

    fig, ax = plt.subplots(1,1)
    for est_name, est in pressure_est.items():
        ax.plot(freqs, 10*np.log10(np.mean(np.abs(est - p_eval)**2, axis=-1)), label=est_name)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("MSE [dB]")
    plt.legend()
    plt.show()
    

def generate_data(sr):
    rng = np.random.default_rng(10)
    side_len = 0.6
    num_mic = 12
    
    eval_region = reg.Rectangle((side_len, side_len), (0,0,0), (0.1, 0.1))
    pos_eval = eval_region.equally_spaced_points()

    pos_mic = np.zeros((num_mic, 3))
    pos_mic[:,:2] = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 2))
    pos_src = np.array([[2,0.05,-0.05]])

    setup = SimulatorSetup(pathlib.Path(__file__).parent.joinpath("figs"))
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [5, 4, 2]
    setup.sim_info.room_center = [0.4, 0.1, 0.1]
    setup.sim_info.rt60 =  0.25
    setup.sim_info.max_room_ir_length = sr // 2
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