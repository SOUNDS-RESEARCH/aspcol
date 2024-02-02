import matplotlib.pyplot as plt
import numpy as np
import pathlib

from aspsim.simulator import SimulatorSetup
import aspsim.room.region as reg

import aspcol.kernelinterpolation as ki
import aspcol.filterdesign as fd
import aspcol.soundfieldestimation as sfe
import aspcol.sphericalharmonics as sph


def run_exp():
    samplerate = 1000
    pos_mic, p_mic_omni, p_mic_cardioid, pos_eval, p_eval, freqs, fig_folder, sim_info = generate_data(samplerate)
    wave_num = freqs / sim_info.c
    
    reg_param = 1e-5

    pressure_est = {}
    #pressure_est["kernel interpolation"] = sfe.est_ki_diffuse_freq(p_mic_omni, pos_mic, pos_eval, wave_num, reg_param)
    #pressure_est["kernel interpolation dir"] = sfe.est_ki_diffuse_freq(p_mic_cardioid, pos_mic, pos_eval, wave_num, reg_param)
    pressure_est["inf harm"] = sfe.inf_dim_shd_analysis(p_mic_omni, pos_mic, pos_eval, wave_num, sph.directivity_omni(), reg_param)
    pressure_est["inf harm dir"] = sfe.inf_dim_shd_analysis(p_mic_cardioid, pos_mic, pos_eval, wave_num, sph.directivity_linear(0.5, np.array([[0,1,0]])), reg_param)

    pressure_est["inf harm dir"] *= np.sqrt(np.mean(np.abs(pressure_est["inf harm"])**2) / np.mean(np.abs(pressure_est["inf harm dir"])**2))

    for est_name, est in pressure_est.items():
        print(f"MSE {est_name}: {10*np.log10(np.mean(np.abs(est - p_eval)**2))} dB")

    fig, ax = plt.subplots(1,1)
    for est_name, est in pressure_est.items():
        ax.plot(freqs, 10*np.log10(np.mean(np.abs(est - p_eval)**2, axis=-1) / np.mean(np.abs(p_eval)**2, axis=-1)), label=est_name)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("MSE [dB]")
    ax.legend()

    fig, ax = plt.subplots(1,1)
    for est_name, est in pressure_est.items():
        ax.plot(freqs, 10*np.log10(np.mean(np.abs(est)**2, axis=-1)), label=est_name)
    ax.plot(freqs, 10*np.log10(np.mean(np.abs(p_eval)**2, axis=-1)), label="true")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power spectrum [dB]")
    ax.legend()

    plt.show()
    




def generate_data(sr):
    rng = np.random.default_rng(10) 
    side_len = 0.4
    num_mic = 10
    
    eval_region = reg.Rectangle((side_len, side_len), (0,0,0), (0.1, 0.1))
    pos_eval = eval_region.equally_spaced_points()

    pos_mic = np.zeros((num_mic, 3))
    pos_mic[:,:2] = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 2))
    
    directivity_dir = np.zeros((num_mic, 3))
    directivity_dir[:,1] = 1
    directivity_type = ["cardioid" for _ in range(num_mic)]

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
    setup.add_mics("mic_dir", pos_mic, directivity_type=directivity_type, directivity_dir=directivity_dir)
    setup.add_mics("eval", pos_eval)
    setup.add_controllable_source("src", pos_src)

    sim = setup.create_simulator()

    num_freqs = 512
    freqs = fd.get_real_freqs(num_freqs, sr)
    num_real_freqs = freqs.shape[0]

    fpaths = {}
    for src, mic, path in sim.arrays.iter_paths():
        fpaths.setdefault(src.name, {})
        fpaths[src.name][mic.name] = np.moveaxis(np.moveaxis(np.fft.fft(path, n=num_freqs), -1, 0),1,2)[:num_real_freqs,...]

    return sim.arrays["mic"].pos, fpaths["src"]["mic"][...,0], fpaths["src"]["mic_dir"][...,0], sim.arrays["eval"].pos, fpaths["src"]["eval"][...,0], freqs, sim.folder_path, sim.sim_info



if __name__ == "__main__":
    run_exp()