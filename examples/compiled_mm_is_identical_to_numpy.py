import numpy as np
import scipy.signal as spsig

import aspsim.room.generatepoints as gp

import aspcol.soundfieldestimation as sfe
import aspcol.filterdesign as fd
import aspcol.fouriertransform as ft
import aspcol.movingmicrophone as mm
import aspcol.sphericalharmonics as shd
import aspcol.plot as aspplot   

import exp_funcs_ideal_sampling as exis
import sys

sys.path.append("C:/research/papers/2024_icassp_moving_mic_sf_estimation/code")
import plots as fplot


def run_exp(fig_folder):
    sig, sim_info, arrays, pos_dyn, seq_len, extra_params, pos_dyn_cardioid, mic_direction = exis.load_session(fig_folder)

    print(f"Simulation start position: {pos_dyn[0,0,:]}, in polar coords: {gp.cart2pol(pos_dyn[0,0,0], pos_dyn[0,0,1])}")
    #print(f"Start position after init delay: {pos_dyn[extra_params['initial_delay'],0,:]}, in polar coords: {gp.cart2pol(pos_dyn[extra_params['initial_delay'],0,0], pos_dyn[extra_params['initial_delay'],0,1])}")

    p = sig["mic"][:,-seq_len:]
    p_eval = sig["eval"][:,-seq_len:]
    p_dyn = sig["mic_dynamic"]
    p_dyn_cardioid = sig["mic_dynamic_cardioid"]
    if pos_dyn.shape[1] != 1:
        raise NotImplementedError
    pos_dyn = pos_dyn[:,0,:]
    pos_dyn_cardioid = pos_dyn_cardioid[:,0,:]

    sequence = sig["src"][0,extra_params["initial_delay"]:]

    num_ls = arrays["src"].num

    num_freqs = seq_len // num_ls
    real_freqs = ft.get_real_freqs(num_freqs, sim_info.samplerate)
    num_real_freqs = real_freqs.shape[-1]

    rir_eval = arrays.paths["src"]["eval"]
    rir_eval_freq = np.moveaxis(ft.rfft(rir_eval), 1, 2)

    reg_param_td = 1e-8
    
    estimates = {}

    sequence = sequence[:seq_len]

    ds_factor = 1
    p_dyn = p_dyn[:,::ds_factor]
    pos_dyn = pos_dyn[::ds_factor,:]
    p_dyn_cardioid = p_dyn_cardioid[:,::ds_factor]
    pos_dyn_cardioid = pos_dyn_cardioid[::ds_factor,:]
    sequence = sequence[::ds_factor]
    mic_direction = mic_direction[::ds_factor,:]
    rir_eval_freq = rir_eval_freq[::ds_factor,...]
    real_freqs = real_freqs[::ds_factor]

    arrays["eval"].pos = arrays["eval"].pos[::10,:]
    rir_eval_freq = rir_eval_freq[:,::10,:]


    dir_coeffs = shd.directivity_linear(0.5, mic_direction)[None,:,:] #* np.ones((pos_dyn.shape[0], 1))
    #estimates["cardioid jax"] = mm.inf_dimensional_shd_dynamic_compiled(p_dyn_cardioid, pos_dyn_cardioid, arrays["eval"].pos, sequence, sim_info.samplerate, sim_info.c, reg_param_td, dir_coeffs)

    estimates["cardioid numpy"] = mm.inf_dimensional_shd_dynamic(p_dyn_cardioid, pos_dyn_cardioid, arrays["eval"].pos, sequence, sim_info.samplerate, sim_info.c, reg_param_td, dir_coeffs)
    #print(f"MSE between the two versions: {np.mean(np.abs(estimates['cardioid jax'] - estimates['cardioid numpy'])**2)}")
    # print(f"MSE between the omni and cardioid: {np.mean(np.abs(estimates['proposed cardioid'] - estimates['proposed'])**2) / np.mean(np.abs(estimates['proposed'])**2)}")
    

if __name__ == "__main__":
    OUTPUT_METHOD = "pdf"
    sr = 1000
    #multi_speed_experiment(sr)
    #fig_folder = exis.generate_signals_circular_updownsample(800, 2, 1)
    fig_folder = exis.generate_signals_circular_differential_microphone(sr, 1, 2)
    #fig_folder = exis.generate_signals_circular(sr, 1, 4)
    #fig_folder = pathlib.Path("c:/research/code/aspcol/examples/figs/figs_2024_06_26_14_49_0")
    #run_exp(fig_folder)
    #fig_folder = exis.generate_signals_circular(sr, 2, 20)
    #run_exp(fig_folder)
    #fig_folder = pathlib.Path("c:/research/research_documents/202305_moving_mic_spatial_cov_estimation/code/figs/figs_2023_08_02_12_44_0")
    #fig_folder = pathlib.Path("c:/research/research_documents/202305_moving_mic_spatial_cov_estimation/code/figs/figs_2023_07_04_15_31_0")
    

    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    run_exp(fig_folder)

    profiler.stop()
    profiler.print()
