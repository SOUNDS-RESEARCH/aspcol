import numpy as np
import pathlib
import json
import yaml
import scipy.signal as spsig

import aspsim.room.generatepoints as gp

import aspcol.pseq as pseq
import aspcol.soundfieldestimation as sfe
import aspcol.filterdesign as fd
import aspcol.fouriertransform as ft
import aspcol.movingmicrophone as mm
import aspcol.sphericalharmonics as shd
import aspcol.plot as aspplot   

import _exp_funcs_ideal_sampling as exis
import sys

sys.path.append("C:/research/papers/2024_icassp_moving_mic_sf_estimation/code")
import plots as fplot


def run_exp(fig_folder):
    sig, sim_info, arrays, pos_dyn, seq_len, extra_params, pos_dyn_cardioid, mic_direction = exis.load_session(fig_folder)

    print(f"Simulation start position: {pos_dyn[0,0,:]}, in polar coords: {gp.cart2pol(pos_dyn[0,0,0], pos_dyn[0,0,1])}")
    p = sig["mic"][:,-seq_len:]
    p_eval = sig["eval"][:,-seq_len:]
    p_dyn = sig["mic_dynamic"]
    p_dyn_cardioid = sig["mic_dynamic_cardioid"]
    if pos_dyn.shape[1] != 1:
        raise NotImplementedError
    pos_dyn = pos_dyn[:,0,:]
    pos_dyn_cardioid = pos_dyn_cardioid[:,0,:]

    sequence = sig["src"][0,extra_params["initial_delay"]:]

    center = np.array(extra_params["center"])[None,:]
    num_ls = arrays["src"].num

    num_freqs = seq_len // num_ls
    real_freqs = ft.get_real_freqs(num_freqs, sim_info.samplerate)

    rir_eval = arrays.paths["src"]["eval"]
    rir_eval_freq = np.moveaxis(ft.rfft(rir_eval), 1, 2)

    reg_param = 1e-8
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

    dir_coeffs = shd.directivity_linear(0.5, mic_direction)[None,:,:] #* np.ones((pos_dyn.shape[0], 1))
    estimates["cardioid jax"] = mm.inf_dimensional_shd_dynamic_compiled(p_dyn_cardioid, pos_dyn_cardioid, arrays["eval"].pos, sequence, sim_info.samplerate, sim_info.c, reg_param_td, dir_coeffs)
    estimates["proposed omni"] = sfe.est_inf_dimensional_shd_dynamic(p_dyn, pos_dyn, arrays["eval"].pos, sequence, sim_info.samplerate, sim_info.c, reg_param_td)

    estimates["nearest neighbour"] = sfe.pseq_nearest_neighbour(p, sequence[:seq_len], arrays["mic"].pos, arrays["eval"].pos)
    estimates["kernel interpolation"] = sfe.est_ki_diffuse(p, sequence[:seq_len], arrays["mic"].pos, arrays["eval"].pos, sim_info.samplerate, sim_info.c, reg_param)


    fplot.soundfield_estimation_comparison(arrays, estimates, rir_eval_freq, real_freqs, fig_folder, shape="circle", center=center, num_ls=num_ls, output_method=OUTPUT_METHOD)


def highpass_microphone_signal(mic_sig, filter_order, samplerate, filter_below):
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



def test_differential_cardioid_generates_same_data_as_pyroomacoustics_for_far_away_source(sr):
    import matplotlib.pyplot as plt
    folder_diff = exis.generate_signals_circular_differential_microphone(sr, 1, 4)
    sig2, sim_info2, arrays2, pos_dyn2, seq_len2, extra_params2, pos_dyn_cardioid2, mic_direction2 = exis.load_session(folder_diff)
    folder_pra = exis.generate_signals_circular(sr, 1, 4)
    
    sig, sim_info, arrays, pos_dyn, seq_len, extra_params, pos_dyn_cardioid, mic_direction = exis.load_session(folder_pra)
    
    filter_below = 50
    signal_paths = exis.get_signal_paths(folder_diff)
    sig_orig2 = exis.load_npz(signal_paths)
    diff_filt_len = seq_len2
    cardioid_signal = fd.differential_cardioid_microphone(sig_orig2["mic_dynamic"], sig_orig2["mic_diff"], extra_params2["differential distance"], diff_filt_len, sim_info2.c, sim_info2.samplerate, filter_below=filter_below)

    signal_paths = exis.get_signal_paths(folder_pra)
    sig_orig = exis.load_npz(signal_paths)

    cardioid_sig_pra = highpass_microphone_signal(sig_orig["mic_dynamic_cardioid"], diff_filt_len, sim_info.samplerate, filter_below)
    cardioid_sig_pra = cardioid_sig_pra[:, 4 * seq_len2 : 5 * seq_len2]
    cardioid_sig_diff = cardioid_signal[:, 4 * seq_len2 : 5 * seq_len2]

    #sig_orig["mic_dynamic_cardioid"] #= cardioid_signal
    ex_idx = 5
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.squeeze(cardioid_sig_pra[ex_idx,:]), label="Pyroomacoustics")
    ax.plot(np.squeeze(cardioid_sig_diff[ex_idx,:]), label="Differential")
    ax.set_title("Microphone signal from stationary")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()

    fig, axes = plt.subplots(3, 1, sharex=True)
    freqs = ft.get_real_freqs(cardioid_sig_pra.shape[-1], sim_info.samplerate)
    freq_pra = ft.rfft(cardioid_sig_pra[ex_idx,:])
    freq_diff = ft.rfft(cardioid_sig_diff[ex_idx,:])
    axes[0].plot(freqs, np.abs(freq_pra), label="Pyroomacoustics")
    axes[0].plot(freqs, np.abs(freq_diff), label="Differential")
    axes[0].set_ylabel("Magnitude")

    axes[1].plot(freqs, np.real(freq_pra), label="Pyroomacoustics")
    axes[1].plot(freqs, np.real(freq_diff), label="Differential")
    axes[1].set_ylabel("Real part")

    axes[2].plot(freqs, np.imag(freq_pra), label="Pyroomacoustics")
    axes[2].plot(freqs, np.imag(freq_diff), label="Differential")
    axes[2].set_ylabel("Imaginary part")

    for ax in axes:
        ax.set_xlabel("Frequency [Hz]")
        ax.legend()
        aspplot.set_basic_plot_look(ax)

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.squeeze(cardioid_sig_pra[ex_idx,:]), label="Pyroomacoustics")
    ax.plot(np.squeeze(cardioid_sig_diff[ex_idx,:]), label="Differential")
    ax.set_title("Moving microphone signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()
    


if __name__ == "__main__":
    OUTPUT_METHOD = "pdf"
    sr = 1000
    fig_folder = exis.generate_signals_circular_differential_microphone(sr, 1, 10)
    run_exp(fig_folder)

