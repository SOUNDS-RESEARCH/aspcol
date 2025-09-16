import numpy as np
import copy

import aspcore.fouriertransform as ft

import aspcol.soundfieldcontrol as sfc
import aspcol.plot as aspplot
import aspcol.kernelinterpolation as ki
import aspcol.spatialcovarianceestimation as sce

import _generate_data as gd

def main(cov_snr = 10):
    rng = np.random.default_rng(6543456)
    num_mic = 4
    num_src = 8
    num_dense = 64
    rt60 = (0.3, 0.3)
    snr = 40
    eval_spacing = 0.1
    cov_snr = cov_snr
    sr = 1600
    c = (343, 343)
    #sim, ir_data, ir_eval, ir_dense, wave_num, freqs = gd.generate_data(num_mic, num_src, num_dense, rt60, snr, eval_spacing=eval_spacing)

    sims, setup, ir_data_bright, ir_data_dark = gd.generate_sound_zone_data(num_mic, num_src, num_dense, sr, rt60, snr, c, eval_spacing = eval_spacing, output_method=PLOT_METHOD)
    
    freqs = ft.get_real_freqs(sims[-1].sim_info.max_room_ir_length, sr)
    wave_num = ft.get_real_wavenum(sims[-1].sim_info.max_room_ir_length, sr, sims[-1].sim_info.c)

    sim = sims[-1]
    ir_data = ir_data_bright
    ir_eval = ft.rfft(sims[-1].arrays.paths["src"]["bright_eval"])
    ir_dense = ft.rfft(sims[-1].arrays.paths["src"]["bright_dense"])

    cov_true = sfc.spatial_cov_freq(ir_eval)

    cov_data_no_noise = sfc.spatial_cov_freq(ir_dense)
    noise_cov = gd.generate_covariance_noise(cov_data_no_noise, cov_snr, rng)
    cov_data = cov_data_no_noise + noise_cov

    #covest.plot_spatial_covariance({"cov_data" : cov_data}, cov_data_no_noise, freqs, sim.folder_path, plot_method=PLOT_METHOD)

    num_freqs = freqs.shape[-1]
    num_extra = 10 # V in the paper
    if num_extra > 0:
        pos_extra = sim.arrays["bright_eval"].region.sample_points(num_extra)
        pos_total = np.concatenate([sim.arrays["bright_mic"].pos, pos_extra], axis=0)
    else:
        pos_total = sim.arrays["bright_mic"].pos

    estimates = {}
    soundfield = {}
    
    spatial_cov_mc_samples = 1000
    steps = 10000
    lr = 2e-3

    reg_param = 1e-5
    cov_reg_param = np.ones((num_freqs)) * 1e-3

    krr_params = {}
    used_pos = {}
    krr_params["KRR"] = np.stack([ki.get_krr_params(ir_data[:,i,:], sim.arrays["bright_mic"].pos, wave_num, reg_param, ki.kernel_diffuse, []) for i in range(sim.arrays["src"].num)], axis=1)
    used_pos["KRR"] = sim.arrays["bright_mic"].pos

    krr_params[f"CIKRR frobenius"] = sce.krr_estimation_cov_informed(ir_data, pos_total, wave_num, reg_param, copy.deepcopy(sim.arrays["bright_eval"].region.sample_points), sim.arrays["bright_eval"].region.volume, spatial_cov_mc_samples, cov_true, cov_reg_param, cost_func=sce._cov_informed_krr_cost_frobenius, num_steps=steps, learning_rate = lr)
    used_pos[f"CIKRR frobenius"] = pos_total

    krr_params[f"CIKRR wasserstein"] = sce.krr_estimation_cov_informed(ir_data, pos_total, wave_num, reg_param, copy.deepcopy(sim.arrays["bright_eval"].region.sample_points), sim.arrays["bright_eval"].region.volume, spatial_cov_mc_samples, cov_true, cov_reg_param, cost_func=sce._cov_informed_krr_cost_wasserstein, num_steps=steps, learning_rate = lr)
    used_pos[f"CIKRR wasserstein"] = pos_total

    for krr_name, param_est in krr_params.items():
        soundfield[krr_name] = np.stack([ki.reconstruct_freq(param_est[:,i,:], sim.arrays["bright_eval"].pos, used_pos[krr_name], wave_num, ki.kernel_diffuse, []) for i in range(sim.arrays["src"].num)], axis=1)
        estimates[krr_name] = sfc.spatial_cov_freq_kernel(param_est, used_pos[krr_name], wave_num, copy.deepcopy(sim.arrays["bright_eval"].region.sample_points), sim.arrays["bright_eval"].region.volume, spatial_cov_mc_samples)

    estimates["original data"] = cov_data
    estimates["sample covariance"] = sfc.spatial_cov_freq(ir_data)

    aspplot.evaluate_freq_spatial_cov(estimates, cov_true, sim.folder_path, freqs, PLOT_METHOD)
    aspplot.soundfield_estimation_comparison(sim.arrays, soundfield, ir_eval, freqs, sim.folder_path, output_method=PLOT_METHOD, num_ls = sim.arrays["src"].num)


if __name__ == "__main__":
    PLOT_METHOD = "tikz"
    main(np.inf)