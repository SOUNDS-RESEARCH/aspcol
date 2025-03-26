import numpy as np
import jax.numpy as jnp
import jax

import aspcol.soundfieldestimation_jax as sfe
import aspcol.kernelinterpolation_jax as kernel
import aspcore.fouriertransform_jax as ft
import aspcore.matrices_jax as aspmat

import matplotlib.pyplot as plt

import pyinstrument


def reconstruct_diffuse_without_map(pos_eval, pos_mic, wave_num, krr_params):
    """See docs for reconstruct_diffuse
    This one is correct, but takes a ton of memory for many evaluation points
    """
    num_mic = pos_mic.shape[0]
    
    if krr_params.ndim == 1:
        krr_params = krr_params.reshape(num_mic, -1)
    assert krr_params.ndim == 2

    gamma_eval = kernel.time_domain_diffuse_kernel(pos_eval, pos_mic, wave_num)
    #estimate_each_mic = jnp.stack([gamma_eval[:,m,...] @ krr_params[m,:] for m in range(num_mic)], axis=0)
    estimate = jnp.squeeze(aspmat.matmul_param(gamma_eval, krr_params[:,None,:,None]), axis=(1,3))

    # also works, but takes longer to compile
    # estimates = []
    # for i in range(num_eval):
    #     gamma_eval = kernel.time_domain_diffuse_kernel(pos_eval[i:i+1,:], pos_mic, wave_num)
    #     estimate_each_mic = jnp.squeeze(aspmat.matmul_param(gamma_eval, krr_params[:,None,:,None]), axis=(1,3))
    #     estimates.append(estimate_each_mic)
    # estimate = jnp.concatenate(estimates, axis=0)
    return estimate



def test_diffuse_reconstruction_is_identical_with_and_without_map():
    rng = np.random.default_rng()
    ir_len = 512
    num_data = 5
    num_eval = 100

    pos_eval = rng.uniform(-1, 1, (num_eval, 3))
    pos_mic = rng.uniform(-1, 1, (num_data, 3))

    wave_num = ft.get_real_wavenum(ir_len, 1000, 343)
    krr_params = rng.uniform(-1, 1, (num_data, ir_len))

    #with jax.disable_jit():
    _ = sfe.reconstruct_diffuse(pos_eval, pos_mic, wave_num, krr_params).block_until_ready()
    _2 = reconstruct_diffuse_without_map(pos_eval, pos_mic, wave_num, krr_params).block_until_ready()

    profiler = pyinstrument.Profiler()
    profiler.start()
    estimate = sfe.reconstruct_diffuse(pos_eval, pos_mic, wave_num, krr_params)
    estimate.block_until_ready()
    profiler.stop()
    profiler.print()

    profiler = pyinstrument.Profiler()
    profiler.start()
    estimate_no_map = reconstruct_diffuse_without_map(pos_eval, pos_mic, wave_num, krr_params)
    estimate_no_map.block_until_ready()
    profiler.stop()
    profiler.print()
    
    #assert False
    assert jnp.allclose(estimate, estimate_no_map, atol=1e-5)