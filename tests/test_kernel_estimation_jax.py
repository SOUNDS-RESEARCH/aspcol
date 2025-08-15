import numpy as np
import jax.numpy as jnp
import jax
import time

import aspcol.soundfieldestimation_jax as sfe
import aspcol.kernelinterpolation_jax as kernel
import aspcore.fouriertransform_jax as ft
import aspcore.matrices_jax as aspmat

import aspcore.montecarlo_jax as mc
import aspcol.planewaves_jax as pw

import aspcol.kernelinterpolation as kernel_numpy
import aspcore.fouriertransform as ft_numpy


import matplotlib.pyplot as plt

import pyinstrument



def test_uniform_rff_converges_to_the_diffuse_kernel_when_the_number_of_basis_functions_increase():
    rng = np.random.default_rng()
    num1 = 64
    ir_len = 500
    samplerate = 1000  # Hz
    c = 343
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)

    pos = rng.uniform(-1, 1, (num1, 3))
    #pos2 = rng.uniform(-1, 1, (num2, 3))
    key = jax.random.key(654378)

    K = kernel.diffuse_kernel(pos, pos, wave_num)

    # Test RFF convergence
    num_basis_list = [32, 64, 128, 256, 512, 1024]
    error = []
    for num_basis in num_basis_list:
        key, subkey = jax.random.split(key)
        basis_directions = mc.uniform_random_on_sphere(num_basis, subkey)
        Z = pw.plane_wave(pos, basis_directions, wave_num) / np.sqrt(num_basis)
        K_rff =  Z @ np.moveaxis(Z.conj(), 1, 2)

        error.append(np.mean(np.abs(K - K_rff)**2) / np.mean(np.abs(K)**2))

    print(f"RFF convergence error: {error}")
    plt.plot(num_basis_list, error, marker='o')
    plt.xlabel("Number of Basis Functions")
    plt.ylabel("Relative Error")
    plt.title("RFF Convergence to Diffuse Kernel")
    plt.grid()
    plt.show()



def test_vonmisesfisher_rff_converges_to_the_directional_kernel_when_the_number_of_basis_functions_increase():
    rng = np.random.default_rng()
    num1 = 32
    ir_len = 500
    samplerate = 1000  # Hz
    c = 343
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)

    pos = rng.uniform(-1, 1, (num1, 3)) #np.concatenate((np.linspace(-1, 1, num1)[:,None], np.zeros((num1, 2), dtype=float)), axis=-1)
    direction =  rng.uniform(-1, 1, 3) #np.array([1.0,0,0])
    direction /= np.linalg.norm(direction)
    beta = 8.0

    key = jax.random.key(7864584)

    K = kernel.directional_kernel_vonmises(pos, pos, wave_num, direction, beta)

    # Test RFF convergence
    num_basis_list = [32, 64, 128, 256, 512, 1024, 2048]
    error = []
    for num_basis in num_basis_list:
        key, subkey = jax.random.split(key)
        basis_directions = mc.vonmises_fisher_on_sphere(num_basis, direction, beta, subkey)
        Z = pw.plane_wave(pos, basis_directions, wave_num) / np.sqrt(num_basis)
        K_rff =  Z @ np.moveaxis(Z.conj(), 1, 2)
        #K_rff = np.conj(K_rff)

        error.append(np.mean(np.abs(K - K_rff)**2) / np.mean(np.abs(K)**2))

    f_list = [20, 50, 120, 180]

    fig, axes = plt.subplots(len(f_list), 2, figsize=(10, 15))
    for i, f in enumerate(f_list):
        axes[i, 0].matshow(np.abs(K[f,:,:]))
        axes[i, 0].set_title(f"K[{f},:,:]")
        axes[i, 1].matshow(np.abs(K_rff[f,:,:]))
        axes[i, 1].set_title(f"K_rff[{f},:,:]")
    fig.suptitle("Abs of kernel matrices")

        
    fig, axes = plt.subplots(len(f_list), 2, figsize=(10, 15))
    for i, f in enumerate(f_list):
        axes[i, 0].matshow(np.real(K[f,:,:]))
        axes[i, 0].set_title(f"K[{f},:,:]")
        axes[i, 1].matshow(np.real(K_rff[f,:,:]))
        axes[i, 1].set_title(f"K_rff[{f},:,:]")
    fig.suptitle("Real part of kernel matrices")

    fig, axes = plt.subplots(len(f_list), 2, figsize=(10, 15))
    for i, f in enumerate(f_list):
        axes[i, 0].matshow(np.imag(K[f,:,:]))
        axes[i, 0].set_title(f"K[{f},:,:]")
        axes[i, 1].matshow(np.imag(K_rff[f,:,:]))
        axes[i, 1].set_title(f"K_rff[{f},:,:]")
    fig.suptitle("Imaginary part of kernel matrices")


    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    pos_idxs = [10, 12, 25]
    for i, pos_i in enumerate(pos_idxs):
        ax[0,i].plot(np.abs(K[:,10,pos_i]), label='KRR')
        ax[0,i].plot(np.abs(K_rff[:,10,pos_i]), label='RFF')
        ax[1,i].plot(np.real(K[:,10,pos_i]), label='KRR')
        ax[1,i].plot(np.real(K_rff[:,10,pos_i]), label='RFF')
    for ax_lst in ax:
        for a in ax_lst:
            a.legend()

    fig, ax = plt.subplots(1,1)
    mse_per_freq = np.mean(np.abs(K - K_rff)**2, axis=(1,2))
    ax.plot(mse_per_freq, label='MSE per frequency')
    ax.set_xlabel("Frequency Index")
    ax.set_ylabel("Mean Squared Error")
    ax.legend()

    fig, ax = plt.subplots(1,1)
    nmse_per_freq = np.mean(np.abs(K - K_rff)**2, axis=(1,2)) / np.mean(np.abs(K)**2, axis=(1,2))
    ax.plot(nmse_per_freq, label='NMSE per frequency')
    ax.set_xlabel("Frequency Index")
    ax.set_ylabel("Normalized Mean Squared Error")
    ax.legend()

    fig, ax = plt.subplots(1,1)
    print(f"RFF convergence error: {error}")
    ax.plot(num_basis_list, error, marker='o')
    ax.set_xlabel("Number of Basis Functions")
    ax.set_ylabel("Relative Error")
    ax.set_title("RFF Convergence to Diffuse Kernel")
    ax.grid()
    plt.show()









def test_speed_of_directional_kernel_for_jax_verus_numpy():
    rng = np.random.default_rng()
    num1 = 1024
    num2 = 512
    ir_len = 500
    samplerate = 1000  # Hz
    c = 343

    pos1 = rng.uniform(-1, 1, (num1, 3))
    pos2 = rng.uniform(-1, 1, (num2, 3))

    direction = rng.uniform(-1, 1, (3,))
    direction /= np.linalg.norm(direction)
    beta = rng.uniform(0, 10)

    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    wave_num_np = ft_numpy.get_real_wavenum(ir_len, samplerate, c)

    kdir = kernel.directional_kernel_vonmises(pos1, pos2, wave_num, direction, beta).block_until_ready()
    start = time.time()
    kdir = kernel.directional_kernel_vonmises(pos1, pos2, wave_num, direction, beta).block_until_ready()
    jax_time = time.time() - start

    start = time.time()
    kdir_np = kernel_numpy.kernel_directional_3d(pos1, pos2, wave_num_np, direction, beta)
    np_time = time.time() - start

    print(f"JAX directional kernel shape: {kdir.shape}")
    print(f"NumPy directional kernel shape: {kdir_np.shape}")

    print(f"JAX directional kernel time: {jax_time}")
    print(f"NumPy directional kernel time: {np_time}")

    assert np.testing.assert_allclose(kdir, kdir_np, rtol = 1e-5, atol = 1e-5), "JAX and NumPy directional kernels do not match"

    assert False, "This test is not meant to be run in CI, only locally for profiling purposes"


def test_speed_of_directional_kernel_versus_diffuse_kernel():
    """Test that the JAX directional kernel is faster than the numpy one"""
    rng = np.random.default_rng()
    num1 = 1024
    num2 = 512
    ir_len = 500
    samplerate = 1000  # Hz
    c = 343

    pos1 = rng.uniform(-1, 1, (num1, 3))
    pos2 = rng.uniform(-1, 1, (num2, 3))

    direction = rng.uniform(-1, 1, (3,))
    direction /= np.linalg.norm(direction)
    beta = rng.uniform(0, 10)

    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)

    kdir = kernel.directional_kernel_vonmises(pos1, pos2, wave_num, direction, beta).block_until_ready() 

    start = time.time()
    kdir = kernel.directional_kernel_vonmises(pos1, pos2, wave_num, direction, beta).block_until_ready() 
    dir_time = time.time() - start

    kdiff = kernel.diffuse_kernel(pos1, pos2, wave_num).block_until_ready()
    start = time.time()
    kdiff = kernel.diffuse_kernel(pos1, pos2, wave_num).block_until_ready() 
    diff_time = time.time() - start

    print(f"dir shape: {kdir.shape}, diff shape: {kdiff.shape}")

    print(f"JAX directional kernel time: {dir_time}")
    print(f"JAX diffuse kernel time: {diff_time}")

    assert False, "This test is not meant to be run in CI, only locally for profiling purposes"
    #assert dir_time < diff_time, "Directional kernel is not faster than diffuse kernel"


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