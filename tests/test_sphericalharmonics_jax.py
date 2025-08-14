import numpy as np
#from hypothesis import given
from pyinstrument import Profiler
import scipy.special as special

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


import aspcore.fouriertransform as ft
import aspcore.montecarlo as mc

import aspcol.sphericalharmonics as sph_numpy
import aspcol.sphericalharmonics_jax as sph_jax
import aspcol.soundfieldestimation.moving_microphone as sfe_numpy
import aspcol.soundfieldestimation_jax.moving_microphone_jax as sfe_jax
import aspcore.utilities as utils


import matplotlib.pyplot as plt


def _random_shd_coeffs(max_order, num_freqs, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    shd_coeffs = rng.normal(size=(num_freqs, sph_numpy.shd_num_coeffs(max_order))) + \
        1j * rng.normal(size=(num_freqs, sph_numpy.shd_num_coeffs(max_order)))
    return shd_coeffs

def _random_mm_arguments(rng, seq_len, num_periods, max_order, num_eval):
    num_pos = seq_len * num_periods
    num_freqs = seq_len

    p = rng.normal(size=(1, num_pos))
    pos = rng.uniform(-1, 1, size=(num_pos, 3))
    pos_eval = rng.uniform(-1, 1, size=(num_eval, 3))
    dir_coeffs = np.stack([_random_shd_coeffs(max_order, 1, rng) for _ in range(num_pos)], axis=1)
    seq = rng.uniform(-1, 1, size=(seq_len,))
    reg_param = 10 ** rng.uniform(-6, 1)
    return p, pos, pos_eval, seq, reg_param, dir_coeffs

def _random_psi_arguments(rng, seq_len, num_periods, max_order, samplerate, c):
    num_pos = seq_len * num_periods
    num_freqs = seq_len

    pos = rng.uniform(-1, 1, size=(num_pos, 3))
    wave_num = ft.get_real_wavenum(num_freqs, samplerate, c)
    num_real_freqs = wave_num.shape[-1]
    dir_coeffs = np.stack([_random_shd_coeffs(max_order, 1, rng) for _ in range(num_pos)], axis=1)
    seq = rng.uniform(-1, 1, size=(seq_len,))
    Phi = sfe_numpy._sequence_stft_multiperiod(seq, num_periods)[:num_real_freqs,:]
    return pos, wave_num, dir_coeffs, Phi, seq_len, num_real_freqs

def test_compare_compiled_psi_implementations():
    rng = np.random.default_rng(1234567)
    samplerate = 1000
    c = 343
    seq_len = 300
    num_periods = 2
    num_pos = seq_len * num_periods
    num_freqs = seq_len
    max_order = 1
    pos, wave_num, dir_coeffs, Phi, seq_len, num_real_freqs = _random_psi_arguments(rng, seq_len, num_periods, max_order, samplerate, c)

    profiler = Profiler()
    profiler.start()
    psi_jax = sfe_jax.calculate_psi(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs)
    profiler.stop()
    profiler.print()

    profiler = Profiler()
    profiler.start()
    psi_jax = sfe_jax.calculate_psi(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs)
    profiler.stop()
    profiler.print()

    profiler = Profiler()
    profiler.start()
    psi_numpy = sfe_numpy.calculate_psi(pos, dir_coeffs, wave_num, Phi, seq_len, num_real_freqs)
    profiler.stop()
    profiler.print()

    assert np.allclose(psi_numpy, psi_jax)


def test_compare_compiled_inf_dimensional_shd_dynamic_implementations():
    rng = np.random.default_rng(1234567)
    samplerate = 1000
    c = 343
    seq_len = 6
    num_periods = 2
    num_pos = seq_len * num_periods
    num_eval = 20
    num_freqs = seq_len
    max_order = 1
    p, pos, pos_eval, seq, reg_param, dir_coeffs = _random_mm_arguments(rng, seq_len, num_periods, max_order, num_eval)

    profiler = Profiler()
    profiler.start()
    p_eval_jax = sfe_jax.inf_dimensional_shd_dynamic(p, pos, pos_eval, seq, samplerate, c, reg_param, dir_coeffs)
    profiler.stop()
    profiler.print()

    profiler = Profiler()
    profiler.start()
    p_eval_numpy = sfe_numpy.inf_dimensional_shd_dynamic(p, pos, pos_eval, seq, samplerate, c, reg_param, dir_coeffs)
    profiler.stop()
    profiler.print()
    
    assert np.allclose(p_eval_numpy, p_eval_jax)


def test_estimate_from_regressor_is_identical_to_original_estimate():
    rng = np.random.default_rng(1234567)
    seq_len = 20
    num_periods = 2
    num_eval = 15
    max_order = 1
    samplerate = 1000
    c = 343
    p, pos, pos_eval, seq, reg_param, dir_coeffs = _random_mm_arguments(rng, seq_len, num_periods, max_order, num_eval)
    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)

    p_eval, regressor, psi = sfe_jax.inf_dimensional_shd_dynamic(p, pos, pos_eval, seq, samplerate, c, reg_param, dir_coeffs, verbose=True)
    p_eval2 = sfe_jax.estimate_from_regressor(regressor, pos, pos_eval, wave_num, dir_coeffs)
    assert np.allclose(p_eval, p_eval2)


def test_estimate_from_regressor_compiled_and_numpy_version_are_identical():
    rng = np.random.default_rng(1234567)
    seq_len = 20
    num_periods = 2
    num_eval = 15
    max_order = 1
    samplerate = 1000
    c = 343
    p, pos, pos_eval, seq, reg_param, dir_coeffs = _random_mm_arguments(rng, seq_len, num_periods, max_order, num_eval)
    wave_num = ft.get_real_wavenum(seq_len, samplerate, c)

    p_eval_orig, regressor, psi = sfe_numpy.inf_dimensional_shd_dynamic(p, pos, pos_eval, seq, samplerate, c, reg_param, dir_coeffs, verbose=True)

    p_eval_jax = sfe_jax.estimate_from_regressor(regressor, pos, pos_eval, wave_num, dir_coeffs)
    p_eval_numpy = sfe_numpy.estimate_from_regressor(regressor, pos, pos_eval, wave_num, dir_coeffs)
    assert np.allclose(p_eval_jax, p_eval_numpy)




def test_spherical_jn_against_scipy():
    rng = np.random.default_rng(1234567)
    num_points = 10000
    max_val = 200
    order = 2

    n = order * np.ones(num_points, dtype=np.int32)#np.arange(10, dtype=np.int32)
    x = rng.uniform(0, max_val, size=(num_points,))
    sph_jn = sfe_jax.spherical_jn(n, x)
    scipy_jn = special.spherical_jn(n, x)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(x, sph_jn, label="Jax", marker="x", linestyle="")
    axes[0].plot(x, scipy_jn, label="Scipy", marker="x", linestyle="")
    axes[1].plot(x, 20 * np.log10(np.abs(sph_jn - scipy_jn)), label="Difference (dB)", marker="x", linestyle="")
    for ax in axes:
        ax.legend()
        utils.set_basic_plot_look(ax)
    plt.show()
    #assert np.allclose(sph_jn, scipy_jn)


def test_translation_operator_against_jax_compiled_implementation():
    """DONT CHANGE BEFORE FIX
    Currently recreates the problem of jax implemententation not matching nunmpy implementation. Seems to 
    be numerically unstable for small inputs, but very non-obvious how to recreate. 
    """
    rng = np.random.default_rng(21345654)
    #pos = 1e-5 * np.array([[1,-1,0], [1,-1,0], [1,-1,0]]) # #1e-4 * rng.normal(size = (3,3)) 

    # pos =  np.array([[ 9.54350994e-05, -8.09626527e-05,  9.76747952e-05],
    #                 [-1.60965802e-04, -8.47193020e-06,  1.21416007e-04],
    #                 [-3.36724737e-05, -7.74186296e-06, -1.45218513e-04]])
    pos = np.array([[1e-4, -8e-5, 1e-4]])


    freq = np.array([100, 100])#rng.uniform(100, 1000, size=(2,))
    wave_num = 2 * np.pi * freq / 343
    max_order_input = 1
    max_order_output = 1

    gaunt = sph_jax._calculate_gaunt_set(max_order_input, max_order_output)
    T_jax = sph_jax.translation_operator(pos, wave_num, gaunt)
    T = sph_numpy.translation_operator(pos, wave_num, max_order_input, max_order_output)

    T_jax = T_jax[0,0,:,:]
    T = T[0,0,:,:]
    # UNCOMMENT TO VIEW DIFFERENCES
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].set_title("Numpy")
    clr = axes[0].matshow(np.squeeze(np.abs(T)))
    plt.colorbar(clr, ax = axes[0])
    clr = axes[1].matshow(np.squeeze(np.real(T)))
    plt.colorbar(clr, ax = axes[1])
    clr = axes[2].matshow(np.squeeze(np.imag(T)))
    plt.colorbar(clr, ax = axes[2])

    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].set_title("Jax")
    clr = axes[0].matshow(np.squeeze(np.abs(T_jax)))
    plt.colorbar(clr, ax = axes[0])
    clr = axes[1].matshow(np.squeeze(np.real(T_jax)))
    plt.colorbar(clr, ax = axes[1])
    clr = axes[2].matshow(np.squeeze(np.imag(T_jax)))
    plt.colorbar(clr, ax = axes[2])

    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    axes[0].set_title("Difference")
    clr = axes[0].matshow(np.squeeze(np.abs(T_jax-T)))
    plt.colorbar(clr, ax = axes[0])
    clr = axes[1].matshow(np.squeeze(np.real(T_jax-T)))
    plt.colorbar(clr, ax = axes[1])
    clr = axes[2].matshow(np.squeeze(np.imag(T_jax-T)))
    plt.colorbar(clr, ax = axes[2])
    plt.show()
    assert np.allclose(T, T_jax)




def test_find_numerical_problems_in_jax_translation_operator():
    rng = np.random.default_rng(21345654)
    num_points = 300
    num_scales = 100


    test_dirs = mc.uniform_random_on_sphere(num_points, rng)
    scalings = np.logspace(-10, 3, num_scales)
    pos = test_dirs[None,:,:] * scalings[:,None,None]
    #pos = pos.reshape(-1, 3)

    freq = np.array([100])#rng.uniform(100, 1000, size=(2,))
    wave_num = 2 * np.pi * freq / 343
    max_order_input = 1
    max_order_output = 1

    gaunt = sph_jax._calculate_gaunt_set(max_order_input, max_order_output)

    mse = []
    maxse = []
    for i, s in enumerate(scalings):
        T_jax = sph_jax.translation_operator(pos[i,:,:], wave_num, gaunt)
        T = sph_numpy.translation_operator(pos[i,:,:], wave_num, max_order_input, max_order_output)
        mse_s = np.mean(np.abs(T_jax - T)**2, axis=(-1, -2))
        mse.append(np.mean(mse_s))
        maxse.append(np.max(mse_s))
        #mse.append(np.mean(np.abs(T_jax - T)**2) / np.mean(np.abs(T)**2))
        #maxse.append(np.max(np.abs(T_jax - T)**2))

    i = np.argmax(maxse)
    max_bessel_arg = wave_num * scalings[i]

    fig, ax = plt.subplots(1, 1)
    ax.plot(wave_num*scalings, mse, label="MSE")
    ax.plot(wave_num*scalings, maxse, label="MaxSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bessel argument")
    ax.set_ylabel("Error")
    ax.set_title(f"Max error at {max_bessel_arg}")
    ax.legend()
    plt.show()







def test_spherical_bessel_function_numerical_stability():
    rng = np.random.default_rng(21345654)
    num_points = 3000

    arg = np.logspace(-10, 3, num_points)
    max_order = 2

    jn_jax = []
    jn_scipy = []
    for i in range(max_order+1):
        jn_scipy.append(special.spherical_jn((i * np.ones_like(arg)).astype(int), arg))
        jn_jax.append(sph_jax.spherical_jn((i * np.ones_like(arg)).astype(int), arg))
    

    error = [jn_j - jn_s for jn_j, jn_s in zip(jn_jax, jn_scipy)]

    fig, axes = plt.subplots(max_order+1, 2, sharex=True, figsize=(16, 9))

    for i in range(max_order+1):
        axes[i][0].plot(arg, jn_jax[i], label="Jax")
        axes[i][0].plot(arg, jn_scipy[i], label="Scipy")
        axes[i][1].plot(arg, np.abs(error[i]), label="Error")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel("Amplitude")
            utils.set_basic_plot_look(ax)
            ax.legend()
    axes[0,0].set_title("Function values")
    axes[0,1].set_title("Error")
    axes[-1,0].set_xlabel("Bessel argument")
    axes[-1,1].set_xlabel("Bessel argument")
    plt.show()



def test_bessel_function_integer_order_numerical_stability():
    rng = np.random.default_rng(21345654)
    num_points = 3000

    arg = np.logspace(-10, -3, num_points)
    max_order = 2

    jn_jax = []
    jn_scipy = []
    jn_jax.append(sph_jax.j0(arg))
    jn_jax.append(sph_jax.j1(arg))
    #or i in range(2, max_order+1):
    #   jn_jax.append(mm.jv(i, arg))

    #jn_jax.append(arg**2 / 15)


    for i in range(max_order+1):
        jn_scipy.append(special.jv((i * np.ones_like(arg)).astype(int), arg))
        #jn_scipy.append(special.jv((i * np.ones_like(arg)).astype(int), arg))

    error = [jn_j - jn_s for jn_j, jn_s in zip(jn_jax, jn_scipy)]

    num_orders = len(jn_jax)
    fig, axes = plt.subplots(num_orders, 2, sharex=True, figsize=(16, 9))

    for i in range(num_orders):
        axes[i][0].plot(arg, jn_jax[i], label="Jax")
        axes[i][0].plot(arg, jn_scipy[i], label="Scipy")
        axes[i][1].plot(arg, np.abs(error[i]), label="Error")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel("Amplitude")
            utils.set_basic_plot_look(ax)
            ax.legend()
    axes[0,0].set_title("Function values")
    axes[0,1].set_title("Error")
    axes[-1,0].set_xlabel("Bessel argument")
    axes[-1,1].set_xlabel("Bessel argument")
    plt.show()



def test_bessel_function_half_integer_order_numerical_stability():
    rng = np.random.default_rng(21345654)
    num_points = 3000

    arg = np.logspace(-10, 6, num_points)
    max_order = 2

    jn_jax = []
    jn_scipy = []

    #for i in range(max_order+1):
    #    jn_scipy.append(special.jv(((0.5+i) * np.ones_like(arg)).astype(int), arg))
        #jn_scipy.append(special.jv((i * np.ones_like(arg)).astype(int), arg))

    jn_jax.append(sph_jax.jneghalf(arg))
    jn_jax.append(sph_jax.jposhalf(arg))
    jn_jax.append(sph_jax.jvplushalf(2, arg))

    jn_scipy.append(special.jv(-0.5, arg))
    jn_scipy.append(special.jv(0.5, arg))
    jn_scipy.append(special.jv(2.5, arg))

    error = [jn_j - jn_s for jn_j, jn_s in zip(jn_jax, jn_scipy)]

    num_orders = len(jn_jax)
    fig, axes = plt.subplots(num_orders, 2, sharex=True, figsize=(16, 9))

    for i in range(num_orders):
        axes[i][0].plot(arg, jn_jax[i], label="Jax")
        axes[i][0].plot(arg, jn_scipy[i], label="Scipy")
        axes[i][1].plot(arg, np.abs(error[i]), label="Error")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel("Amplitude")
            utils.set_basic_plot_look(ax)
            ax.legend()
    axes[0,0].set_title("Function values")
    axes[0,1].set_title("Error")
    axes[-1,0].set_xlabel("Bessel argument")
    axes[-1,1].set_xlabel("Bessel argument")
    plt.show()