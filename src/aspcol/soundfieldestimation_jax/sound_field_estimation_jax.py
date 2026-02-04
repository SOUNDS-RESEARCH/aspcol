"""Collection of algorithms for sound field estimation implemented in JAX

Follows the same API as the numpy versions of the algorithms wherever possible / convenient. Sometimes the methods in this module are more restricted or with a slightly difference API. 

* Kernel interpolation [uenoKernel2018]
* Infinite dimensional spherical harmonic analysis for moving microphones [brunnstromBayesianSubmitted]
* Spatial spectrum estimation for moving microphones [katzbergSpherical2021]

References
----------
[uenoKernel2018] N. Ueno, S. Koyama, and H. Saruwatari, “Kernel ridge regression with constraint of Helmholtz equation for sound field interpolation,” in 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), Tokyo, Japan: IEEE, Sep. 2018, pp. 436–440. doi: 10.1109/IWAENC.2018.8521334. `[link] <https://doi.org/10.1109/IWAENC.2018.8521334>`__ \n
[brunnstromBayesianSubmitted] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, submitted. \n
[katzbergSpherical2021] F. Katzberg, M. Maass, and A. Mertins, “Spherical harmonic representation for dynamic sound-field measurements,” in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Jun. 2021, pp. 426–430. doi: 10.1109/ICASSP39728.2021.9413708. `[link] <https://doi.org/10.1109/ICASSP39728.2021.9413708>`__ \n
"""
#import numpy as np
#import scipy.spatial.distance as spdist
import jax.numpy as jnp
import jax
from functools import partial

import aspcore.fouriertransform_jax as ft
import aspcore.montecarlo_jax as mc
import aspcore.filterdesign_jax as fd

import aspcol.kernelinterpolation_jax as ki
#import aspcol.sphericalharmonics_jax as sph
import aspcol.planewaves_jax as pw





#============= FREQUENCY DOMAIN METHODS - STATIONARY MICROPHONES =============
@jax.jit
def est_ki_freq(p_freq, pos, pos_eval, wave_num, reg_param, direction = None, beta = None):
    """Estimates the RIR in the frequency domain using kernel interpolation
    
    Uses the frequency domain sound pressure as input

    Parameters
    ----------
    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    wave_num : ndarray of shape (num_real_freqs)
        wavenumbers
    reg_param : float
        regularization parameter for kernel interpolation

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    References
    ----------
    [uenoKernel2018]
    """
    num_pos = pos.shape[0]

    if direction is None:
        assert beta is None, "beta must be None if direction is None"
        K = ki.kernel_diffuse(pos, pos, wave_num)
        k_est = ki.kernel_diffuse(pos_eval, pos, wave_num)
    else:
        assert beta is not None, "beta must be set if direction is set"
        K = ki.kernel_directional_vonmises(pos, pos, wave_num, direction, beta)
        k_est = ki.kernel_directional_vonmises(pos_eval, pos, wave_num, direction, beta)
    
    reg_matrix = reg_param * jnp.eye(num_pos, dtype=K.dtype)[None,...]
    a = jnp.linalg.solve(K + reg_matrix, p_freq[...,None])
    p_est = jnp.squeeze(k_est @ a, axis=-1)
    return p_est


def est_ki_freq_rff(p_freq, pos, pos_eval, wave_num, reg_param, num_basis = 64, key = None, direction=None, beta = None):
    """Estimates the RIR in the frequency domain using random Fourier features
    
    Uses the frequency domain sound pressure as input

    Parameters
    ----------
    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    k : ndarray of shape (num_freq)
        wavenumbers
    reg_param : float
        regularization parameter for kernel interpolation
    num_basis : int
        number of basis directions to use for the random fourier features
    direction : ndarray of shape (3,)
        direction of the directional weighting. 
        This should be towards the source, i.e. in the opposite of the propagation direction
    beta : float
        strength of the directional component

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    Notes
    -----
    If direction is set, the basis directions are sampled from a von Mises-Fisher distribution, 
    with pdf p(x) = e^{-beta * direction^T x}.

    References
    ----------
    [uenoKernel2018]
    """
    if key is None:
        key = jax.random.key(0)

    if direction is None:
        assert beta is None, "beta must be None if direction is None"
        basis_directions = mc.uniform_random_on_sphere(num_basis, key)
    else:
        assert beta is not None, "beta must be set if direction is set"
        basis_directions = mc.vonmises_fisher_on_sphere(num_basis, -direction, beta, key)

    Z = pw.plane_wave(pos, basis_directions, wave_num) / jnp.sqrt(num_basis)
    system_mat = jnp.moveaxis(Z.conj(), 1, 2) @ Z
    system_mat += reg_param * jnp.eye(num_basis, dtype=system_mat.dtype)[None,...]

    projected_data = jnp.moveaxis(Z.conj(),1,2) @ p_freq[:,:,None]
    params = jnp.linalg.solve(system_mat, projected_data)

    z_eval = pw.plane_wave(pos_eval, basis_directions, wave_num) / jnp.sqrt(num_basis)
    p_est = z_eval @ params
    return jnp.squeeze(p_est, axis=-1)





#@jax.jit
def est_ki_freq_multisrc(p_freq, pos, pos_eval, wave_num, reg_param, src_weighting = None):
    """Estimates the RIR in the frequency domain using kernel interpolation
    
    Uses the frequency domain sound pressure as input

    Parameters
    ----------
    p_freq : ndarray of shape (num_real_freqs, num_src, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    wave_num : ndarray of shape (num_real_freqs)
        wavenumbers
    reg_param : float
        regularization parameter for kernel interpolation

    Returns
    -------
    est_sound_pressure : ndarray of shape (num_real_freqs, num_eval)
        estimated RIR per frequency at the evaluation points

    References
    ----------
    [uenoKernel2018]
    """
    num_pos = pos.shape[0]
    num_src = p_freq.shape[1]

    krr_params = ki.get_krr_params_multisrc(p_freq, pos, wave_num, reg_param, ki.kernel_multisrc, [num_src, src_weighting])
    p_est = ki.reconstruct_multisrc(krr_params, pos_eval, pos, wave_num, ki.kernel_multisrc, [num_src, src_weighting])
    return p_est





#@jax.jit(static_argnames=['ir_len', 'extra_delay'])
@partial(jax.jit, static_argnames=['ir_len', 'extra_delay'])
def free_space_ir(pos_src, pos_mic, samplerate, c, ir_len, extra_delay):
    """Computes the free space impulse response between sources and microphones

    Parameters
    ----------
    pos_src : ndarray of shape (num_src, 3)
        positions of the sources
    pos_mic : ndarray of shape (num_mic, 3)
        positions of the microphones
    samplerate : int
        sampling rate of the impulse response
    c : float
        speed of sound
    extra_delay : int
        extra delay in addition to the propagation delay. The fractional delay filter is 
        of order extra_delay * 2 + 1, so a higher value gives a better filter approximation. 

    Returns
    -------
    ir : ndarray of shape (num_src, num_mic, ir_len)
        free space impulse response between sources and microphones
    """
    num_src = pos_src.shape[0]
    num_mic = pos_mic.shape[0]

    dists = jnp.linalg.norm(pos_src[:,None,:] - pos_mic[None,:,:], axis=-1) # shape (num_src, num_mic)
    delay_samples = samplerate * dists / c # shape (num_src, num_mic)
    integer_delay = jnp.floor(delay_samples).astype(jnp.int32)  # shape (num_src, num_mic)
    frac_delay = delay_samples - integer_delay

    even = False
    if ir_len % 2 == 0:
        ir_len -= 1  # make odd
        even = True

    frac_len = extra_delay * 2 + 1
    total_len = ir_len + frac_len

    irs_frac = jax.vmap(fd.frac_dly_windowed_sinc, in_axes=(0,None))(frac_delay.reshape(-1), frac_len)  # shape (num_src, num_mic, frac_len)
    irs_frac = irs_frac.reshape((num_src, num_mic, -1))  # shape (num_src, num_mic, frac_len)
    ir_frac = irs_frac / (4 * jnp.pi * dists[...,None])  # shape (num_src, num_mic, frac_len)

    def place_frac_ir(frac_ir, start_idx):
        ir = jnp.zeros((total_len,))
        return jax.lax.dynamic_update_slice(ir, frac_ir, (start_idx,))

    irs = jax.vmap(jax.vmap(place_frac_ir))(irs_frac, integer_delay)

    if even:
        irs = jnp.concatenate([irs, jnp.zeros((num_src, num_mic, 1))], axis=-1)

    irs = irs[..., :-frac_len]
    return irs


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(1234)
    pos_src = rng.uniform(-1,1,(1,3)) + np.array([2, 0, 0])
    pos_mic = np.zeros((100, 3))
    pos_mic[:,1] = np.linspace(0,4,100)

    samplerate = 2000
    c = 343
    ir_len = 4000
    extra_delay = 2800
    ir = free_space_ir(pos_src, pos_mic, samplerate, c, ir_len, extra_delay)

    plt.imshow(np.log10(np.abs(ir[0,:,:])), aspect='auto')
    plt.title("Free space impulse responses")
    plt.xlabel("Time (samples)")
    plt.ylabel("Microphone index")
    plt.colorbar()
    plt.show()

