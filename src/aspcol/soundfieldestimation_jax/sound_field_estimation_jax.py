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

import aspcore.fouriertransform_jax as ft
import aspcore.montecarlo_jax as mc

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
        K = ki.diffuse_kernel(pos, pos, wave_num)
        k_est = ki.diffuse_kernel(pos_eval, pos, wave_num)
    else:
        assert beta is not None, "beta must be set if direction is set"
        K = ki.directional_kernel_vonmises(pos, pos, wave_num, direction, beta)
        k_est = ki.directional_kernel_vonmises(pos_eval, pos, wave_num, direction, beta)
    
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


