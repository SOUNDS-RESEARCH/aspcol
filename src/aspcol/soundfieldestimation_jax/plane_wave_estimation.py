"""Sound field estimation using plane wave basis expansion methods


"""
import jax.numpy as jnp
import jax
import numpy as np



import aspcol.planewaves_jax as pw
import aspcore.montecarlo_jax as mc
import aspcore.linear_systems_jax as ls



def find_best_reg_param_pw_tikhonov(p_freq, pos, wave_num, reg_params, order = 7):
    """Finds the best regularization parameter for plane wave Tikhonov estimation using GCV

    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    wave_num : ndarray of shape (num_freq)
        wavenumbers, defined as 2 * pi * freq / c
    reg_params : array of shape (k,)
        candidate regularization parameters
    order : int
        order of the t-design to use as basis directions

    Returns
    -------
    best_reg_params : ndarray of shape (num_real_freqs,)
        best regularization parameter for each frequency
    scores : ndarray of shape (num_real_freqs, k)
        GCV scores for each frequency and candidate regularization parameter
    """
    basis_directions = pw.t_design(order)
    Z = pw.plane_wave(pos, basis_directions, wave_num)

    scores = jnp.stack([jax.vmap(ls.gcv_score, in_axes=(0, 0, None))(Z, p_freq, rp) for rp in reg_params], axis=-1)
    scores = jnp.mean(scores, axis=0)

    best_idx = jnp.argmin(scores, axis=-1)
    best_reg_params = reg_params[best_idx]
    return best_reg_params, scores

def est_pw_tikhonov(p_freq, pos, pos_eval, wave_num, reg_param, order = 7):
    """Estimation using least squares estimation with Tikhonov regularization

    Estimation is performed in the frequency domain for each frequency separately
    For now the directions are random, but will be changed to some approptiate quadrature. 
    Use RFF for random directions

    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    wave_num : ndarray of shape (num_freq)
        wavenumbers, defined as 2 * pi * freq / c
    reg_param : float
        regularization parameter for the least squares estimation
    order : int
        order of the t-design to use as basis directions

    Returns
    -------
    p_est : ndarray of shape (num_real_freqs, num_eval)
        estimated sound pressure at the evaluation positions

    """
    basis_directions = pw.t_design(order)
    num_basis = basis_directions.shape[0]

    Z = pw.plane_wave(pos, basis_directions, wave_num)
    system_mat = jnp.moveaxis(Z.conj(), 1, 2) @ Z
    system_mat = system_mat + reg_param * jnp.eye(num_basis, dtype=system_mat.dtype)[None,...]

    projected_data = jnp.moveaxis(Z.conj(),1,2) @ p_freq[:,:,None]
    params = jax.scipy.linalg.solve(system_mat, projected_data, assume_a='pos')

    z_eval = pw.plane_wave(pos_eval, basis_directions, wave_num)
    p_est = z_eval @ params
    return jnp.squeeze(p_est, axis=-1)



def est_pw_irls(p_freq, pos, pos_eval, wave_num, snr=None, order = 11):
    """Estimation using l1 penalty with IRLS

    Estimation is performed in the frequency domain for each frequency separately
    For now the directions are random, but will be changed to some approptiate quadrature. 
    Use RFF for random directions

    p_freq : ndarray of shape (num_real_freqs, num_mics)
        sound pressure in frequency domain at num_mic microphone positions
    pos : ndarray of shape (num_mic, 3)
        positions of the microphones
    pos_eval : ndarray of shape (num_eval, 3)
        positions of the evaluation points
    wave_num : ndarray of shape (num_freq)
        wavenumbers, defined as 2 * pi * freq / c
    snr : float or None
        estimated signal to noise ratio. If None, no regularization is performed. Supplying an snr
        is highly recommended, it usually improves the results significantly. If the SNR is not known, consider
        using an arbitrary high value (corresponding to a very small regularization), which will at
        least stabilize from numerical noise. 
    order : int
        order of the t-design to use as basis directions

    Returns
    -------
    p_est : ndarray of shape (num_real_freqs, num_eval)
        estimated sound pressure at the evaluation positions

    """
    basis_directions = pw.t_design(order)
    Z = pw.plane_wave(pos, basis_directions, wave_num)

    if snr is not None:
        params, l1_norm, residual = jax.vmap(ls.irls_reg, in_axes=(0,0,None))(Z, p_freq, snr)
    else:
        params, l1_norm, residual = jax.vmap(ls.irls, in_axes=(0,0))(Z, p_freq)
        params = params.at[0,:].set(0)

    z_eval = pw.plane_wave(pos_eval, basis_directions, wave_num)
    p_est = jnp.squeeze(z_eval @ params[...,None], axis=-1)
    return p_est