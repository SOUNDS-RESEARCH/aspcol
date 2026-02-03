import numpy as np
import jax.numpy as jnp
import jax
from functools import partial

import aspcore.fouriertransform_jax as ft
import aspcol.kernelinterpolation_jax.kernel as kernel
import aspcol.kernelinterpolation_jax.kernel_multifreq as kernel_multifreq


@partial(jax.jit, static_argnames=["real_nyquist"])
def kernel_time_domain_diffuse(pos1, pos2, wave_num, real_nyquist=False):
    """Time domain diffuse sound field kernel. 

    Assumes the total DFT length was even. Any number of real frequencies / wave numbers can 
    represent both an odd and even number of frequencies. 

    Defined for each position pair as F^{-1} Gamma(r, r') F, where Gamma(r, r') is the multifrequency kernel, and
    F and F^{-1} are the real DFT and inverse DFT transforms. 

    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first point.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second point.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    real_nyquist : bool, optional
        If True, the kernel is calculated as 1/2 * k(r, r') + k(r, -r') for the highest frequency bin. This is
        the kernel derived in [brunnströmTime2025] for even DFT lengths. It should be set to False for odd DFT lengths,
        but it is also not likely to make a big difference, as the sound field model for the Nyquist frequency is 
        non-physical anyway. Default is False.

    Returns
    -------
    np.ndarray of shape (num_points1, num_points2, dft_len, dft_len)
        The time domain kernel matrix. 
        The dft_len is assumed to be even, and the number of real frequencies is dft_len//2 + 1.
    
    Notes
    -----
    This function can be substantially optimized by implementing in terms of only the real frequencies and the FFT rather
    than DFT matrices. This is left for future work, whereas this is clear and easy to check for correctness. 
    """
    freq_kernel = kernel_multifreq.kernel_multifreq_diffuse(pos1, pos2, wave_num, diag_mat=False)

    if real_nyquist:
        nyquist_extra_kernel = kernel.kernel_diffuse(pos1, -pos2, wave_num[-1])[0,...]
        freq_kernel.at[..., -1].add(nyquist_extra_kernel)
        freq_kernel.at[..., -1].multiply(0.5)

    kernel_matrix = freq_to_time_domain_kernel_matrix(freq_kernel)
    return kernel_matrix


def kernel_time_domain_directional_vonmises(pos1, pos2, wave_num, direction, beta):
    """Time-domain directional sound field kernel. 

    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first point.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second point.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    direction : np.ndarray of shape (3,1)
        The direction of the directional weighting.
    beta : float
        The strength of the directional weighting. A larger value will give more regularization.

    Returns
    -------
    np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs)
        The kernel matrix.

    References
    ----------
    [uenoDirectionally2021]
    [brunnströmTime2025]
    """
    kernel_val = kernel.kernel_directional_vonmises(pos1, pos2, wave_num, direction, beta)
    #kernel_val = jnp.squeeze(kernel_val, axis=1)
    kernel_val = jnp.moveaxis(kernel_val, 0, -1)

    kernel_matrix = freq_to_time_domain_kernel_matrix(kernel_val)
    return kernel_matrix


def freq_to_time_domain_kernel_matrix(freq_kernel):
    """Turns a diagonal frequency domain kernel matrix into a time domain kernel matrix.
    
    Parameters
    ----------
    freq_kernel : np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs) or
        np.ndarray of shape (num_points1, num_points2, num_real_freqs)
        The kernel matrix. Assumed to be diagonal

    Returns
    -------
    np.ndarray of shape (num_points1, num_points2, dft_len, dft_len)
        The time domain kernel matrix. 
        The dft_len is assumed to be even, and the number of real frequencies is dft_len//2 + 1.
    """

    assert freq_kernel.ndim == 4 or freq_kernel.ndim == 3
    if freq_kernel.ndim == 3:
        diag_mat = False
    else:
        diag_mat = True
    dft_len = freq_kernel.shape[-1] * 2 - 2

    if diag_mat:
        freq_kernel = jnp.diagonal(freq_kernel, axis1=-2, axis2=-1)
    a_ext = ft.insert_negative_frequencies(freq_kernel.T, even=True).T
    a_mat = jnp.eye(dft_len)[None,None,...] * a_ext[...,None,:]

    # The FFT is the correct fast way to do this 
    #kernel_matrix = np.fft.fft(np.fft.ifft(a_mat, axis=-2), axis=-1)
    # for consistency, we use the other time-convention as defined by aspcol
    b = jnp.moveaxis(ft.ifft(jnp.moveaxis(a_mat,-2, 0)), -1, 2)
    kernel_matrix = jnp.moveaxis(ft.fft(b), 0, -1)
    
    # Below is a more readable version 
    #F = splin.dft(dft_len)
    #Finv = F.conj().T / dft_len
    #kernel_matrix = Finv[None,None,...] @ a_mat @ F[None,None,...]

    kernel_matrix = jnp.real(kernel_matrix)
    return kernel_matrix



@jax.jit
def kernel_time_domain_envelope(pos1, pos2, wave_num, envelope_reg, reg_points):
    """The kernel Gamma_r(r, r') of the time domain diffuse sound field with envelope regularization.

    This is regularization option 2 in [brunnströmTime2025], which is constructed as a regularization
    at a finite set of points. 

    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first set of points.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second set of points.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    envelope_reg : np.ndarray of shape (dft_len,) or (num_reg_points, dft_len)
        The envelope regularization weighting. The values must be positive and real-valued. The parameter
        represents the diagonal values of D^H D in [brunnströmTime2025].
    reg_points : np.ndarray of shape (num_reg_points, 3)
        The regularization points. These are the points where the regularization is applied.
        num_reg_points is V in [brunnströmTime2025].

    Returns
    -------
    np.ndarray of shape (num_points1, num_points2, dft_len, dft_len)
        The time domain kernel matrix. 
        The dft_len is assumed to be even, and the number of real frequencies is dft_len//2 + 1.

    References
    ----------
    [brunnströmTime2025]
    """
    num_reg_points = reg_points.shape[0]
    if envelope_reg.ndim == 1:
        envelope_reg = envelope_reg[None,:]

    NUM_EACH_LOOP = 10
    num_loops = int(np.ceil(envelope_reg.shape[0] / NUM_EACH_LOOP))

    kernel_mat = jnp.zeros((pos1.shape[0], pos2.shape[0], envelope_reg.shape[-1], envelope_reg.shape[-1]))
    for i in range(num_loops):
        gamma1 = kernel_time_domain_diffuse(pos1, reg_points[i*NUM_EACH_LOOP:(i+1)*NUM_EACH_LOOP,:], wave_num)
        gamma2 = envelope_reg[:,None,:,None] * kernel_time_domain_diffuse(reg_points[i*NUM_EACH_LOOP:(i+1)*NUM_EACH_LOOP,:], pos2, wave_num)
    #gamma1 = gamma1 @ jnp.diag(envelope_reg)[None,None,:,:]
        kernel_mat = kernel_mat + _matmul_param(gamma1, gamma2) #/ (num_reg_points**2)
    return kernel_mat / num_reg_points


@jax.jit
def time_domain_frequency_weighted_kernel(pos1, pos2, wave_num, weighting_mat):
    """
    
    Corresponds to the directional kernel, if the directional weighting is chosen to be a
    constant matrix (independent of direction). Then, only a weighting which couples the different frequencies 
    is introduced. 

    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first set of points.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second set of points.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    weighting_mat : np.ndarray of shape (num_real_freqs, num_real_freqs)
        The weighting matrix. Must be positive semi-definite, and corresponds to W^H W in the paper. 
    """
    sinc_arg = jnp.linalg.norm(wave_num[None,None,None,:,None] * pos1[:,None,:,None,None] - wave_num[None,None,None,None,:] * pos2[None,:,:,None,None], axis=2)
    kernel_val = jnp.sinc(sinc_arg / np.pi)

    weighted_kernel = weighting_mat[None,None,:,:] * kernel_val
    K = freq_to_time_domain_kernel_matrix(weighted_kernel)
    return K


def _matmul_param(mat1, mat2):
    """Multiplies two parametrized block matrices without explicitly converting to full matrices.

    Is equivalent to _blockmat2param(_param2blockmat(mat1) @ _param2blockmat(mat2), num_mic, ir_len). 

    Parameters
    ----------
    mat1 : np.ndarray of shape (dim1, dim2, ir_len, ir_len)
        The first matrix.
    mat2 : np.ndarray of shape (dim2, dim3, ir_len, ir_len)
        The second matrix.

    Returns
    -------
    np.ndarray of shape (dim1, dim3, ir_len, ir_len)
        The product matrix.
    
    """
    dim1, dim2, ir_len, _ = mat1.shape
    dim2b, dim3, _, _ = mat2.shape
    assert dim2 == dim2b, "The inner dimensions must match."
    assert mat1.dtype == mat2.dtype, "The matrices must have the same dtype."
    def _matmul_param_inner(m1, m2):
        return m1[:,None,:,:] @ m2[None,:,:,:]
    
    all_outer_products = jax.vmap(_matmul_param_inner, in_axes=(1, 0))(mat1, mat2)
    matmul_result = jnp.sum(all_outer_products, axis=0)
    return matmul_result
