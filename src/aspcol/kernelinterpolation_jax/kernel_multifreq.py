import numpy as np
import jax.numpy as jnp
import jax
from functools import partial

import aspcore.fouriertransform_jax as ft
import aspcol.kernelinterpolation_jax.kernel as kernel

@partial(jax.jit, static_argnames=["diag_mat"])
def kernel_diffuse_multifreq(pos1, pos2, wave_num, diag_mat=True):
    """Multiple frequency diffuse sound field kernel. 

    Defined for each position pair as diag{}_{i=0}^{L//2} j_0 (k_i lVert r - r' rVert_2^2) 
    where L is the (even) length of the real DFT, and hence L//2 + 1 is the number of real frequencies. 

    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first point.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second point.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.

    Returns
    -------
    np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs)
        Returned if diag_mat is true. Is a diagonal matrix
    np.ndarray of shape (num_points1, num_points2, num_real_freqs)
        Returned if diag_mat is false. Contains the same values as the diagonal matrix, so 
        is a more space-efficient representation. 

    Notes
    -----
    Clearly this is space-inefficient implementation as a diagonal matrix is stored as a full matrix. But it 
    is provided to easy combine with other functions and check correctness. 

    References
    ----------
    [uenoKernel2018]
    [brunnströmTime2025]
    """
    kernel_val = kernel.kernel_diffuse(pos1, pos2, wave_num)
    kernel_val = np.moveaxis(kernel_val, 0, -1)

    if diag_mat:
        kernel_matrix = np.eye(kernel_val.shape[-1])[None,None,...] * kernel_val[...,None,:]
        return kernel_matrix
    return kernel_val