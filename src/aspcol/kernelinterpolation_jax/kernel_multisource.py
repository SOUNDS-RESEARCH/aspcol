"""Contains frequency-domain kernel functions for sound field estimation of multiple sources, as introduced in [brunnströmSpatial2025].

For now the functions assume that all sources are measured at the same positions. But this can be generalized in the future, at 
the cost of more complicated code. The numpy module aspcol/kernelinterpolation/kernel_multisource.py contains more general implementations for 
when the sources are measured at different positions.

[brunnströmSpatial2025] J. Brunnström, M. B. Møller, J. Østergaard, T. van Waterschoot, M. Moonen, and F. Elvander, “Spatial covariance estimation for sound field reproduction using kernel ridge regression,” presented at the European Signal Processing Conference (EUSIPCO), Palermo, Italy, Sep. 2025.
"""

import jax 
import jax.numpy as jnp
import aspcol.kernelinterpolation_jax.kernel as kernel

import aspcore.matrices_jax as aspmat

def kernel_multisource(pos1, pos2, wave_num, num_src, weighting_func = None):
    """General kernel for joint estimation of a multisource sound field.
    
    Parameters
    ----------
    pos1 : ndarray of shape (num_pos1, 3)
    pos2 : ndarray of shape (num_pos2, 3)
    wave_num : ndarray of shape (num_freq)
        wave numbers for the frequencies of interest, defined as 2*pi*f/c
    base_kernel : optional, callable
        The base kernel function to use. If not given, this defaults to the diffuse Helmholtz kernel.
        Pick any single-frequency kernel in the kernelinterpolation module.
    base_kernel_args : optional, list
        The arguments to pass to the base kernel function, if required. 

    Returns
    -------
    full_kernel_matrix : ndarray of shape (num_freq, num_params1, num_params2)
        The kernel matrix for the given positions and sources. num_params1 and num_params2 are the number of parameters
        which can be calculated as sum(src_idx1) and sum(src_idx2) respectively. This should be exactly the total
        number of measurements in total. 
    """

    num_freq = wave_num.shape[0]
    if weighting_func is None:
        return _kernel_multisource_diffuse(pos1, pos2, wave_num, num_src)

    #assert num_src == src_idx2.shape[0]

    num_pos1 = pos1.shape[0]
    num_pos2 = pos2.shape[0]
    num_params1 = np.sum(src_idx1).astype(int)
    num_params2 = np.sum(src_idx2).astype(int) 
    
    if base_kernel is None:
        base_kernel = ki.kernel_helmholtz_3d
    if base_kernel_args is None:
        base_kernel_args = []
    if src_weighting is None:
        src_weighting = np.eye(num_src)[None,:,:]

    #src_weighting_sqrt = np.linalg.cholesky(src_weighting)
    #src_weighting_sqrt = np.stack([splin.sqrtm(src_weighting[i,:,:]) for i in range(src_weighting.shape[0])], axis=0)

    #rng = np.random.default_rng(2345654)
    #U = spstats.unitary_group.rvs(src_weighting.shape[1], random_state=rng)
    #src_weighting_sqrt = np.stack([src_weighting_sqrt[i,:,:] @ U for i in range(src_weighting.shape[0])], axis=0)

    kernel_vals = base_kernel(pos1, pos2, wave_num, *base_kernel_args)
    if kernel_vals.ndim == 3:
        kernel_vals = kernel_vals[:,None,:,:]

    full_kernel_matrix = np.zeros((num_freq, num_params1, num_params2), dtype=np.complex128)
    S_m = np.cumsum(np.concatenate(([0], np.sum(src_idx1, axis=0))))
    S_n = np.cumsum(np.concatenate(([0], np.sum(src_idx2, axis=0))))
    for m in range(num_pos1):
        for n in range(num_pos2):
            P_m = _src_idx_to_projection(src_idx1[:,m])
            P_n = _src_idx_to_projection(src_idx2[:,n])

            kernel_diag = np.eye(num_src)[None,:,:] * kernel_vals[:,None,:,m,n]
            kernel_block = src_weighting @ kernel_diag @ np.moveaxis(src_weighting, 1,2).conj()

            #kernel_block = kernel_vals[:,:,None,m,n] * src_weighting # corresponds to K @ B where K is diagonal and B is the weighting
            kernel_block = P_m[None,:,:] @ kernel_block @ P_n.T[None,:,:]

            full_kernel_matrix[:, S_m[m]:S_m[m+1], S_n[n]:S_n[n+1]] = kernel_block

    return full_kernel_matrix

def _kernel_multisource_diffuse(pos1, pos2, wave_num, num_src):
    """Diffuse sound field kernel for multiple sources.
    
    Parameters
    ----------
    pos1 : ndarray of shape (num_pos1, 3)
    pos2 : ndarray of shape (num_pos2, 3)
    wave_num : ndarray of shape (num_freq)
        wave numbers for the frequencies of interest, defined as 2*pi*f/c
    num_src : int
        The number of sources.
        
    Returns
    -------
    full_kernel_matrix : ndarray of shape (num_freq, num_pos1 * num_src, num_pos2 * num_src)
        The kernel matrix for the given positions and sources. num_params1 and num_params2 are the number of parameters
        which can be calculated as num_src * num_pos1 and num_src * num_pos2 respectively. This should be exactly the total
        number of measurements in total.
    """
    kernel_vals = kernel.kernel_diffuse(pos1, pos2, wave_num)
    kernel_mat = aspmat.block_diagonal_same(kernel_vals, num_src)
    return kernel_mat