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



def reconstruct_multisrc(krr_params, pos_output, pos_data, wave_num, kernel_func, kernel_args):
    num_freq = wave_num.shape[0]
    kernel_matrix = kernel_func(pos_output, pos_data, wave_num, *kernel_args)

    reconstructed = np.squeeze(kernel_matrix @ krr_params[:,:,None], axis=-1)
    reconstructed = np.reshape(reconstructed, (num_freq, pos_output.shape[0], -1))
    reconstructed = np.moveaxis(reconstructed, 1, 2)
    return reconstructed

def get_krr_params_multisrc(data, pos, wave_num, reg_param, kernel_func, kernel_args):
    """
    data : ndarray of shape (num_freq, num_measurements). 
        IMPORTANT: the data first has the measurements for pos[:,0] for all sources that was measured there,
        then the data for pos[:,1] and so on. 
        num_measurements is somewhere between num_pos and num_pos * num_src, 
        depending on how many sources were measured at each position.
    pos : ndarray of shape (num_pos, 3)
        the positions of all measurements where at least one source was measured
    src_idx : ndarray of shape (num_src, num_pos) with boolean values
        indicates whether that source was measured at that position. 
    wave_num : ndarray of shape (num_freq)
    reg_param : float
        positive value to regularize the problem. Corresponds to lambda in the optimization problem. 
    kernel_func : callable
        with calling signature kernel_func(pos1, pos2, *kernel_args)
        should return ndarray (..., num_pos1, num_pos2)
    kernel_args : list
        extra arguments that are needed for the kernel function
    """
    num_measurements = data.shape[-1]
    kernel_matrix = kernel_func(pos, pos, wave_num, *kernel_args)

    K_reg = kernel_matrix + reg_param * np.eye(num_measurements)[None,:,:]
    a = np.linalg.solve(K_reg, data)
    return a

def kernel_multisrc(pos1, pos2, wave_num, num_src, weighting_func = None):
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
        return _kernel_multisrc_diffuse(pos1, pos2, wave_num, num_src)
    elif isinstance(weighting_func, jax.Array) or isinstance(weighting_func, jnp.ndarray):
        return _kernel_multisrc_diffuse_srcweighted(pos1, pos2, wave_num, num_src, weighting_func)

    num_pos1 = pos1.shape[0]
    num_pos2 = pos2.shape[0]
    num_params1 = num_pos1 * num_src
    num_params2 = num_pos2 * num_src

    kernel_vals = kernel.kernel_diffuse(pos1, pos2, wave_num)

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

def _kernel_multisrc_diffuse(pos1, pos2, wave_num, num_src):
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

def _kernel_multisrc_diffuse_srcweighted(pos1, pos2, wave_num, num_src, src_weighting):
    """Diffuse sound field kernel for multiple sources with a constant source weighting.
    
    Parameters
    ----------
    pos1 : ndarray of shape (num_pos1, 3)
    pos2 : ndarray of shape (num_pos2, 3)
    wave_num : ndarray of shape (num_freq)
        wave numbers for the frequencies of interest, defined as 2*pi*f/c
    num_src : int
        The number of sources.
    src_weighting : ndarray of shape (num_src, num_src) or (num_freq, num_src, num_src)
        The source weighting matrix, must be Hermitian and positive definite. If given as (num_src, num_src), the same
        weighting is used for all frequencies. 

    Returns
    -------
    full_kernel_matrix : ndarray of shape (num_freq, num_pos1 * num_src, num_pos2 * num_src)
        The kernel matrix for the given positions and sources. num_params1 and num_params2 are the number of parameters
        which can be calculated as num_src * num_pos1 and num_src * num_pos2 respectively. This should be exactly the total
        number of measurements in total.
    """
    kernel_vals = kernel.kernel_diffuse(pos1, pos2, wave_num)
    num_freq = wave_num.shape[0]
    assert isinstance(src_weighting, (jax.Array, jnp.ndarray))
    assert src_weighting.shape == (num_src, num_src) or src_weighting.shape == (num_freq, num_src, num_src)

    num_pos1 = pos1.shape[0]
    num_pos2 = pos2.shape[0]
    num_params1 = num_pos1 * num_src
    num_params2 = num_pos2 * num_src


    kernel_mat = aspmat.block_diagonal_same(kernel_vals, num_src)
    return kernel_mat