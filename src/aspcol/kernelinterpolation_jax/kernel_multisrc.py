"""Contains frequency-domain kernel functions for sound field estimation of multiple sources, as introduced in [brunnströmSpatial2025].

For now the functions assume that all sources are measured at the same positions. But this can be generalized in the future, at 
the cost of more complicated code. The numpy module aspcol/kernelinterpolation/kernel_multisource.py contains more general implementations for 
when the sources are measured at different positions.

[brunnströmSpatial2025] J. Brunnström, M. B. Møller, J. Østergaard, T. van Waterschoot, M. Moonen, and F. Elvander, “Spatial covariance estimation for sound field reproduction using kernel ridge regression,” presented at the European Signal Processing Conference (EUSIPCO), Palermo, Italy, Sep. 2025.
"""

import jax 
import jax.numpy as jnp
import aspcol.kernelinterpolation_jax.kernel as kernel
import aspcol.planewaves_jax as pw
import aspcore.montecarlo_jax as mc

import aspcore.matrices_jax as aspmat

def reconstruct_multisrc(krr_params, pos_output, pos_data, wave_num, kernel_func, kernel_args):
    """Reconstruct the sound field at the desired positions.

    Parameters
    ----------
    krr_params : ndarray of shape (num_freq, num_src, num_pos)
        The KRR parameters obtained from get_krr_params_multisrc.
    pos_output : ndarray of shape (num_eval, 3)
        The positions where the sound field should be reconstructed.
    pos_data : ndarray of shape (num_pos, 3)
        The positions where the sound field was measured.
    wave_num : ndarray of shape (num_freq)
        wave numbers for the frequencies of interest, defined as 2*pi*f/c
    kernel_func : callable
        with calling signature kernel_func(pos1, pos2, wave_num, *kernel_args)
        should return ndarray (num_freq, num_pos1, num_pos2, num_src, num_src)
    kernel_args : list
        extra arguments that are needed for the kernel function

    Returns
    -------
    reconstructed : ndarray of shape (num_freq, num_src, num_eval)
        The reconstructed sound field at the desired positions.
    """
    num_freq = wave_num.shape[0]
    num_output = pos_output.shape[0]
    num_src = krr_params.shape[1]

    kernel_vals = kernel_func(pos_output, pos_data, wave_num, *kernel_args)
    kernel_vals = aspmat.param2blockmat(kernel_vals)

    krr_params = jnp.moveaxis(krr_params, 1, 2).reshape((num_freq, -1))
    reconstructed = jnp.squeeze(kernel_vals @ krr_params[...,None], axis=-1)

    reconstructed = jnp.moveaxis(jnp.reshape(reconstructed, (num_freq, num_output, num_src)), 1, 2)
    return reconstructed

def get_krr_params_multisrc(data, pos, wave_num, reg_param, kernel_func, kernel_args):
    """Get the optimal KRR parameters for standard kernel interpolation with multiple sources.

    Assumes that all sources are measured at the same positions.

    The kernel function k(r,r') is matrix-valued, where each value is a matrix of size (num_src, num_src).
    Therefore each kernel function should return something with shape (num_freq, num_pos1, num_pos2, num_src, num_src).

    So the kernel matrix K has shape (num_freq, num_pos * num_src, num_pos * num_src).

    Parameters
    ----------
    data : ndarray of shape (num_freq, num_src, num_pos). 
        The measured complex sound pressure at the measurement positions.
    pos : ndarray of shape (num_pos, 3)
        the positions of the measurements 
    wave_num : ndarray of shape (num_freq)
    reg_param : float
        positive value to regularize the problem. Corresponds to lambda in the optimization problem. 
    kernel_func : callable
        with calling signature kernel_func(pos1, pos2, wave_num, *kernel_args)
        should return ndarray (num_freq, num_pos1, num_pos2, num_src, num_src)
    kernel_args : list
        extra arguments that are needed for the kernel function

    Returns
    -------
    krr_params : ndarray of shape (num_freq, num_src, num_pos)
        The optimal KRR parameters for the given data and kernel.
    """
    num_pos = pos.shape[0]
    num_src = data.shape[1]
    kernel_vals = kernel_func(pos, pos, wave_num, *kernel_args)
    K = aspmat.param2blockmat(kernel_vals)
    K_reg = K + reg_param * jnp.eye(K.shape[-1])[None,:,:]

    data = jnp.moveaxis(data, 1, 2).reshape((wave_num.shape[0], -1))
    a = jnp.squeeze(jnp.linalg.solve(K_reg, data[...,None]), axis=-1)
    a = jnp.moveaxis(jnp.reshape(a, (wave_num.shape[0], num_pos, num_src)), 1, 2)
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
        return kernel_multisrc_diffuse(pos1, pos2, wave_num, num_src)
    elif isinstance(weighting_func, jax.Array) or isinstance(weighting_func, jnp.ndarray):
        return kernel_multisrc_diffuse_srcweighted(pos1, pos2, wave_num, num_src, weighting_func)
    else: #assume its a callable 
        return kernel_multisrc_numerical(pos1, pos2, wave_num, num_src, weighting_func)
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


def kernel_multisrc_numerical(pos1, pos2, wave_num, num_src, src_weighting, num_points = 256, key = None):
    """General kernel for joint estimation of a multisource sound field.
    
    Parameters
    ----------
    pos1 : ndarray of shape (num_pos1, 3)
    pos2 : ndarray of shape (num_pos2, 3)
    wave_num : ndarray of shape (num_freq)
        wave numbers for the frequencies of interest, defined as 2*pi*f/c
    num_src : int
        The number of sources.
    src_idx1 : ndarray of shape (num_src, num_pos1), dtype=bool
        Boolean array indicating which positions in pos1 correspond to which sources.
        Each column should have exactly one True value, and each row should have at least one True
        value.
    """
    num_freqs = wave_num.shape[0]
    num_pos1 = pos1.shape[0]
    num_pos2 = pos2.shape[0]

    if key is None:
        key = jax.random.PRNGKey(1234567)
    directions = mc.uniform_random_on_sphere(num_points, key)

    pos_diff = pos1[:,None,:] - pos2[None,:,:] # shape (num_pos1, num_pos2, 3)
    plane_waves = pw.plane_wave(pos_diff.reshape(-1, 3), directions, wave_num) # shape (num_freq, num_pos^2, num_dirs)
    plane_waves = jnp.reshape(plane_waves, (num_freqs, num_pos1, num_pos2, num_points)) # shape (num_freq, num_pos1, num_pos2, num_dirs)
    plane_waves = jnp.moveaxis(plane_waves, -1, 1) # shape (num_freqs, num_dirs, num_pos1, num_pos2)

    func_values = src_weighting(directions) # shape (num_freqs, num_dirs, num_src, num_src)
     
    kernel_val = 4 * jnp.pi * jnp.mean(func_values[:,:,None,None,:,:] * plane_waves[...,None,None], axis=1)

    return kernel_val

def kernel_multisrc_diffuse(pos1, pos2, wave_num, num_src):
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
    full_kernel_matrix : ndarray of shape (num_freq, num_pos1, num_pos2, num_src, num_src)
        The kernel matrix for the given positions and sources. The matrix consists of blocks of size (num_src, num_src), each
        of which is the matrix-valued kernel for pos1[i] and pos2[j]. 
    """
    kernel_vals = kernel.kernel_diffuse(pos1, pos2, wave_num)
    kernel_mat = kernel_vals[...,None,None] * jnp.eye(num_src, dtype = int)[None,None,:,:]

    #kernel_mat = jnp.kron(kernel_vals, jnp.eye(num_src, dtype = int)) #aspmat.block_diagonal_same(kernel_vals, num_src)
    return kernel_mat

def kernel_multisrc_diffuse_srcweighted(pos1, pos2, wave_num, num_src, src_weighting):
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

    if src_weighting.ndim == 2:
        src_weighting = src_weighting[None,:,:] # make it (num_freq, num_src, num_src)

    kernel_mat = kernel_vals[...,None,None] * src_weighting[:,None,None,...]

    #num_pos1 = pos1.shape[0]
    #num_pos2 = pos2.shape[0]
    #num_params1 = num_pos1 * num_src
    #num_params2 = num_pos2 * num_src


    #kernel_mat = aspmat.block_diagonal_same(kernel_vals, num_src)
    return kernel_mat