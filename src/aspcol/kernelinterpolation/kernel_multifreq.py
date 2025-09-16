import numpy as np

import aspcol.kernelinterpolation.kernel as ki
import aspcore.fouriertransform as ft
import aspcore.matrices as aspmat

def multifreq_diffuse_kernel(pos1, pos2, wave_num, diag_mat=True):
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
    kernel_val = ki.kernel_diffuse(pos1, pos2, wave_num)
    kernel_val = np.moveaxis(kernel_val, 0, -1)

    if diag_mat:
        kernel_matrix = np.eye(kernel_val.shape[-1])[None,None,...] * kernel_val[...,None,:]
        return kernel_matrix
    return kernel_val

def multifreq_directional_kernel_vonmises(pos1, pos2, wave_num, direction, beta, diag_mat=True):
    """Multiple frequency directional sound field kernel. 

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
    direction : np.ndarray of shape (3,1)
        The direction of the directional weighting.
    beta : float
        The strength of the directional weighting. A larger value will give more regularization.

    Returns
    -------
    np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs)
        The kernel matrix.

    Notes
    -----
    Clearly this is space-inefficient implementation as a diagonal matrix is stored as a full matrix. But it 
    is provided to easy combine with other functions and check correctness. 

    References
    ----------
    [uenoDirectionally2021]
    [brunnströmTime2025]
    """
    # minus direction because the ki module uses the other time convention (and therefore plane wave definitions)
    kernel_val = ki.kernel_directional(pos1, pos2, wave_num, direction, beta)
    kernel_val = np.squeeze(kernel_val, axis=1)
    kernel_val = np.moveaxis(kernel_val, 0, -1)

    if diag_mat:
        kernel_matrix = np.eye(kernel_val.shape[-1])[None,None,...] * kernel_val[...,None,:]
        return kernel_matrix
    return kernel_val



def _weighting_mat_from_frequency_domain_envelope_reg(envelope_reg, num_freqs, dft_len, freqs_to_remove_low=0):
    if envelope_reg.ndim == 2:
        envelope_reg = envelope_reg[None,:,:]
    c_diag = ft.rdft_weighting(num_freqs, dft_len, freqs_to_remove_low=freqs_to_remove_low)
    envelope_reg_adjoint = (1/c_diag)[None,:,None] * c_diag[None,None,:] * np.moveaxis(envelope_reg.conj(), -1, -2) # equals C^{-1} @ envelope_reg^H @ C
    weighting_mat = envelope_reg_adjoint @ envelope_reg
    return weighting_mat

def multifreq_envelope_kernel(pos1, pos2, wave_num, envelope_reg, reg_points, dft_len, freqs_to_remove_low=0):
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
    envelope_reg : np.ndarray of shape (num_freqs, num_freqs) or (num_reg_points, num_freqs, num_freqs)
        The envelope regularization weighting. Can be computed from the time-domain values of the envelope regularization
        as D_f = F D_t F^{-1}. If only a single matrix is provided, it is assumed to be the same for all regularization points.
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
    num_freqs = wave_num.shape[0]

    weight_mat, B = _weighting_mat_from_frequency_domain_envelope_reg(envelope_reg, num_freqs, dft_len, freqs_to_remove_low=freqs_to_remove_low)

    gamma1 = multifreq_diffuse_kernel(pos1, reg_points, wave_num)
    gamma2 = multifreq_diffuse_kernel(reg_points, pos2, wave_num)

    gamma2 = weight_mat[:,None,:,:] @ gamma2
    return aspmat.matmul_param(gamma1, gamma2) / (num_reg_points**2)