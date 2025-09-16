import numpy as np

import scipy.linalg as splin

import aspcol.kernelinterpolation.kernel as ki
import aspcore.fouriertransform as ft
import aspcore.matrices as aspmat

import aspcol.planewaves as pw
import aspcore.montecarlo as mc

import aspcol.kernelinterpolation.kernel_multifreq as kernel_multifreq


def time_domain_diffuse_kernel(pos1, pos2, wave_num):
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
    freq_kernel = kernel_multifreq.multifreq_diffuse_kernel(pos1, pos2, wave_num, diag_mat=False)
    # for m in range(pos1.shape[0]):
    #     for m2 in range(pos2.shape[0]):
    #         if m != m2:
    #             freq_kernel[m,m2,0] = 0.9
    kernel_matrix = freq_to_time_domain_kernel_matrix(freq_kernel)
    return kernel_matrix

def time_domain_directional_kernel_vonmises(pos1, pos2, wave_num, direction, beta):
    freq_kernel = kernel_multifreq.multifreq_directional_kernel_vonmises(pos1, pos2, wave_num, direction, beta, diag_mat=False)
    kernel_matrix = freq_to_time_domain_kernel_matrix(freq_kernel)
    return kernel_matrix

def time_domain_directional_kernel_vonmises_approx(pos1, pos2, wave_num, direction, beta):
    if direction.ndim == 1:
        direction = direction[None,:]

    rng = np.random.default_rng(12345)
    num_samples = int(1e4)

    def _vonmises_dir_function(dir):
        """The von Mises directional function.
        dir : (num_dirs, 3)
        """
        return np.exp(beta * np.sum(direction * dir, axis=-1))
    
    pos_diff = pos1[:,None,:] - pos2[None,:,:]
    pos_diff = np.reshape(pos_diff, (-1, 3))
    integral = pw.plane_wave_integral(_vonmises_dir_function, pos_diff, np.zeros(3), wave_num, rng, num_samples)

    kernel_matrix = np.reshape(integral, (wave_num.shape[0], pos1.shape[0], pos2.shape[0]))
    kernel_matrix = np.moveaxis(kernel_matrix, 0, -1)
    return freq_to_time_domain_kernel_matrix(kernel_matrix)

def time_domain_directional_kernel(pos1, pos2, wave_num, dir_function):
    """The time domain directional kernel for a general directional function.

    The dir_function supplied to this function is W^H(d) W(d) in [brunnströmTime2025], as there is little 
    reason to use a rectangular W in general. That means that the dir_function has to be positive semi-definite.
    
    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first set of points.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second set of points.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    dir_function : callable
        A function that takes a set of direction unit vectors of shape (num_dirs, 3) 
        and returns a positive semi-definite matrix for each direction, a ndarray of shape (num_dirs, num_real_freqs, num_real_freqs).
    
    Returns
    -------
    np.ndarray of shape (num_points1, num_points2, dft_len, dft_len)
        The time domain kernel matrix. 
        The dft_len is assumed to be even, and the number of real frequencies is dft_len//2 + 1.

    References
    ----------
    [brunnströmTime2025]
    """
    assert pos1.ndim == 2 and pos2.ndim == 2, "The positions must be 2D arrays."
    assert pos1.shape[1] == 3 and pos2.shape[1] == 3, "The positions must be 3-dimensional."
    assert wave_num.ndim == 1, "The wave number must be a 1D array."
    num_real_freqs = wave_num.shape[0]

    rng = np.random.default_rng(12345)

    num_samples = int(1e3)
    dir_vecs = mc.uniform_random_on_sphere(num_samples, rng)

    dir_matrices = dir_function(dir_vecs) #shape (num_dir, num_real_freqs, num_real_freqs)
    E1 = pw.plane_wave(pos1, dir_vecs, wave_num).T # shape (num_dir, num_points1, num_real_freqs)
    E2 = pw.plane_wave(-pos2, dir_vecs, wave_num).T # shape (num_dir, num_points2, num_real_freqs)

    kernel_matrix = np.zeros((pos1.shape[0], pos2.shape[0], num_real_freqs, num_real_freqs), dtype=complex)
    MAX_DIR = 100
    num_batches = 1 + num_samples // MAX_DIR
    for i in range(num_batches):
        print(f"Batch {i+1} of {num_batches}")
        start_idx = i*MAX_DIR 
        end_idx = np.min([(i+1)*MAX_DIR, num_samples])
        if start_idx == end_idx:
            break

        kernel_matrix += np.mean(E1[start_idx:end_idx,:,None,:,None] * 
                                dir_matrices[start_idx:end_idx,None,None,:,:] * 
                                E2[start_idx:end_idx,None,:,None,:], axis=0)

    kernel_matrix *= 4 * np.pi / num_batches

    #the expression before taking the mean should be of shape (num_dir, num_points1, num_points2, num_real_freqs, num_real_freqs)
    #integral = 4 * np.pi *  np.mean(E1[:,:,None,:,None] * dir_matrices[:,None,None,:,:] * E2[:,None,:,None,:], axis=0)
    return freq_to_time_domain_kernel_matrix(kernel_matrix)











def freq_to_time_domain_kernel_matrix(freq_kernel):
    """Turns a frequency domain kernel matrix into a time domain kernel matrix.

    Could be made faster by using the RFFT directly, but this is a clear and easy to check implementation.
    Write unit tests against this function to implement a faster version.
    
    Parameters
    ----------
    freq_kernel : np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs) 
        or (num_points1, num_points2, num_real_freqs)
        The frequency domain kernel matrix. If dimension is 3, the kernel is assumed to be diagonal.
    
    
    """
    assert freq_kernel.ndim == 4 or freq_kernel.ndim == 3
    if freq_kernel.ndim == 3:
        diag_mat = False
    else:
        diag_mat = True
    dft_len = freq_kernel.shape[-1] * 2 - 2

    if not diag_mat:
        freq_diag_mat = np.zeros((freq_kernel.shape[0], freq_kernel.shape[1], freq_kernel.shape[2], freq_kernel.shape[2]), dtype=complex)
        for i in range(freq_kernel.shape[0]):
            for j in range(freq_kernel.shape[1]):
                freq_diag_mat[i,j,:,:] = np.diag(freq_kernel[i,j,:])
        freq_kernel = freq_diag_mat

    F = ft.rdft_mat(dft_len)[None, None,:,:]
    B = ft.irdft_mat(dft_len)[None, None,:,:]
    td_kernel = np.real(B @ freq_kernel @ F)

    if not np.allclose(td_kernel.imag, 0):
        raise ValueError("Something went wrong, the time domain kernel matrix is not real-valued.")
    td_kernel = np.real(td_kernel)
    return td_kernel



def _freq_to_time_domain_kernel_matrix_diagonal(freq_kernel):
    """Turns a diagonal frequency domain kernel matrix into a time domain kernel matrix.

    Only really makes sense if the frequency domain kernel is diagonal.
    
    Parameters
    ----------
    freq_kernel : np.ndarray of shape (num_points1, num_points2, num_real_freqs, num_real_freqs)
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


    #freq_kernel_dup = np.zeros((pos1.shape[0], pos2.shape[0], dft_len, dft_len))
    #freq_kernel_dup[..., :num_real_freqs, :num_real_freqs] = kernel_matrix
    #freq_kernel_dup[..., num_real_freqs:, num_real_freqs:] = np.flip(kernel_matrix[...,1:-1,1:-1], axis=(-2,-1))
    if diag_mat:
        freq_kernel = np.diagonal(freq_kernel, axis1=-2, axis2=-1)
    a_ext = ft.insert_negative_frequencies(freq_kernel.T, even=True).T
    a_mat = np.eye(dft_len)[None,None,...] * a_ext[...,None,:]

    # The FFT is the correct fast way to do this 
    #kernel_matrix = np.fft.fft(np.fft.ifft(a_mat, axis=-2), axis=-1)

    # for consistency, we use the other time-convention as defined by aspcol
    b = np.moveaxis(ft.ifft(np.moveaxis(a_mat,-2, 0)), -1, 2)
    kernel_matrix = np.moveaxis(ft.fft(b), 0, -1)

    # Below is a more readable version 
    #F = splin.dft(dft_len)
    #Finv = F.conj().T / dft_len
    #kernel_matrix = Finv[None,None,...] @ a_mat @ F[None,None,...]

    if not np.allclose(kernel_matrix.imag, 0):
        raise ValueError("Something went wrong, the time domain kernel matrix is not real-valued.")
    kernel_matrix = np.real(kernel_matrix)
    return kernel_matrix




def time_domain_envelope_kernel(pos1, pos2, wave_num, envelope_reg, reg_points):
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

    gamma1 = time_domain_diffuse_kernel(pos1, reg_points, wave_num)
    gamma2 = envelope_reg[:,None,:,None] * time_domain_diffuse_kernel(reg_points, pos2, wave_num) # equals diag(envelope_reg)[:,None,:,:] @ gamma2
    return aspmat.matmul_param(gamma1, gamma2) / (num_reg_points**2)

def time_domain_envelope_integral_kernel(pos1, pos2, wave_num, envelope_reg, reg_points, integral_volume):
    """The kernel Gamma_r(r, r') of the time domain diffuse sound field with envelope regularization.
    
    This is regularization option 1 in [brunnströmTime2025], which is constructed as a weighting of the 
    sound field as a whole. 

    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first set of points.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second set of points.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    envelope_reg : np.ndarray of shape (dft_len,)
        The envelope regularization weighting. The values must be positive and real-valued. The parameter
        represents the diagonal values of D^H D in [brunnströmTime2025]. 
    reg_points : callable
        A function that takes an integer and returns that many uniformly sampled points in the domain of the 
        integral. The function should have signature reg_points(num_points), and return points of shape (num_points, 3).
    integral_volume : float
        The volume of the integral domain.
    """
    NUM_POINTS = 40
    NUM_EACH_BATCH = 5
    num_batches = NUM_POINTS // NUM_EACH_BATCH

    assert envelope_reg.ndim == 1, "The envelope regularization must be a 1D array."
    dft_len = envelope_reg.shape[0]
    num_pos1 = pos1.shape[0]
    num_pos2 = pos2.shape[0]

    
    if np.array_equal(pos1, pos2):
        same_pos = True
    else:
        same_pos = False

    int_points = reg_points(NUM_POINTS)

    int_value = np.zeros((num_pos1, num_pos2, dft_len, dft_len), dtype=float)
    for i in range(num_batches):
        print(f"monte carlo batch for r {i+1} of {num_batches}")
        int_batch = int_points[i*NUM_EACH_BATCH:(i+1)*NUM_EACH_BATCH,:]
        gamma1 = time_domain_diffuse_kernel(pos1, int_batch, wave_num)
        gamma1_weighted = gamma1 * envelope_reg[None,None,None,:] # equal to gamma @ np.diag(envelope_reg)[None,None,:,:]

        if same_pos:
            gamma2 = aspmat.param_transpose(gamma1)
        else:
            gamma2 = time_domain_diffuse_kernel(int_batch, pos2, wave_num)

        ival = aspmat.matmul_param(gamma1_weighted, gamma2)
        int_value += ival
    int_value *= integral_volume / NUM_POINTS
    
    return int_value
