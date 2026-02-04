"""A number of hyperparameters are needed for kernel ridge regression

In this module there are methods for finding appropriate values of hyperparameters from the data using cross validation
"""




import numpy as np
import jax.numpy as jnp
import jax
import optax
from functools import partial

import aspcore.fouriertransform_jax as ft
import aspcore.matrices_jax as aspmat

import aspcol.kernelinterpolation_jax as kernel

import aspcol.soundfieldestimation_jax.kernel_estimation_jax as ksjax


@partial(jax.jit, static_argnames="score")
def find_reg_param_time_domain_sound_field(data, pos, samplerate, c, reg_params, data_weighting=None, score="gcv", **kwargs):
    """Computes the GCV score for time-domain sound field kernel interpolation.

    data : np.ndarray of shape (num_data, data_dim)
        For sound field estimation, this is the sound pressure at the measurement positions.
        for time-domain kernel interpolation, data_dim is the length of the impulse response.
        for frequency-domain kernel interpolation, data_dim is 1
    pos : np.ndarray of shape (num_data, 3)
        Positions of the measurement points.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    reg_param : float
        regularization parameter to evaluate the GCV score for.

    score : str in {'gcv', 'ml', 'loocv', 'loocv_spatially_blocked'}
        decides which type of score to use to determine the best regularization parameter.
        gcv is generalized cross-validation
        ml is maximum likelihood
        loocv is leave-one-out cross-validation

    Returns
    -------
    gcv_score : float
        GCV score for the given regularization parameter.

    References
    ----------
    Time-domain sound field estimation using kernel ridge regression, J Brunnström, M. B. Møller, J Østergaard, S Koyama, T van Waterschoot, M Moonen, 2025
    """
    num_pos = pos.shape[0]
    ir_len = data.shape[-1]
    even_dft_length = ir_len % 2 == 0

    mat_size = num_pos * ir_len
    wave_num = ft.get_real_wavenum(ir_len, samplerate, c)
    K = kernel.kernel_time_domain_diffuse(pos, pos, wave_num, real_nyquist=even_dft_length)
    K = aspmat.param2blockmat(K)

    if data_weighting is not None:
        data_weighting = ksjax._data_weighting_argument_parsing(data_weighting, num_pos, ir_len)
        data_weighting = jnp.diag(1 / data_weighting)
        reg_matrix_base = data_weighting
    else:
        reg_matrix_base = jnp.eye(mat_size)

    def compute_score(reg):
        reg_matrix = reg * reg_matrix_base
        if score == "gcv":
            score_value = gcv_score(K, reg_matrix, data)
        elif score == "ml":
            score_value = ml_score(K, reg_matrix, data)
        elif score == "loocv":
            score_value = loocv_score(K, reg_matrix, data)
        elif score == "loocv_spatially_blocked":
            score_value = loocv_score_spatially_blocked(K, reg_matrix, data, pos, block_radius=kwargs["block_radius"])
        else:
            raise ValueError("unrecognized score parameter")
        return score_value
    
    score_values = jax.lax.map(compute_score, reg_params, batch_size=1)  

    best_index = jnp.argmin(score_values)
    best_reg_param = reg_params[best_index]

    return best_reg_param, score_values


def gcv_score(K, reg_matrix, data):
    """Computes the generalized cross-validation (GCV) score for kernel interpolation.

    Parameters
    ----------
    K : np.ndarray of shape (mat_size, mat_size)
        Kernel matrix.
    reg_matrix : np.ndarray of shape (mat_size, mat_size)
        Regularization matrix. Is often an identity matrix. This is scaled by the regularization parameter
    data : np.ndarray of shape (num_data, data_dim)
        The data, which can be vector-valued at each point.

    Returns
    -------
    gcv_score : float
        GCV score for the given regularization parameter.

    References
    ----------
    Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter - Gene H. Golub, Michael Heath, and Grace Wahba, 1979
    """
    mat_size = K.shape[0]
    K_reg = K + reg_matrix

    smoothing_matrix = jax.scipy.linalg.solve(K_reg.T, K.T, assume_a="pos").T
    I_minus_S = jnp.eye(mat_size) - smoothing_matrix
    trace_I_minus_S = jnp.trace(I_minus_S)

    numerator = jnp.sum((I_minus_S @ data.flatten()) ** 2) / mat_size
    denom = (trace_I_minus_S / mat_size) ** 2
    score = numerator / denom
    return score

def ml_score(K, reg_matrix, data):
    """Negative log likelihood score for regularization parameter for kernel interpolation.

    Assumes that the noise is Gaussian with zero mean and covariance matrix equal to reg_matrix.

    Parameters
    ----------
    K : np.ndarray of shape (mat_size, mat_size)
        Kernel matrix.
    reg_matrix : np.ndarray of shape (mat_size, mat_size)
        Regularization matrix. Is often an identity matrix. This is scaled by the regularization parameter
    data : np.ndarray of shape (num_data, data_dim)
        The data, which can be vector-valued at each point.

    Returns
    -------
    score : float
        The negative log likelihood score.
    """
    data = data.flatten()
    K_reg = K + reg_matrix

    data_error = jnp.real(data.conj() @ jax.scipy.linalg.solve(K_reg, data, assume_a="pos"))
    complexity_penalty = jnp.linalg.slogdet(K_reg)[1]

    score = data_error + complexity_penalty
    return score

def loocv_score(K, reg_matrix, data):
    """Leave-one-out cross-validation score for kernel interpolation.
    
    Parameters
    ----------
    K : np.ndarray of shape (mat_size, mat_size)
        Kernel matrix.
    reg_matrix : np.ndarray of shape (mat_size, mat_size)
        Regularization matrix. Is often an identity matrix. This is scaled by the regularization parameter
    data : np.ndarray of shape (num_data, data_dim)
        The data, which can be vector-valued at each point.

    Returns
    -------
    score : float
        The LOOCV score.
    """
    num_data = data.shape[0]
    data_dim = data.shape[1]
    mat_size = num_data * data_dim

    K_reg = K + reg_matrix

    score = 0.0
    for i in range(num_data):
        mask = jnp.concatenate((jnp.arange(0, i), jnp.arange(i+1, num_data)))
        mask_full = jnp.concatenate((jnp.arange(0, i*data_dim), jnp.arange((i+1)*data_dim, mat_size)))

        data_masked = data[mask,:].flatten()
        K_masked = K_reg[mask_full,:]
        K_masked = K_masked[:,mask_full]

        krr_params = jax.scipy.linalg.solve(K_masked, data_masked, assume_a="pos")

        predict_mask = jnp.arange(i*data_dim, (i+1)*data_dim)
        K_predict = K[:,mask_full]
        K_predict = K_predict[predict_mask, :]
        p_est = K_predict @ krr_params
        score = score + jnp.mean(jnp.abs(p_est - data[i:i+1,:])**2)

    score = score / num_data
    return score


def loocv_score_spatially_blocked(K, reg_matrix, data, pos, block_radius = 1.0):
    """Leave-one-out cross-validation score with spatial blocking for kernel interpolation.
    
    Instead using all data points except one for training, all points within the block_radius 
    of the test point are excluded from training. This can help when the noise is spatially correlated.

    Parameters
    ----------
    K : np.ndarray of shape (mat_size, mat_size)
        Kernel matrix.
    reg_matrix : np.ndarray of shape (mat_size, mat_size)
        Regularization matrix. Is often an identity matrix. This is scaled by the regularization parameter
    data : np.ndarray of shape (num_data, data_dim)
        The data, which can be vector-valued at each point.
    pos : np.ndarray of shape (num_data, 3)
        Positions of the measurement points. In machine learning literature this is likely to be called the data points, and 
        what we call data here is often called labels.
    block_radius : float
        The radius around each test point where data points are excluded from training.

    Returns
    -------
    score : float
        The LOOCV score.
    """
    num_data = data.shape[0]
    data_dim = data.shape[1]
    mat_size = num_data * data_dim

    K_reg = K + reg_matrix
    score = 0.0

    def inner_loop(args):
        pos_i, data_i, i = args
        pos_i = pos_i[None,:]
        data_i = data_i[None,:]

        distances = jnp.linalg.norm(pos - pos_i, axis=1)
        point_mask = distances > block_radius
        full_mask = jnp.repeat(point_mask, data_dim)
        data_masked = jnp.where(full_mask, data.flatten(), 0.0)

        mask_matrix = full_mask[:, None] & full_mask[None, :]
        K_masked = jnp.where(mask_matrix, K_reg, 0.0)
        K_masked = K_masked + jnp.diag((~full_mask) * 1e6)

        krr_params = jax.scipy.linalg.solve(K_masked, data_masked, assume_a="pos")

        predict_mask = jax.lax.dynamic_slice(jnp.arange(mat_size), (i*data_dim,), (data_dim,))
        K_predict = K[predict_mask, :]
        p_est = K_predict @ krr_params
        score = jnp.mean(jnp.abs(p_est - data_i)**2)
        return score
    
    scores = jax.lax.map(inner_loop, (pos, data, jnp.arange(num_data)), batch_size=1)
    score = jnp.sum(scores)

    return score


def l_curve_plot(K, reg_matrix, data, reg_params):
    """Returns the x and y values for an L-curve plot

    This can be used to manually select a regularization parameter by finding the corner of the L-curve.
    In practice, there are ways to automatically find the corner, by computing the curvature of the L-curve.

    Parameters
    ----------
    K : np.ndarray of shape (mat_size, mat_size)
        Kernel matrix.
    reg_matrix : np.ndarray of shape (mat_size, mat_size)
        Regularization matrix. Is often an identity matrix. This is scaled by the regularization parameter.
    data : np.ndarray of shape (num_data, data_dim)
        For sound field estimation, this is the sound pressure at the measurement positions.
        for time-domain kernel interpolation, data_dim is the length of the impulse response.
        for frequency-domain kernel interpolation, data_dim is 1
    reg_params : np.ndarray of shape (num_reg_params,)
        Regularization parameters to evaluate the L-curve for.


    Returns
    -------
    values : np.ndarray of shape (num_reg_params, 2)
        The x and y values for the L-curve plot in log scale. 
    
    References
    ----------
    The L-curve and its use in the numerical treatment of inverse problems, P. C. Hansen
    https://www.sintef.no/globalassets/project/evitameeting/2005/lcurve.pdf
    """
    data = data.flatten()
    data_errors = []
    reg_sizes = []

    for reg_param in reg_params:
        reg_matrix_scaled = reg_param * reg_matrix
        K_reg = K + reg_matrix_scaled
        
        krr_params = jax.scipy.linalg.solve(K_reg, data, assume_a="pos")
        data_est = K @ krr_params

        data_error = jnp.linalg.norm(data - data_est) 
        model_norm = jnp.sqrt(jnp.real(krr_params.conj().T @ K @ krr_params))

        # Store log-log values
        data_errors.append(jnp.log10(data_error))
        reg_sizes.append(jnp.log10(model_norm))
    values = jnp.stack([jnp.array(data_errors), jnp.array(reg_sizes)], axis=1)
    return values
