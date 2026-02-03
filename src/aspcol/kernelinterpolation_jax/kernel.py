import jax.numpy as jnp
import jax
from functools import partial

@jax.jit
def kernel_diffuse(pos1, pos2, wave_num):
    """Diffuse sound field kernel.
    
    Defined for each position pair as j_0 (k lVert r - r' rVert_2^2) where j_0 is the zeroth order Bessel function

    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first set of points.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second set of points.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.

    Returns
    -------
    np.ndarray of shape (num_points1, num_points2, num_real_freqs)
        The kernel matrix.
    """
    dist_diff = jnp.expand_dims(pos1,1) - jnp.expand_dims(pos2,0)
    dist_mat = jnp.linalg.norm(dist_diff, ord=2, axis=-1)
    return jnp.sinc(jnp.expand_dims(dist_mat,0) * wave_num.reshape(-1,1,1) / jnp.pi)

@jax.jit
def kernel_directional_vonmises(pos1, pos2, wave_num, direction, beta):
    """Directional sound field kernel. 

    For sound field estimation, direction should be chosen as the primary propagation direction.
    This means that for a receiver at [0,0,0] and a source at [10,0,0], the direction should be [-1,0,0].

    Parameters
    ----------
    pos1 : np.ndarray of shape (num_points1, 3)
        Position of the first set of points.
    pos2 : np.ndarray of shape (num_points2, 3)
        Position of the second set of points.
    wave_num : np.ndarray of shape (num_real_freqs,)
        Wave number, defined as 2*pi*f/c, where f is the frequency and c is the speed of sound.
    direction : np.ndarray of shape (3,) or (num_dirs,3)
        The direction of the directional weighting.
    beta : float
        The strength of the directional weighting. A larger value will give more regularization.

    Returns
    -------
    np.ndarray of shape (num_real_freqs, num_points1, num_points2) or (num_real_freqs, num_dirs, num_points1, num_points2)
        The kernel matrix. If only one direction is given, the axis is removed. 

    Notes
    -----
    This function implements the kernel k(r, r') = j_0(sqrt{xi^T xi})
    where xi = k * (r - r') - 1j * beta * direction 
    and k is the wave number.

    This kernel is obtained from the inner product 
    langle u, v rangle_H = int_{S^2} u(d) conj(v(d)) / gamma(d) ds(d)
    where the weighting function gamma(d) = e^{-beta direction^T d}

    The norm is therefore higher for u(eta), and u(eta) is defined as the 
    plane wave coefficient for a plane wave incoming from eta. Therefore 
    such a wave is less preferred. -> A wave incoming from -eta is more preferred, 
    -> a wave propagating towards eta is preferred. 

    Another expression for the kernel here is k(r, r') = int_{S^2} e^{-i k (r-r')^T d} gamma(d) ds(d),
    which if evaluated gives the closed form solution above. 
    """
    angle_term = 1j * beta * direction.reshape((1,-1,1,1,direction.shape[-1]))
    pos_term = wave_num.reshape((-1,1,1,1,1)) * (pos1.reshape((1,1,-1,1,pos1.shape[-1])) - pos2.reshape((1,1,1,-1,pos2.shape[-1])))
    kernel_values = jnp.sinc(jnp.sqrt(jnp.sum((pos_term - angle_term)**2, axis=-1)) / jnp.pi)

    normalization = 2 * beta / (jnp.exp(beta) - jnp.exp(-beta))
    normalization = jnp.where(beta == 0, 1.0, normalization)  # Avoid division by zero for beta=0
    kernel_values = kernel_values * normalization

    if direction.ndim == 1 or direction.shape[0] == 1:
        kernel_values = jnp.squeeze(kernel_values, axis=1)  
    return kernel_values
