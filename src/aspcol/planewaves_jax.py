"""Module for working with plane waves and plane wave models.

The plane wave is defined as $e^{-ik(r-r_c)^T d}$ where r is the position, d is the direction of the plane wave.
and r_c is the expansion center (the point around which the directions are calculated).

Using the time-harmonic convention of $exp(-iwt)$, the plane wave is defined as $exp(ikr^T d)$ where d is the plane
wave propagation direction [martinMultiple2006]. Therefore the direction provided is the direction from which the plane wave is incoming.

References
----------
[martinScattering2006] P. A. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles, vol. 107. in Encyclopedia of mathematics and its applications, vol. 107. Cambridge, UK: Cambridge University Press, 2006.
[brunnstromBayesian2025] J. Brunnström, M. B. Møller, and M. Moonen, “Bayesian sound field estimation using moving microphones,” IEEE Open Journal of Signal Processing, vol. 6, pp. 312–322, Jan. 2025, doi: 10.1109/OJSP.2025.3526546.

"""

import aspcore.quadrature_jax as quad
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def plane_wave(pos, direction, wave_num, exp_center=None):
    """The complex response of a plane wave for a specific frequency for a set of positions.

    Implements exp(-ik(r-r_c)^T d) where r is the position, d is the direction of the plane wave.
    and r_c is the expansion center (the point around which the directions are calculated).

    Using the time-harmonic convention of exp(-iwt), the plane wave is defined as exp(ikr^T d) where d is the plane
    wave propagation direction [martinMultiple2006]. Therefore the direction provided is the direction from which the plane wave is incoming.

    pos : ndarray of shape (num_positions, 3)
        The positions where the plane wave is evaluated
    direction : ndarray of shape (num_direction, 3)
        The directionfrom which the plane wave is incoming. Must be a unit vector
    wave_num : ndarray of shape (num_real_freqs,)
        The wave number of the plane wave. Defined as 2*pi*f/c where f is the frequency
        and c is the speed of sound.
    exp_center : ndarray of shape (3,) or (1,3), optional
        The expansion center around which the directions are calculated. If None, the origin is used.
        Usually not required, but is provided for generality

    Returns
    -------
    plane_wave : ndarray of shape (num_positions, num_direction) or (num_real_freqs, num_positions, num_direction)
        The complex response of the plane wave at the positions.

    References
    ----------
    [martinMultiple2006] P. A. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles, vol. 107. in Encyclopedia of mathematics and its applications, vol. 107. Cambridge, UK: Cambridge University Press, 2006.
    """
    if direction.ndim == 1:
        direction = direction[None, :]
    # assert direction.ndim == 2
    # assert direction.shape[-1] == 3
    # assert np.allclose(np.linalg.norm(direction, axis=-1), 1)
    # assert pos.ndim == 2
    # assert pos.shape[1] == 3

    if exp_center is not None:
        if exp_center.ndim == 1:
            exp_center = exp_center[None, :]
        # assert exp_center.shape == (1,3)
        pos = pos - exp_center

    if jnp.isscalar(wave_num):
        wave_num = jnp.array([wave_num])
    # assert wave_num.ndim == 1
    num_freqs = wave_num.shape[0]

    pw_values = jnp.exp(
        -1j
        * wave_num[:, None, None]
        * jnp.sum(pos[:, None, :] * direction[None, :, :], axis=-1)[None, ...]
    )

    if num_freqs == 1:
        pw_values = jnp.squeeze(pw_values, axis=0)
    return pw_values


def plane_wave_integral(
    dir_func, pos, wave_num, exp_center=np.zeros(3), order=7, method="t-design"
):
    """Computes the integral of a function multiplied with a plane wave over a sphere.

    Defined according to (6) in [brunnstromBayesian2025]
    int_{S^2} f(d) exp(-ik(r-r_c)^T d) ds(d)
    where S^2 is the unit sphere, f(d) is the function, r is the position, r_c is the expansion center,
    d is the incoming direction of the plane wave, k is the wave number, and ds(d) is the surface element
    of the sphere.

    Parameters
    ----------
    dir_func : function
        A function that takes direction unit vectors, ndarray of shape (num_points, 3)
        and returns a complex value response of shape (num_points)
    pos : ndarray of shape (num_pos, 3)
        The position where
    exp_center : ndarray of shape (1,3)
        The center of the expansion
    wave_num : float
        The wave number of the plane wave
    rng : numpy.random.Generator
        The random number generator to use

    Returns
    -------
    est : ndarray of shape (num_pos,)
        The estimated value of the integral evaluated at all the supplied positions

    Notes
    -----
    Same definition is (9) in Ribeiro 2023, but with a sign difference in the complex exponential,
    and without an expansion center.
    """
    if method == "t-design":
        directions = quad.t_design(order)
    elif method == "montecarlo":
        raise NotImplementedError("Monte Carlo method not implemented yet")
        # directions = mc.uniform_random_on_sphere(num_samples, rng)
    else:
        raise ValueError(f"Unknown method {method}")

    func_values = dir_func(directions)
    planewave_values = plane_wave(pos - exp_center, directions, wave_num)
    mean_integrand = jnp.mean(func_values[None, :] * planewave_values, axis=-1)

    sphere_area = 4 * jnp.pi  # must multiply by area of integration domain
    est = sphere_area * mean_integrand
    return est
