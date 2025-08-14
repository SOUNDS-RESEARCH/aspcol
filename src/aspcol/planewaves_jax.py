"""Module for working with plane waves and plane wave models.

The plane wave is defined as $e^{-ik(r-r_c)^T d}$ where r is the position, d is the direction of the plane wave.
and r_c is the expansion center (the point around which the directions are calculated). 

Using the time-harmonic convention of $exp(-iwt)$, the plane wave is defined as $exp(ikr^T d)$ where d is the plane 
wave propagation direction [martinMultiple2006]. Therefore the direction provided is the direction from which the plane wave is incoming.

References
----------
[martinScattering2006] P. A. Martin, Multiple scattering: Interaction of time-harmonic waves with N obstacles, vol. 107. in Encyclopedia of mathematics and its applications, vol. 107. Cambridge, UK: Cambridge University Press, 2006.

"""

import numpy as np
import jax
import jax.numpy as jnp


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
        direction = direction[None,:]
    # assert direction.ndim == 2
    # assert direction.shape[-1] == 3
    # assert np.allclose(np.linalg.norm(direction, axis=-1), 1)
    # assert pos.ndim == 2
    # assert pos.shape[1] == 3

    if exp_center is not None:
        if exp_center.ndim == 1:
            exp_center = exp_center[None,:]
        #assert exp_center.shape == (1,3)
        pos = pos - exp_center

    if jnp.isscalar(wave_num):
        wave_num = jnp.array([wave_num])
    #assert wave_num.ndim == 1
    num_freqs = wave_num.shape[0]

    pw_values = jnp.exp(-1j * wave_num[:,None,None] * jnp.sum(pos[:,None,:] * direction[None,:,:], axis=-1)[None,...])

    if num_freqs == 1:
        pw_values = jnp.squeeze(pw_values, axis=0)
    return pw_values