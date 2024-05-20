import numpy as np
import itertools as it
import scipy.linalg as splin
import scipy.special as special
import scipy.spatial.distance as distance

import aspcol.kernelinterpolation as ki
import aspcol.utilities as utils
import aspcol.filterdesign as fd




def plane_wave(pos, direction, wave_num):
    """The complex response of a plane wave for a specific frequency for a set of positions.

    Implements exp(-ikr^T d) where r is the position and d is the direction of the plane wave.

    pos : ndarray of shape (num_positions, 3)
        The positions where the plane wave is evaluated
    direction : ndarray of shape (num_direction, 3)
        The direction of the plane wave. Must be a unit vector
    wave_num : float
        The wave number of the plane wave. Defined as 2*pi*f/c where f is the frequency 
        and c is the speed of sound.

    Returns
    -------
    plane_wave : ndarray of shape (num_positions, num_direction)
        The complex response of the plane wave at the positions
    """
    if direction.ndim == 1:
        direction = direction[None,:]
    assert direction.ndim == 2
    assert direction.shape[-1] == 3
    assert np.allclose(np.linalg.norm(direction, axis=-1), 1)
    assert pos.ndim == 2
    assert pos.shape[1] == 3

    return np.exp(-1j * wave_num * np.sum(pos[:,None,:] * direction[None,:,:], axis=-1))


def omni_directivity_function():
    """

    """
    def dir_func(dir_vec):
        """
        dir_vec : ndarray of shape (num_directions, 3)
        these are the directions for which we test what the
        microphone response is. 
        """
        assert dir_vec.ndim == 2
        assert dir_vec.shape[1] == 3
        return np.ones((dir_vec.shape[0],), dtype=complex)
    return dir_func

def linear_directivity_function(A, d_mic):
    """
    
    Omni directivity is obtained by setting A = 0
    Cardoid directivity is obtained by setting A = 1/2
    Figure-8 directivity is obtained by setting A = 1

    d_mic : ndarray of shape (3,) or (1, 3)
        direction microphone is pointing, the peak directivity of e.g. a cardioid mic. 
    """
    if d_mic.ndim == 1:
        d_mic = d_mic[None,:]
    assert d_mic.ndim == 2
    assert d_mic.shape[1] == 3
    assert 0 <= A <= 1

    def dir_func(dir_vec):
        """
        dir_vec : ndarray of shape (num_directions, 3)
        these are the directions for which we test what the
        microphone response is. 
        """
        assert dir_vec.ndim == 2
        assert dir_vec.shape[1] == 3
        #dir_vec = np.expand_dims(dir_vec, axis=1)
        return (1-A) + A * np.sum(dir_vec * d_mic, axis=-1, dtype=complex)
    return dir_func
