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
import scipy.special as special

import aspcol.utilities as utils
import aspcol.montecarlo as mc
import aspcol.sphericalharmonics as sph



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
    wave_num : float or ndarray of shape (num_real_freqs,)
        The wave number of the plane wave. Defined as 2*pi*f/c where f is the frequency 
        and c is the speed of sound.

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
    assert direction.ndim == 2
    assert direction.shape[-1] == 3
    assert np.allclose(np.linalg.norm(direction, axis=-1), 1)
    assert pos.ndim == 2
    assert pos.shape[1] == 3

    if exp_center is not None:
        if exp_center.ndim == 1:
            exp_center = exp_center[None,:]
        assert exp_center.shape == (1,3)
        pos = pos - exp_center

    if np.isscalar(wave_num):
        wave_num = np.array([wave_num])
    assert wave_num.ndim == 1
    num_freqs = wave_num.shape[0]

    pw_values = np.exp(-1j * wave_num[:,None,None] * np.sum(pos[:,None,:] * direction[None,:,:], axis=-1)[None,...])

    if num_freqs == 1:
        pw_values = np.squeeze(pw_values, axis=0)
    return pw_values



def find_matching_plane_wave(plane_wave_data, freq, pw_dir, pos_mic, c):
    """Matches a plane wave to the given data

    The function returns directly the sound pressure of the matched plane wave, but also
    enough information for the plane wave to be recreated as gain_adjustment * pw.plane_wave(pos_mic, pw_dir, wave_num, exp_center)

    Parameters
    ----------
    plane_wave_data : ndarray of shape (num_mic, 1)
        data from a plane wave or plane-wave like sound field that should be matched by the analytic plane wave
    freq : float
        frequency of the plane wave
    pw_dir : ndarray of shape (1, 3)
        Direction from which the plane wave is incoming. Must be a unit vector
    pos_mic : ndarray of shape (num_mic, 3)
        positions of the microphones (the points where the plane wave and the plane_wave_data are measured)
    c : float
        speed of sound

    Returns
    -------
    p_mic_analytic : ndarray of shape (num_mic, 1)
        the sound pressure at pos_mic of the analytic plane wave that matches the plane_wave_data. 
        If plane_wave_data is a plane wave or very close, then p_mic_analytic should be 
        essentially identical to plane_wave_data
    exp_center : ndarray of shape (1, 3)
        the expansion center of the plane wave
    gain_adjustment : float
        A gain adjustment that should be applied to the analytic plane wave to match the plane_wave_data
    
        
    Notes
    -----
    The gain adjustment is found as the mean amplitude of the plane_wave_data. 

    The phase adjustment is found as the solution to $\lVert e^{i \gamma} \bm{p} - \bm{s} \rVert_2^2$, where $\bm{p}$ is 
    the vector of analytic pressure values and $\bm{s}$ is the vector of plane_wave_data. The solution is found analytically
    by taking the derivative and setting to zero. 

    The expansion center is defined by $exp(-ik(r-r_c)^T d) = exp(-i gamma) exp(ikr^T d)$, where r_c is the unknown expansion center
    and the phase adjustment gamma is found by the algorithm. Since that relationship is not necessarily unique, the 
    expansion center is found as min \lVert r_c \rVert_2^2 subject to r_c^T d = gamma / wave_num, which is a 
    least squares problem with linear constrained, solved analytically using the Lagrange multiplier method.
    """
    freq = np.ones(1) * freq
    wave_num = 2 * np.pi * freq / c
    p_mic_analytic = np.squeeze(plane_wave(pos_mic, pw_dir, wave_num), axis=-1)
    #p_mic_analytic = np.conj(p_mic_analytic) # to change time convention

    sim_amplitude = np.mean(np.abs(plane_wave_data))
    analytic_amplitude = np.mean(np.abs(p_mic_analytic))
    gain_adjustment = sim_amplitude / analytic_amplitude
    p_mic_analytic = p_mic_analytic * gain_adjustment

    phase_adjustment_angle = -(1j / 2) * (np.log(np.sum(np.conj(p_mic_analytic) * plane_wave_data)) - np.log(np.sum(p_mic_analytic * np.conj(plane_wave_data))) )
    phase_adjustment = np.exp(1j * phase_adjustment_angle)
    p_mic_analytic = p_mic_analytic * phase_adjustment

    exp_center = np.real_if_close(pw_dir * phase_adjustment_angle / wave_num)
    return p_mic_analytic, exp_center, gain_adjustment

def shd_coeffs_for_planewave(pw_direction, max_order):
    """Spherical harmonic coefficients for a plane wave exp(-ikr^T d)
    where r is the position and d is the direction of the plane wave.

    The expansion center is assumed to be at the origin.

    Parameters
    ----------
    pw_direction : ndarray of shape (num_pw, 3)
        The direction of the plane wave. Must be a unit vector
    max_order : int
        The maximum order of the spherical harmonics expansion    
    
    Returns
    -------
    shd_coeffs : ndarray of shape (num_pw, num_coeffs)
        The spherical harmonic coefficients for the plane wave
    """
    assert pw_direction.ndim == 2
    assert pw_direction.shape[1] == 3
    assert np.allclose(np.linalg.norm(pw_direction, axis=-1), 1)

    rad, angles = utils.cart2spherical(pw_direction)
    orders, degrees = sph.shd_num_degrees_vector(max_order)
    const_factor = (-1j)**orders * np.sqrt(4 * np.pi)
    sph_harm = special.sph_harm(degrees[None,:], orders[None,:], angles[...,0,None], angles[...,1,None])
    shd_coeffs = const_factor[None,:] * np.conj(sph_harm)
    return shd_coeffs

def _shd_coeffs_for_planewave_positive(pw_direction, max_order):
    """Harmonic coefficients for a plane wave exp(ikr^T d)
    where r is the position and d is the direction of the plane wave.

    The expansion center is assumed to be at the origin.

    Parameters
    ----------
    pw_direction : ndarray of shape (num_pw, 3)
        The direction of the plane wave. Must be a unit vector
    max_order : int
        The maximum order of the spherical harmonics expansion    
    
    Returns
    -------
    shd_coeffs : ndarray of shape (num_pw, num_coeffs)
        The spherical harmonic coefficients for the plane wave
    """
    assert pw_direction.ndim == 2
    assert pw_direction.shape[1] == 3
    assert np.allclose(np.linalg.norm(pw_direction, axis=-1), 1)

    rad, angles = utils.cart2spherical(pw_direction)
    orders, degrees = sph.shd_num_degrees_vector(max_order)

    # Calculate harmonic coefficients
    const_factor = 1j**orders * np.sqrt(4 * np.pi)
    sph_harm = special.sph_harm(degrees[None,:], orders[None,:], angles[...,0,None], angles[...,1,None])
    shd_coeffs = const_factor[None,:] * np.conj(sph_harm)
    return shd_coeffs


def plane_wave_integral(dir_func, pos, exp_center, wave_num, rng, num_samples):
    """Computes the integral of a function multiplied with a plane wave over a sphere.

    Defined according to (6) in Brunnström et al 2024.
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

    Performs the integration using Monte Carlo integration. For computational cost sensitive applications,
    there are more efficient methods available, such as using a Lebedev quadrature.
    """
    dir_vecs = mc.uniform_random_on_sphere(num_samples, rng)

    func_values = dir_func(dir_vecs)
    planewave_values = plane_wave(pos - exp_center, dir_vecs, wave_num)
    mean_integrand = np.mean(func_values[None,:] * planewave_values, axis=-1)

    sphere_area = 4 * np.pi # must multiply by area of integration domain
    est = sphere_area * mean_integrand
    return est








# ==============================================================================
# MEASUREMENTS WITH PLANE WAVE MODELS

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



def apply_measurement(pw_coeffs, dir_func, rng, num_samples):
    """Returns the signal that a microphone with directionality given by dir_func would record. 
    The microphone must be located at the expansion center for the plane wave coefficients.

    Defined as (7) in Brunnström et al 2024.

    Parameters
    ----------
    pw_coeffs : function
        the coefficients of the plane wave expansion defining the soundfield. This is a function that
        takes a direction unit vector and returns a complex value response.
    dir_func : function
        A function that takes direction unit vectors, ndarray of shape (num_points, 3)
        and returns a complex value response of shape (num_points)
    rng : numpy.random.Generator
        The random number generator to use
    num_samples : int
        The number of samples to use for the monte carlo integration

    Returns
    -------
    
    """
    dir_vecs = mc.uniform_random_on_sphere(num_samples, rng)
    
    pw_vals = pw_coeffs(dir_vecs)
    dir_vals = dir_func(dir_vecs)
    mean_integrand = np.mean(pw_vals * dir_vals, axis=-1)
    
    sphere_area = 4 * np.pi
    est = sphere_area * mean_integrand
    return est