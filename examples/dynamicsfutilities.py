import numpy as np

def angular_speed_limit(c : float, seq_len : int, radius : float):
    """
    From Hahn & Spors - Continuous measuremet of impulse responses
    on a circle using a uniformly moving microphone

    Parameters
    ----------
    c : float
        the speed of sound in meters per second
    seq_len : int
        The length of the perfect sequence in samples
        Should also be larger than the room impulse response length
    radius : float
        The radius in meters of the circle where the measurements are made

    Returns
    -------
    angular_speed : float
        Maximum speed the microphone can move without spatial aliasing
        the angular speed is in radians per second
    """
    angular_speed = c / (radius * seq_len)
    return angular_speed

def max_frequency_for_angular_speed(angular_speed, c, samplerate, seq_len, radius):
    """
    Returns max frequency in Hz
    """
    return c * samplerate / (2 * angular_speed * seq_len * radius)

def angular_speed_to_angular_period(angular_speed):
    return 2 * np.pi / angular_speed

def angular_period_to_angular_speed(angular_period):
    return 2 * np.pi / angular_period

def periodic_angular_period(c : float, seq_len : int, radius : float, samplerate : int, factor=1):
    """
    Gives back an angular speed of the microphone around a circle
    which respects the sampling theorem of dsu.angular_speed_limit
    but also makes sure that the sequence period lines up exactly
    with the period of the movement around a circle

    factor adjusts the reference (the maximum) speed. set below 1 to 
    have a slower speed
    """
    angular_speed_max = factor * angular_speed_limit(c, seq_len, radius)
    angular_period_min = 2 * np.pi / angular_speed_max
    num_samples_min = samplerate * angular_period_min

    num_required_full_periods = num_samples_min // seq_len
    num_periods = int(num_required_full_periods + 1)

    num_samples = num_periods * seq_len
    angular_period = num_samples / samplerate
    return angular_period

def max_order_spherical_harmonics(wavenumber, radius):
    """
    According to the definition in katzberg et al. 
    SPHERICAL HARMONIC REPRESENTATION FOR DYNAMIC SOUND-FIELD MEASUREMENTS

    Parameters
    ----------
    wavenumber : ndarray of shape (num_freqs)
    radius : float
        represents r_max in the Katzberg paper

    Returns
    -------
    M_f : ndarray of shape (num_freqs)
        contains an integer which is the minimum order of the spherical harmonics
    """
    return np.ceil(wavenumber * radius).astype(int)




#=========== Below are implemented separately
# check them against the earlier to see which are correct


def minimum_sampling_points(f_max, c, radius):
    """
    M_eff \geq \frac{4 pi f_max radius}{c}
    """
    return 4 * np.pi * f_max * radius / c

def max_frequency(M_eff, c, radius):
    w_max = M_eff * c / (2 * radius)
    return w_max / (2 * np.pi)

def minimum_sampling_points_shd_version(f_max, c, radius):
    """
    from Simultaneous Measurement of Spatial Room Impulse Responses from Multiple Sound Sources Using a Continuously Moving Microphone
    should give same result (or +-1) as minimum_sampling_points, but uses a different formulation to get there
    """
    return 2 * np.ceil(2 * np.pi * f_max * radius / c) + 1