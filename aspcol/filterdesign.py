"""Helper functions to design filters

Can be used to obtain FIR filters from frequency responses. Also contains
helper functions associated with the discrete Fourier transform. 
"""
import numpy as np
import scipy.signal as signal
import itertools as it


# ================== BASIC DFT FUNCTIONS =============================

def get_frequency_values(num_freq, samplerate):
    """
    Deprecated, use get_real_freqs instead
    
    Get the frequency values of all positive frequency bins in Hz
    numFreq is the number of frequency bins INCLUDING negative frequencies
    If numFreq is even, it will return (numFreq/2)+1 values
    If numFreq is odd, it will return (numFreq+1)/2 values
    use np.fft.fftfreq to double check values when implementing. 
    """
    print(f"get_frequency_values is deprecated, use get_real_freqs instead")
    if num_freq % 2 == 0:
        return (samplerate / (num_freq)) * np.arange(num_freq // 2 + 1)
    elif num_freq % 2 == 1:
        raise NotImplementedError
    else:
        raise ValueError

def get_freqs(num_freq : int, samplerate : int):
    """
    Returns the sampled frequencies in Hz for a discrete Fourier transform

    Parameters
    ----------
    num_freq : int
        should equal the length of the sequence 
        so it includes the number of negative frequencies
    samplerate : int

    Returns
    -------
    freqs : ndarray of shape (num_freq,)    
    """
    return np.arange(num_freq) * samplerate / num_freq

def get_wavenum(num_freq : int, samplerate : int, c : float):
    """
    Returns the wave numbers associated with the sampled frequencies
    of the DFT

    See documentation for get_freqs
    """
    return get_angular_freqs(num_freq, samplerate) / c

def get_angular_freqs(num_freq : int, samplerate : int):
    """
    Returns the angular frequencies associated with the sampled frequencies
    of the DFT

    See documentation for get_freqs
    """
    return 2 * np.pi * get_freqs(num_freq, samplerate)

def get_real_freqs(num_freq : int, samplerate : int):
    """
    Returns the real sampled frequencies in Hz for a discrete Fourier transform

    Parameters
    ----------
    num_freq :int 
        should equal the length of the sequence, so it includes 
        the number of negative frequencies
    samplerate : int
    
    Returns
    -------
    freqs : ndarray of shape (num_real_freq,)
        if num_freq is even, num_real_freq = num_freq // 2 + 1
        if num_freq is odd, num_real_freq = (num_freq + 1) // 2
    """
    if num_freq % 2 == 0:
        return (samplerate / (num_freq)) * np.arange(num_freq // 2 + 1)
    elif num_freq % 2 == 1:
        raise NotImplementedError
    else:
        raise ValueError

def get_real_wavenum(num_freq : int, samplerate : int, c : float):
    """
    Get wave numbers associated with the real frequencies
    of the DFT

    See documentation for get_real_freqs
    """
    return get_real_angular_freqs(num_freq, samplerate) / c

def get_real_angular_freqs(num_freq : int, samplerate : int):
    """
    Returns angular frequencies associated with the real frequencies 
    of the DFT

    See documentation for get_real_freqs
    """
    return 2 * np.pi * get_real_freqs(num_freq, samplerate)


def insert_negative_frequencies(freq_signal, even):
    """
    Inserts all negative frequency values under the
    assumption of conjugate symmetry, i.e. a real impulse response.

    To be used in conjunction with get_real_freqs
    
    Parameters
    ----------
    freq_signal : ndarray of shape (num_real_freq, ...)
    even : bool
        if True, the full dft length was even

    Returns
    -------
    freq_signal_full : ndarray of shape (num_freq, ...)
        num_freq is even if even=True, and odd if even=False
    
    """
    if even:
        return np.concatenate(
            (freq_signal, np.flip(freq_signal[1:-1, :, :].conj(), axis=0)), axis=0
        )
    else:
        raise NotImplementedError
    


def dft_vector (freq_idx : int, dft_len : int):
    """
    Returns the vector with all complex exponential values used
    to calculate the DFT

    Parameters
    ----------
    freq_idx : int
        the index of the frequency bin
    dft_len : int
        the number of frequency bins
    
    Returns
    -------
    exp_vector : ndarray of shape (dft_len,)
    """
    exp_factor = -2 * np.pi * 1j * freq_idx / dft_len
    return np.exp(np.arange(dft_len) * exp_factor)

def idft_vector (freq_idx : int, dft_len : int):
    """
    See documentation fpr dft_vector
    """
    exp_factor = 2 * np.pi * 1j * freq_idx / dft_len
    return np.exp(np.arange(dft_len) * exp_factor) / dft_len




# ============== DESIGN FIR FILTERS ====================

def fir_from_freqs_window(freq_filter, ir_len, two_sided=True, window="hamming"):
    """
    Constructs a FIR filter from frequency values
    
    Currently works only for two_sided=True and odd ir_len

    Parameters
    ----------
    freq_filter : ndarray of shape (num_freqs, ...)
        should include both positive and negative frequencies
    ir_len : int
        the length of the impulse response
    two_sided : bool
        if True, the impulse response is assumed to be centered in the middle
    window : str or None
        the window to apply to the impulse response
        
    Returns
    -------
    trunc_filter : ndarray of shape (ir_len, ...)
    trunc_error : float
        the relative truncation error of the impulse response 
    """
    assert ir_len % 1 == 0
    assert freq_filter.shape[0] % 2 == 0
    if two_sided:
        #halfLen = irLen // 2
        mid_point = freq_filter.shape[0] // 2

        time_filter = np.fft.ifft(freq_filter, axis=0)
        new_axis_order = np.concatenate((np.arange(1, freq_filter.ndim), [0]))
        time_filter = np.real_if_close(np.transpose(time_filter, new_axis_order))

        time_filter = np.concatenate((time_filter[...,-mid_point:], time_filter[...,:mid_point]), axis=-1)

        trunc_filter, trunc_error = truncate_filter(time_filter, ir_len, True)


        if window == "hamming":
            trunc_filter = trunc_filter * signal.windows.hamming(ir_len).reshape(
            (1,) * (trunc_filter.ndim - 1) + trunc_filter.shape[-1:]
            )
        elif window is None:
            pass
        else:
            raise ValueError("Invalid value for window argument")
    else:
        raise NotImplementedError

    return trunc_filter, trunc_error

def truncate_filter(ir, ir_len, two_sided):
    """
    Truncates the impulse response to the desired length
    Currently only works for two_sided=True and odd ir_len

    Parameters
    ----------
    ir : ndarray of shape (..., ir_len_original)
    ir_len : int
        the length of the truncated impulse response
    two_sided : bool
        if True, the impulse response is assumed to be centered in the middle

    Returns
    -------
    trunc_filter : ndarray of shape (..., ir_len)
    """
    if two_sided:
        assert ir_len % 2 == 1
        assert ir.shape[-1] % 2 == 0
        half_len = ir_len // 2
        mid_point = ir.shape[-1] // 2
        trunc_filter = ir[..., mid_point-half_len:mid_point+half_len+1]

        trunc_power = np.sum(ir[...,:mid_point-half_len]**2) + np.sum(ir[...,mid_point+half_len+1:]**2)
        total_power = np.sum(ir**2)
        rel_trunc_error = 10 * np.log10(trunc_power / total_power)
    else:
        raise NotImplementedError
    return trunc_filter, rel_trunc_error



def calc_truncation_error(ir, ir_len, two_sided=True):
    """Calculates the relative truncation error of an impulse response
    The relative error is how much of the power of the impulse response that is lost by truncating.

    Parameters
    ----------
    ir : ndarray of shape (..., ir_len_original)
    ir_len : int
        the length of the truncated impulse response
    two_sided : bool
        if True, the impulse response is assumed to be centered in the middle
    
    Returns
    -------
    rel_trunc_error : float
    """
    if two_sided:
        assert ir_len % 2 == 1
        half_len = ir_len // 2
        mid_point = ir.shape[-1]
        trunc_power = np.sum(ir[...,:mid_point-half_len]**2) + np.sum(ir[...,mid_point+half_len:]**2)
        total_power = np.sum(ir**2)
        rel_trunc_error = 10 * np.log10(trunc_power / total_power)
    else:
        raise NotImplementedError
    return rel_trunc_error
    

def min_truncated_length(ir, two_sided=True, max_rel_trunc_error=1e-3):
    """Calculates the minimum length you can truncate a filter to.
    The minimum length will be calculated independently for all impulse responses, and
    the longest length chosen. The relative error is how much of the
    power of the impulse response that is lost by truncating.
    
    Parameters
    ----------
    ir : ndarray of shape (..., ir_len)
    two_sided : bool
        if True, the impulse response is assumed to be centered in the middle
    max_rel_trunc_error : float
        the maximum relative truncation error allowed

    Returns
    -------
    req_filter_length : int
        the minimum length you can truncate the filter to
    """
    ir_len = ir.shape[-1]
    ir_shape = ir.shape[:-1]

    total_energy = np.sum(ir ** 2, axis=-1)
    energy_needed = (1 - max_rel_trunc_error) * total_energy

    if two_sided:
        center_idx = ir_len // 2
        casual_sum = np.cumsum(ir[..., center_idx:] ** 2, axis=-1)
        noncausal_sum = np.cumsum(np.flip(ir[..., :center_idx] ** 2, axis=-1), axis=-1)
        energy_sum = casual_sum
        energy_sum[..., 1:] += noncausal_sum
    else:
        energy_sum = np.cumsum(ir ** 2, axis=-1)

    enough_energy = energy_sum > energy_needed[..., None]
    trunc_indices = np.zeros(ir_shape, dtype=int)
    for indices in it.product(*[range(dimSize) for dimSize in ir_shape]):
        trunc_indices[indices] = np.min(
            np.nonzero(enough_energy[indices + (slice(None),)])[0]
        )

    req_filter_length = np.max(trunc_indices)
    if two_sided:
        req_filter_length = 2 * req_filter_length + 1
    return req_filter_length