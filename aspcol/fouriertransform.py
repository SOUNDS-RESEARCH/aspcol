"""Methods for discrete Fourier transform, and linear convolution and correlation in the frequency domain

The forward transforms will operate on the last axis, and put the resulting frequency axis first. The inverse transforms will operate on the first axis and put the resulting time axis last. 
A time domain sample can be accessed as time_signal[...,n]
A frequency domain sample can be accessed as freq_signal[f,...]

The motivation is that the methods are build for multichannel audio processing, where the assumption is that only 1-D FFTs are needed. The most common use case is that time domain processing operates directly on the last axis, and frequency domain processing operates on the channel axes, but broadcasts over the frequency axis. This behaviour for the transforms results in much fewer transpositions, and makes the code more readable.

Important
---------
Note the time convention, which is not the same as in the numpy.fft module. This is to achieve consistency with 
the acoustics literature that is used for the sound field analysis modules. The definition of the DFT is
u(k) = \sum_{n=0}^{N-1} u[n] e^{i 2pi k n / N}, for frequency bins k = 0, 1, ..., N-1,
and the definition of the Inverse DFT is
u[n] = \frac{1}{N} \sum_{k=0}^{N-1} u(k) e^{-i 2pi k n / N}.

It can be viewed as a direct consequence of choosing the definition of a plane wave propagating in the d direction to be
u(r) = e^{i k r^T d}, where k is the wavenumber.



The following methods can be helpful for filtering using frequency-domain filters
when multiple channels are involved. When possible, for correlation and convolution 
the functions will zero-pad the signals correctly in order to avoid circular convolution. 

In particular, the functions can be useful frequency domain adaptive filters, where overlap-save
is used for both linear convolutions and correlations. It can be tricky there to get 
the zero-padding right. 

References
----------

"""
import numpy as np

# =============================================================================
# Discrete Fourier Transform
# =============================================================================
def fft(time_sig, n=None):
    """Computes the FFT

    Computes along the last axis, and moves the resulting frequency axis to the first axis. 
    Uses the opposite time convention as numpy.fft.
    
    Parameters
    ----------
    time_sig : ndarray of shape (..., num_samples)
        The signal to be transformed. The last axis should correspond to time
    n : int, optional
        length of the FFT. If None, the length is the length of the last axis of time_sig
    
    Returns
    -------
    freq_signal : ndarray (num_freqs, ...)
        The transformed signal. 
    """
    if n is None:
        n = time_sig.shape[-1]
    freq_signal = np.fft.ifft(time_sig, n=n, axis=-1) * n
    freq_signal = np.moveaxis(freq_signal, -1, 0)
    return freq_signal

def ifft(freq_signal):
    """Computes the inverse FFT

    Computes along the first axis, and moves the resulting time axis to the last axis. 
    Uses the opposite time convention as numpy.fft.

    Parameters
    ----------
    freq_signal : ndarray of shape (num_freqs, ...)
        The signal to be transformed. The first axis should 
        correspond to frequency

    Returns
    -------
    time_signal : ndarray of shape (..., num_samples)
        The transformed signal
    """
    n = freq_signal.shape[0]
    time_signal = np.fft.fft(freq_signal, axis=0) / n
    time_signal = np.moveaxis(time_signal, 0, -1)
    return time_signal

def rfft(time_sig, n=None):
    """Computes the real FFT
    
    Parameters
    ----------
    time_sig : ndarray
        The signal to be transformed. The last axis should correspond to time
    n : int, optional
        length of the FFT. If None, the length is the length of the last axis of time_sig
    
    Returns
    -------
    freq_signal : ndarray
        The transformed signal

    Notes
    -----
    Corresponds to numpy.fft.rfft, but with the time convention used in this package. 
    """
    if n is None:
        n = time_sig.shape[-1]
    freq_signal = np.fft.ifft(time_sig, n=n, axis=-1) * n
    freq_signal = np.moveaxis(freq_signal, -1, 0)

    num_real_freqs = n // 2 + 1
    freq_signal = freq_signal[:num_real_freqs,...]
    return freq_signal

def irfft(freq_signal, n=None):
    """Inverse FFT, and moves the first axis to the last axis

    Parameters
    ----------
    freq_signal : ndarray
        The signal to be transformed. The first axis should 
        correspond to frequency
    n : int, optional
        length of the FFT. This is the length of the resulting time domain signal, not the frequency domain input.
        If not supplied, it is assumed to be 2 * (freq_signal.shape[0] - 1), corresponding to the 
        output of rfft without argument. To get an odd output length, you need to supply n.

    Returns
    -------
    time_signal : ndarray
        The transformed signal
    """
    if n is None:
        freq_signal = insert_negative_frequencies(freq_signal, even = True)
        #freq_signal = np.concatenate((freq_signal, np.flip(freq_signal[1:-1, ...].conj(), axis=0)), axis=0)
        n = freq_signal.shape[0]
    else:
        if n == 2 * (freq_signal.shape[0] - 1):
            freq_signal = insert_negative_frequencies(freq_signal, even = True)
            n = freq_signal.shape[0]
        elif n == 2 * (freq_signal.shape[0] - 1) + 1:
            freq_signal = insert_negative_frequencies(freq_signal, even = False)
            n = freq_signal.shape[0]
        else:  
            raise NotImplementedError("irfft with arbitrary output length not implemented")
    time_signal = np.fft.fft(freq_signal, axis=0) / n
    time_signal = np.moveaxis(time_signal, 0, -1)
    return np.real_if_close(time_signal)

def insert_negative_frequencies(freq_signal, even):
    """Inserts the values associated with negative frequencies.
    
    Requires the assumption of conjugate symmetry, i.e. the original signal was real. 
    Can be used in conjunction with get_real_freqs
    
    Parameters
    ----------
    freq_signal : ndarray of shape (num_real_freq, ...)
    even : bool
        if True, the full dft length was even, if False it was odd.

    Returns
    -------
    freq_signal_full : ndarray of shape (num_freq, ...)
        num_freq is even if even=True, and odd if even=False
    
    """
    if even:
        num_real_freqs = freq_signal.shape[0]
        all_freq_values = np.concatenate((freq_signal, np.flip(freq_signal[1:-1, ...].conj(), axis=0)), axis=0)
        all_freq_values[0, ...] = np.real(all_freq_values[0, ...])
        all_freq_values[num_real_freqs-1, ...] = np.real(all_freq_values[num_real_freqs-1, ...])
        return all_freq_values
    else:
        num_real_freqs = freq_signal.shape[0]
        all_freq_values = np.concatenate((freq_signal, np.flip(freq_signal[1:, ...].conj(), axis=0)), axis=0)
        all_freq_values[0, ...] = np.real(all_freq_values[0, ...])
        return all_freq_values



def dft_vector (freq_idx : int, dft_len : int):
    """All exponential values to calculate the DFT for a specific frequency bin

    Returns exp(i 2pi k n / N) / N for all time steps n = 0, 1, ..., N-1

    Parameters
    ----------
    freq_idx : int
        the index of the frequency bin. Symbol k in the formula above. 
    dft_len : int
        the number of frequency bins. Symbol N in the formula above.
    
    Returns
    -------
    exp_vector : ndarray of shape (dft_len,)

    Notes
    -----
    The values corresponds to the DFT definition used here which is u(k) = sum_n u[n] exp(i 2pi k n / N). See documentation for the fft function.

    Although inefficient, the DFT can be calculated by multiplying the input signal with this vector and summing. Preferably, use the fft and ifft functions. 
    """
    exp_factor = 2 * np.pi * 1j * freq_idx / dft_len
    return np.exp(np.arange(dft_len) * exp_factor)

def idft_vector (freq_idx : int, dft_len : int):
    """All exponential values to calculate the IDFT for a specific output time step

    Returns exp(i 2pi k n / N) for all frequency bins k = 0, 1, ..., N-1
    Which corresponds to the IDFT definition used here which is u(n) = sum_k u(k) exp(i 2pi k n / N) / N

    Parameters
    ----------
    time_idx : int
        the index of the time step bin. Symbol n in the formula above.
    dft_len : int
        the number of frequency bins. Symbol N in the formula above.
    
    Returns
    -------
    exp_vector : ndarray of shape (dft_len,)

    Notes
    -----
    The values corresponds to the DFT definition used here which is u(n) = \frac{1}{N} sum_k u(k) exp(-i 2pi k n / N). See documentation for the ifft function.

    Although inefficient, the DFT can be calculated by multiplying the input signal with this vector and summing. Preferably, use the fft and ifft functions. 
    """
    exp_factor = -2 * np.pi * 1j * freq_idx / dft_len
    return np.exp(np.arange(dft_len) * exp_factor) / dft_len





# =============================================================================
# Helper function for Discrete Fourier Transform
# =============================================================================
def get_freqs(num_freq : int, samplerate : int):
    """Returns the sampled frequencies in Hz for a discrete Fourier transform

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
    """Returns the wave numbers associated with the sampled frequencies of the DFT

    See documentation for get_freqs
    """
    return get_angular_freqs(num_freq, samplerate) / c

def get_angular_freqs(num_freq : int, samplerate : int):
    """Returns the angular frequencies associated with the sampled frequencies of the DFT

    See documentation for get_freqs
    """
    return 2 * np.pi * get_freqs(num_freq, samplerate)

def get_real_freqs(num_freq : int, samplerate : int):
    """Returns the real sampled frequencies in Hz for a discrete Fourier transform

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
        num_real_freq = num_freq // 2 + 1
        return (samplerate / (num_freq)) * np.arange(num_real_freq)
    elif num_freq % 2 == 1:
        num_real_freq = (num_freq + 1) // 2
        return (samplerate / (num_freq)) * np.arange(num_real_freq)
        #f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)
        #raise NotImplementedError
    else:
        raise ValueError

def get_real_wavenum(num_freq : int, samplerate : int, c : float):
    """Get wave numbers associated with the real frequencies of the DFT

    See documentation for get_real_freqs
    """
    return get_real_angular_freqs(num_freq, samplerate) / c

def get_real_angular_freqs(num_freq : int, samplerate : int):
    """Returns angular frequencies associated with the real frequencies of the DFT

    See documentation for get_real_freqs
    """
    return 2 * np.pi * get_real_freqs(num_freq, samplerate)








# =============================================================================
# The following methods are for filtering in the frequency domain.
# =============================================================================
def correlate_sum_tt(time_filter, time_signal):
    """Computes linear correlation between two time-domain signals

    The next to last dimension is summed over. 
    Pads the filter before the signal, which ensures no artifacts due
    to circular convolution. 
    
    Parameters
    ----------
    time_filter : ndarray
        The filter. The last axis should correspond to time
    time_signal : ndarray
        The signal. The last axis should correspond to time
        Should be exactly twice as long as the filter
    
    Returns
    -------
    correlation : ndarray
        The linear correlation between the filter and the signal
    """
    assert 2 * time_filter.shape[-1] == time_signal.shape[-1]
    # assert(timeFilter.shape[-2] == timeSignal.shape[-2])
    freq_filter = fft(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1)
    )
    return correlate_sum_ft(freq_filter, time_signal)


def correlate_sum_ft(freq_filter, time_signal):
    """Computes linear correlation between a time-domain signal and a frequency-domain filter
    
    Last dimension of the signal is summed over.
    Next to last dimension of the filter is summed over.

    Parameters
    ----------
    freq_filter : ndarray
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response before the impulse response. 
    time_signal : ndarray
        The signal. The last axis should correspond to time. 
        Should be exactly twice as long as the filter.
    
    Returns
    -------
    correlation : ndarray
        The linear correlation between the filter and the signal
    """
    freq_signal = fft(time_signal)
    return correlate_sum_ff(freq_filter, freq_signal)

def correlate_sum_tf(time_filter, freq_signal):
    """Computes linear correlation between a frequency-domain signal and a time-domain filter
    
    Last dimension of the filter is summed over.
    Next to last dimension of the signal is summed over.

    Parameters
    ----------
    time_filter : ndarray
        The filter. The last axis should correspond to time.
    freq_signal : ndarray
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should 
        have been twice as long as the filter
        
    Returns
    -------
    correlation : ndarray
        The linear correlation between the filter and the signal
    """
    freq_filter = fft(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1)
    )
    return correlate_sum_ff(freq_filter, freq_signal)


def correlate_sum_ff(freq_filter, freq_signal):
    """Computes linear correlation between two frequency-domain signals
    
    The last dimension is summed over. 

    Parameters
    ----------
    freq_filter : ndarray
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response before the impulse response.
    freq_signal : ndarray
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should 
        have been twice as long as the impulse response of the filter.
    
    Returns
    -------
    correlation : ndarray   
        The linear correlation between the filter and the signal
    """
    assert freq_filter.shape[0] == freq_signal.shape[0]
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2
    filtered_signal = ifft(np.sum(freq_filter * freq_signal.conj(), axis=-1))
    return np.real_if_close(filtered_signal[..., :output_len])


def correlate_cartesian_tt(time_filter, time_signal):
    """Computes the linear correlation between two time-domain signals

    Computes the correlation for all combinations of channels, as in
    the cartesian product of the channels.
    
    Parameters
    ----------
    time_filter : ndarray
        The filter. The last axis should correspond to time
    time_signal : ndarray
        The signal. The last axis should correspond to time
        Should be exactly twice as long as the filter. It should be 
        twice as long as the impulse response of the filter. 
    
    Returns
    -------
    correlation : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The linear correlation between the filter and the signal
    """
    assert 2 * time_filter.shape[-1] == time_signal.shape[-1]
    freq_filter = fft(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1)
    )
    return correlate_euclidian_ft(freq_filter, time_signal)


def correlate_euclidian_ft(freq_filter, time_signal):
    """Correlates every channel of input with every channel of the filter

    Parameters
    ----------
    freq_filter : ndarray
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *before* the impulse response.
    time_signal : ndarray
        The signal. The last axis should correspond to time.
        Should be exactly twice as long as the impulse response of the filter.

    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The linear correlation between the filter and the signal
    """
    freq_signal = fft(time_signal)
    return correlate_euclidian_ff(freq_filter, freq_signal)

def correlate_euclidian_tf(time_filter, freq_signal):
    """Correlates every channel of input with every channel of the filter
    
    Parameters
    ----------
    time_filter : ndarray of shape (f_1, f_2, ..., ir_len)
        The filter. The last axis should correspond to time.
    freq_signal : ndarray of shape (num_freq, s_1, s_2, ..., s_n)
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should
        have been twice as long as the impulse response of the filter.
    
    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The linear correlation between the filter and the signal
    """
    freq_filter = fft(
        np.concatenate((np.zeros_like(time_filter), time_filter), axis=-1)
    )
    return correlate_euclidian_ff(freq_filter, freq_signal)


def correlate_euclidian_ff(freq_filter, freq_signal):
    """Correlates every channel of input with every channel of the filter

    Parameters
    ----------
    freq_filter : ndarray of shape (num_freq, f_1, f_2, ..., f_n)
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *before* the impulse response.
    freq_signal : ndarray of shape (num_freq, s_1, s_2, ..., s_n)
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should
        have been twice as long as the impulse response of the filter.
    
    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The linear correlation between the filter and the signal
    """
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = (
        freq_filter.reshape(freq_filter.shape + (1,) * (freq_signal.ndim - 1))
        * freq_signal.reshape(
            freq_signal.shape[0:1] + (1,) * (freq_filter.ndim - 1) + freq_signal.shape[1:]
        ).conj()
    )
    filtered_signal = ifft(filtered_signal)
    return np.real_if_close(filtered_signal[..., :output_len])



def convolve_sum(freq_filter, time_signal):
    """Performs linear convolution between a time-domain signal and a frequency-domain filter
    
    The last dimension of the filter is summed over.
    The next to last dimension of the signal is summed over.

    Parameters
    ----------
    freq_filter : ndarray
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *after* the impulse response.
    time_signal : ndarray
        The signal. The last axis should correspond to time.
        Should be exactly twice as long as the impulse response of the filter.
    
    Returns
    -------
    filtered_signal : ndarray
        The signal filtered through the frequency domain filter
    """
    assert freq_filter.shape[-1] == time_signal.shape[-2]
    assert freq_filter.shape[0] == time_signal.shape[-1]
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = freq_filter @ fft(time_signal)[...,None]
    filtered_signal = np.squeeze(ifft(filtered_signal), axis=-1)
    return np.real_if_close(filtered_signal[..., output_len:])


def convolve_euclidian_ff(freq_filter, freq_signal):
    """Convolves every channel of input with every channel of the filter
    
    Parameters
    ----------
    freq_filter : ndarray of shape (num_freq, f_1, f_2, ..., f_n)
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *after* the impulse response.
    freq_signal : ndarray of shape (num_freq, s_1, s_2, ..., s_n)
        The signal. The first axis should correspond to frequency.
        Before being transformed into the frequency domain, it should
        have been twice as long as the impulse response of the filter.

    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The signal filtered through the frequency domain filter
    """
    assert freq_filter.shape[0] % 2 == 0
    output_len = freq_filter.shape[0] // 2

    filtered_signal = freq_filter.reshape(
        freq_filter.shape + (1,) * (freq_signal.ndim - 1)
    ) * freq_signal.reshape(
        freq_signal.shape[0:1] + (1,) * (freq_filter.ndim - 1) + freq_signal.shape[1:]
    )
    filtered_signal = ifft(filtered_signal)
    return np.real_if_close(filtered_signal[..., output_len:])


def convolve_euclidian_ft(freq_filter, time_signal):
    """Convolves every channel of input with every channel of the filter
    
    Parameters
    ----------
    freq_filter : ndarray of shape (num_freq, f_1, f_2, ..., f_n)
        The filter. The first axis should correspond to frequency.
        Before it was transformed, it should have been padded with zeros
        equal to the length of the impulse response *after* the impulse response.
    time_signal : ndarray of shape (s_1, s_2, ..., s_n, num_samples)
        The signal. The last axis should correspond to time.
        Should be exactly twice as long as the impulse response of the filter.

    Returns
    -------
    filtered_signal : ndarray of shape (filter.shape[:-1], signal.shape[:-1], num_samples)
        The signal filtered through the frequency domain filter
    
    """
    assert freq_filter.shape[0] == time_signal.shape[-1]

    freq_signal = fft(time_signal)
    return convolve_euclidian_ff(freq_filter, freq_signal)
