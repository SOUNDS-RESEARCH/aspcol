import numpy as np
import scipy.signal as signal
import itertools as it


# =======================================================================

def get_frequency_values(num_freq, samplerate):
    """Get the frequency values of all positive frequency bins in Hz
    numFreq is the number of frequency bins INCLUDING negative frequencies
    If numFreq is even, it will return (numFreq/2)+1 values
    If numFreq is odd, it will return (numFreq+1)/2 values
    use np.fft.fftfreq to double check values when implementing. 
    """
    if num_freq % 2 == 0:
        return (samplerate / (num_freq)) * np.arange(num_freq // 2 + 1)
    elif num_freq % 2 == 1:
        raise NotImplementedError
    else:
        raise ValueError


def insert_negative_frequencies(freq_signal, even):
    """To be used in conjunction with getFrequencyValues
    Inserts all negative frequency values under the
    assumption of conjugate symmetry, i.e. a real impulse response.
    Parameter even: boolean indicating if an even or odd number
    of bins is desired. This must correspond to num_freq value
    set in get_frequency_values
    
    Frequencies must be on axis=0"""
    if even:
        return np.concatenate(
            (freq_signal, np.flip(freq_signal[1:-1, :, :].conj(), axis=0)), axis=0
        )
    else:
        raise NotImplementedError

def fir_from_freqs_window(freq_filter, ir_len, two_sided=True):
    """Use this over the other window methods,
    as they might be wrong. freqFilter is with both positive
    and negative frequencies.

    Makes FIR filter from frequency values.
    Works only for odd impulse response lengths
    Uses hamming window"""
    assert ir_len % 1 == 0
    assert freq_filter.shape[0] % 2 == 0
    if two_sided:
        #halfLen = irLen // 2
        mid_point = freq_filter.shape[0] // 2

        time_filter = np.fft.ifft(freq_filter, axis=0)
        new_axis_order = np.concatenate((np.arange(1, freq_filter.ndim), [0]))
        time_filter = np.real_if_close(np.transpose(time_filter, new_axis_order))

        #truncError = calcTruncationError(fullTimeFilter, irLen, twoSided)
        time_filter = np.concatenate((time_filter[...,-mid_point:], time_filter[...,:mid_point]), axis=-1)

        #truncFilter = timeFilter[..., midPoint-halfLen:midPoint+halfLen+1]
        trunc_filter, trunc_error = truncate_filter(time_filter, ir_len, True)
        
        # timeFilter = np.concatenate(
        #     (fullTimeFilter[..., -halfLen:], fullTimeFilter[..., : halfLen + 1]), axis=-1
        # )


        # ONLY TEMPORARILY COMMENTED. THE WINDOW CODE HERE REALLY WORKS. 
        # ADD A BOOLEAN ARGUMENT INSTEAD
        #truncFilter = truncFilter * signal.windows.hamming(irLen).reshape(
        #    (1,) * (truncFilter.ndim - 1) + truncFilter.shape[-1:]
        #)
        
    else:
        raise NotImplementedError

    return trunc_filter, trunc_error

def truncate_filter(ir, ir_len, two_sided):
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
    ir has shape (..., irLength)
    if twosided, the ir will be assumed centered in the middle.
    The filter can be multidimensional, the minimum length will
    be calculated independently for all impulse responses, and
    the longest length chosen. The relative error is how much of the
    power of the impulse response that is lost by truncating."""
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