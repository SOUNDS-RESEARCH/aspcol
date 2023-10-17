"""Functions associated with perfect periodic sequences

Such sequences are deterministic periodic sequences with an impulse as periodic autocorrelation. 
"""
import numpy as np
import scipy.linalg as splin
import matplotlib.pyplot as plt
import itertools as it

import aspcol.correlation as cr



def decorrelate(sig, pseq, v=0):
    """
    Can be used to identify a LTI system from a PSEQ input signal

    Parameters
    ----------
    sig : ndarray of shape (num_channels, num_samples)
        the signal that has been convolved with the system
    pseq : ndarray of shape (1, num_samples,)
        the perfect sequence that was used as input to the system
    v : int
        index between 0 and PSEQ period length
        declares which index should be considered the start of the sequence

    Returns
    -------
    ir : ndarray of shape (num_channels, num_samples)
        the estimated impulse response of the system

    """
    assert sig.ndim == 2
    assert sig.shape[-1] == pseq.shape[-1]
    if pseq.ndim == 2:
        pseq = np.squeeze(pseq, axis=0)

    normalize_factor = np.sum(pseq**2)
    p_n = np.flip(np.roll(pseq, -1-v))
    p_n = p_n / normalize_factor

    pn_rev = np.concatenate((np.array([0]), np.flip(p_n[1:])))
    rir_est = splin.matmul_toeplitz((p_n, pn_rev), sig.T)
    return rir_est.T


def create_pseq(seq_len : int):
    """
    Creates a perfect sweep with constant magnitude spectrum
    Adapted from MATLAB code by Aulis Telle, IND, 2008

    Parameters
    ----------
    seq_len : int
        length of the sequence in samples

    Returns
    -------
    seq : ndarray of shape (1, seq_len)
    """
    is_even = seq_len % 2 == 0
    N_half = int(np.ceil(seq_len / 2))
    ph = np.zeros((seq_len))

    group_delay = np.arange(N_half) / (1/2)
    delta_omega = 2 * np.pi * 1 / seq_len
    ph[:N_half] = - group_delay * np.arange(N_half) * delta_omega / 2

    if is_even:
        ph[N_half:] = np.concatenate((np.array([0]), -np.flip(ph[1:N_half])))
    else:
        ph[N_half:] = -np.flip(ph[1:N_half])

    c = 10*np.exp(1j*ph)
    s = np.real_if_close(np.fft.ifft(c, axis=-1))

    # Normalize signal to have max amplitude 1
    s = s / np.max(np.abs(s))
    return s[None,:]
    
def create_pseq_lowfreq(seq_len : int, sr : int, max_pseq_freq : int):
    """
    Creates pseq with a maximum frequency to be used at a higher samplerate

    Parameters
    ----------
    seq_len : int
    sr : int
        samplerate where the sequence should be used
    max_pseq_freq : int
        could in theory be a float, but is assumed to be integer for simplicity

    Returns
    -------
    seq : ndarray of shape (short_seq_len, )
    
    
    """
    import samplerate as sr_convert
    low_sr = 2 * max_pseq_freq
    assert low_sr <= sr
    if low_sr == sr:
        return create_pseq(seq_len)
    sampling_factor = sr / low_sr
    if sr % low_sr != 0:
        print(f"Warning: resampling by non-integer in create_pseq_lowfreq")
    if seq_len % sampling_factor != 0:
        print(f"Warning: seq_len can not be set exactly in create_pseq_lowfreq")
    
    seq_len_short = int(seq_len / sampling_factor)

    seq_low_sr = create_pseq(seq_len_short)[0,:]
    seq_low_sr = np.tile(seq_low_sr, 3)
    seq_upsampled = sr_convert.resample(seq_low_sr, sampling_factor)
    
    seq_len_upsampled = len(seq_upsampled) // 3
    seq_upsampled = seq_upsampled[seq_len_upsampled:2*seq_len_upsampled]
    return seq_upsampled[None,:]

def create_shifted_pseq(pseq, num_channels, rir_len):
    """
    For system identification of MISO system, each source should be
    given a shifted version of the same sequence to be able to 
    perfectly identify each channel.

    Parameters
    ----------
    pseq : ndarray of shape (1, rir_len*num_channels) or (rir_len*num_channels)
        the perfect sequence to be shifted
    num_channels : int
    rir_len : int

    Returns
    -------
    shifted_pseq : ndarray of shape (num_channels, rir_len*num_channels)
        source l should be assigned the signal shifted_pseq[l,:]
    """
    if pseq.ndim == 1:
        pseq = pseq[None,:]
    assert pseq.shape == (1, rir_len * num_channels)
    shifted_pseq = np.concatenate([np.roll(pseq, i*rir_len) for i in range(num_channels)], axis=0)
    return shifted_pseq


def verify_pseq(pseq, plot=False, samples_to_show=300):
    """
    Used to verify that a sequence is a perfect sequence
    
    Parameters
    ----------
    pseq : ndarray of shape (1, pseq_len)
        the perfect sequence to be verified
    plot : bool
        choose true to plot the sequence and its autocorrelation
    samples_to_show : int
        number of samples to show in the zoomed in plot

    Returns
    -------
    info : dict
        contains information about the sequence
    """
    if pseq.ndim == 1:
        pseq = pseq[None,:]

    autocorr = cr.periodic_autocorr(pseq)

    info = {
        "PSEQ Length (samples)" : pseq.shape[1],
        "Maximum amplitude": np.max(np.abs(pseq)),
        "Average power" : np.mean(np.abs(pseq)**2),
        "Perfect periodic autocorrelation" : np.abs(autocorr[0]) > 1e-3 and np.allclose(autocorr[1:], 0),
    }

    if plot:
        plt.figure()
        plt.plot(pseq[0,:])
        plt.title("Perfect sequence")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        plt.figure()
        joined_signal = np.concatenate((pseq[0,-samples_to_show:], pseq[0,:samples_to_show]))
        plt.plot(np.arange(-samples_to_show, samples_to_show), joined_signal)
        plt.xlabel("Samples (0 is new period)")
        plt.ylabel("Amplitude")
        plt.title("Close look at point of new period")

        plt.figure()
        plt.plot(autocorr)
        plt.title("Periodic autocorrelation")
        plt.xlabel("Lag (samples)")
        plt.ylabel("Correlation")
        plt.show()
    return info

