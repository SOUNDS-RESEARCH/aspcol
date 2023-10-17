"""Utility classes for filtering

Implementations for standard linear convolution can be found in scipy.signal and aspcore. 

* Weighted overlap-add (WOLA) [1,2]
* IIR filter
* Mean with forgetting factor

`[1] <doi.org/10.1109/TASSP.1980.1163353>`_ R. Crochiere, “A weighted overlap-add method of short-time Fourier analysis/synthesis,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 28, no. 1, pp. 99–102, Feb. 1980, doi: 10.1109/TASSP.1980.1163353.
`[2] <doi.org/10.23919/EUSIPCO54536.2021.9616352>`_ S. Ruiz, T. Dietzen, T. van Waterschoot, and M. Moonen, “A comparison between overlap-save and weighted overlap-add filter banks for multi-channel Wiener filter based noise reduction,” in 2021 29th European Signal Processing Conference (EUSIPCO), Aug. 2021, pp. 336–340. doi: 10.23919/EUSIPCO54536.2021.9616352. 
"""
import numpy as np
import scipy.signal as spsig
import numexpr as ne

import aspcol.utilities as util

class MovingAverage:
    def __init__(self, forget_factor, dim, dtype=np.float64):
        self.state = np.zeros(dim, dtype=dtype)
        self.forget_factor = forget_factor
        self.forget_factor_inv = 1 - forget_factor

        self.initialized = False
        self.init_counter = 0
        if self.forget_factor == 1:
            self.num_init = np.inf
        else:
            self.num_init = int(np.ceil(1 / self.forget_factor_inv))

    def update(self, new_data_point, count_as_updates=1):
        """
        
        count_as_updates can be used if the datapoint is already average
        outside of this class. So if new_data_point is the average of N data 
        points, count_as_updates should be set to N.
        """
        assert new_data_point.shape == self.state.shape
        if self.initialized:
            if count_as_updates > 1:
                raise NotImplementedError

            self.state[...] = ne.evaluate("state*ff + new_data_point*ff_inv", 
                                local_dict={"state":self.state, "new_data_point":new_data_point,
                                            "ff":self.forget_factor, "ff_inv":self.forget_factor_inv})
            #self.state *= self.forget_factor
            #self.state += new_data_point * self.forget_factor_inv
        else:
            self.state[...] = ne.evaluate("state*(i/(i+j)) + new_data_point*(j/(i+j))", 
                                local_dict={'state': self.state, "new_data_point":new_data_point, 
                                            'i': self.init_counter, "j" : count_as_updates})

            #self.state *= (self.init_counter / (self.init_counter + 1))
            #self.state += new_data_point / (self.init_counter + 1)
            self.init_counter += count_as_updates
            if self.init_counter >= self.num_init:
                self.initialized = True
                if self.init_count > self.num_init:
                    print("Initialization happened not exactly at self.num_init")

    def reset(self):
        self.initialized = False
        self.num_init = np.ceil(1 / self.forget_factor_inv)
        self.init_counter = 0




class IIRFilter:
    """
    num_coeffs and denom_coeffs should be a list of ndarrays,
        containing the parameters of the rational transfer function
        If only one channel is desired, the arguments can just be a ndarray
    """
    def __init__(self, num_coeffs, denom_coeffs):

        if not isinstance(num_coeffs, (list, tuple)):
            assert not isinstance(denom_coeffs, (list, tuple))
            num_coeffs = [num_coeffs]
            denom_coeffs = [denom_coeffs]
        assert isinstance(denom_coeffs, (list, tuple))
        self.num_coeffs = num_coeffs
        self.denom_coeffs = denom_coeffs

        self.num_channels = len(self.num_coeffs)
        assert len(self.num_coeffs) == len(self.denom_coeffs)

        self.order = [max(len(nc), len(dc)) for nc, dc in zip(self.num_coeffs, self.denom_coeffs)]
        self.filter_state = [spsig.lfiltic(nc, dc, np.zeros((len(dc)-1))) 
                                        for nc, dc in zip(self.num_coeffs, self.denom_coeffs)]

    def process(self, data_to_filter):
        assert data_to_filter.ndim == 2
        num_channels = data_to_filter.shape[0]
        filtered_sig = np.zeros_like(data_to_filter)
        for ch_idx in range(num_channels):
            filtered_sig[ch_idx,:], self.filter_state[ch_idx] = spsig.lfilter(self.num_coeffs[ch_idx], self.denom_coeffs[ch_idx], data_to_filter[ch_idx,:], axis=-1, zi=self.filter_state[ch_idx])
        return filtered_sig







class WOLA:
    def __init__(self, num_in : int, num_out : int, block_size : int, overlap : int):
        self.num_in = num_in
        self.num_out = num_out
        self.block_size = block_size
        self.overlap = overlap
        self.hop = block_size - overlap
        assert self.hop > 0

        if self.block_size % 2 == 0:
            self.num_freqs = 1 + self.block_size // 2
        else:
            raise NotImplementedError

        self.win = get_window_wola(self.block_size, self.overlap)
        #self.win = np.sqrt(1/2)*np.ones(self.block_size)
        self.buf_in = np.zeros((self.num_in, self.overlap), dtype=float)
        self.buf_out = np.zeros((self.num_in, self.num_out, self.overlap), dtype=float)
        self.spectrum = np.zeros((self.num_in, self.num_out, self.num_freqs), dtype=complex)

    def analysis(self, sig):
        """Performs WOLA analysis and saved the contents to self.spectrum 
        
        Parameters
        ----------
        sig : ndarray (self.num_in, self.hop)
            the new samples from the signal that should be analyzed
        """
        assert sig.ndim == 2
        assert sig.shape[-1] == self.hop
        if sig.shape[0] == 1:
            sig = np.broadcast_to(sig, (self.num_in, self.hop))

        sig_to_analyze = np.concatenate((self.buf_in, sig), axis=-1)
        self.buf_in[...] = sig_to_analyze[:,-self.overlap:]

        self.spectrum[...] = wola_analysis(sig_to_analyze, self.win)[:,None,:]
        

    def synthesis(self):
        """Performs WOLA synthesis, sums with previous blocks and returns 
            the self.hop number of valid samples 
        
        Parameters
        ----------

        Return
        ------


        """
        sig = wola_synthesis(self.spectrum, self.buf_out, self.win, self.overlap)

        #sig[:,:self.overlap] += self.buf_out
        self.buf_out[...] = sig[...,-self.overlap:]

        #complete_sig = sig[:,:self.hop] += sig_last_block
        return sig[...,:self.hop]



def get_window_wola(block_size : int, overlap : int):
    """
    block_size is number of samples
    overlap is number of samples
    """
    win = spsig.windows.hann(block_size, sym=False)
    assert spsig.check_COLA(win, block_size, overlap)
    return np.sqrt(win)

def wola_analysis(sig, window):
    """Generate WOLA spectrum from time domain signal

    Parameters
    ----------
    sig : ndarray (num_channels, num_samples)
    window : ndarray (num_samples)

    Returns
    -------
    spectrum : ndarray (num_channels, num_samples//2 + 1)
    """
    num_samples = sig.shape[-1]
    assert sig.ndim == 2
    assert window.ndim == 1
    assert window.shape[0] == num_samples
   
    spectrum = np.fft.rfft(sig * window[None,:], axis=-1)
    return spectrum

def wola_synthesis(spectrum, sig_last_block, window, overlap):
    """Generate time domain signal from WOLA spectrum
        
    Keep in mind that only the first block_size-overlap are correct
    the last overlap samples should be saved until last block to 
    be overlapped with the next block

    Parameters
    ----------
    spectrum : ndarray (..., num_channels, num_samples//2 + 1)
        the spectrum associated with the positive frequencies (output from rfft), 
        which will be num_samples // 2 + 1 frequencies. 
    sig_last_block : ndarray (..., num_channels, overlap)
    window : ndarray (num_samples)
    overlap : int 

    Returns
    -------
    sig : ndarray (..., num_channels, num_samples)
    """
    assert spectrum.ndim >= 2
    assert sig_last_block.ndim >= 2
    assert window.ndim == 1

    block_size = window.shape[0]
    num_channels = spectrum.shape[-2]

    assert spectrum.shape[:-2] == sig_last_block.shape[:-2]
    assert spectrum.shape[-1] == block_size // 2 + 1
    assert sig_last_block.shape[-2:] == (num_channels, overlap)
    
    if overlap != block_size // 2:
        raise NotImplementedError
    if not util.is_power_of_2(block_size):
        raise NotImplementedError

    win_broadcast_dims = spectrum.ndim - 1
    window = window.reshape(win_broadcast_dims*(1,) + (-1,))

    sig = window * np.real_if_close(np.fft.irfft(spectrum, axis=-1))
    
    sig[...,:overlap] += sig_last_block
    return sig