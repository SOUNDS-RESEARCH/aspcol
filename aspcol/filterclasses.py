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




class SinglePoleLowPass:
    def __init__(self, forget_factor, dim, dtype=np.float64):
        self.state = np.zeros(dim, dtype=dtype)
        self.forget_factor = forget_factor
        self.inv_forget_factor = 1 - forget_factor
        self.initialized = False

    def set_forget_factor(self, cutoff, samplerate):
        """Set cutoff in in Hz"""
        wc = 2 * np.pi * cutoff / samplerate
        y = 1 - np.cos(wc)
        self.forget_factor = -y + np.sqrt(y ** 2 + 2 * y)
        self.inv_forget_factor = 1 - self.forget_factor

    def update(self, new_data_point):
        assert new_data_point.shape == self.state.shape
        if self.initialized:
            self.state *= self.forget_factor
            self.state += new_data_point * self.inv_forget_factor
        else:
            self.state = new_data_point
            self.initialized = True






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
        #num_samples = data_to_filter.shape[1]
        filtered_sig = np.zeros_like(data_to_filter)
        for ch_idx in range(num_channels):
            filtered_sig[ch_idx,:], self.filter_state[ch_idx] = spsig.lfilter(self.num_coeffs[ch_idx], self.denom_coeffs[ch_idx], data_to_filter[ch_idx,:], axis=-1, zi=self.filter_state[ch_idx])
        return filtered_sig




# Free function for applying a filtersum once
# Edge effects will be present
def applyFilterSum(data, ir):
    numIn = ir.shape[0]
    numOut = ir.shape[1]
    filtLen = ir.shape[2]

    out = np.zeros((numOut, data.shape[1] + filtLen - 1))
    for outIdx in range(numOut):
        for inIdx in range(numIn):
            out[outIdx, :] += spsig.convolve(data[inIdx, :], ir[inIdx, outIdx, :], "full")
    return out




class WOLA:
    def __init__(self, num_channels : int, block_size : int, overlap : int):
        self.num_channels = num_channels
        self.block_size = block_size
        self.overlap = overlap
        self.hop = block_size - overlap
        assert self.hop > 0

        self.win = get_window_wola(block_size, overlap)
        #self.buf_in = np.zeros((num_channels, overlap), dtype=float)
        self.buffer = np.zeros((num_channels, overlap), dtype=float)
        self.spectrum = np.zeros((num_channels, block_size), dtype=complex)

    def analysis(self, sig):
        self.spectrum[...] = wola_analysis(sig, self.win)

    def synthesis(self):
        sig = wola_synthesis(self.spectrum, self.buffer, self.win, self.overlap)
        self.buffer[...] = sig[:,-self.overlap:]
        return sig



def get_window_wola(block_size : int, overlap : int):
    """
    block_size is number of samples
    overlap is number of samples
    """
    win = spsig.windows.hann(block_size, sym=False)
    assert spsig.check_COLA(win, block_size, overlap)
    return np.sqrt(win)

def wola_analysis(sig, window):
    """
    sig is (num_channels, num_samples)
    window is (num_samples)
    
    """
    num_samples = sig.shape[-1]
    assert sig.ndim == 2
    assert window.ndim == 1
    assert window.shape[0] == num_samples
   
    spectrum = np.fft.fft(sig * window[None,:], axis=-1)
    return spectrum

def wola_synthesis(spectrum, sig_last_block, window, overlap):
    """Generate time domain signal from WOLA spectrum

    Parameters
    ----------
    spectrum : ndarray (num_channels, num_samples)
    sig_last_block : ndarray (num_channels, overlap)
    window : ndarray (num_samples)
    overlap : int 

    Returns
    -------
    sig : ndarray (num_channels, num_samples)
    """
    assert spectrum.ndim == 2
    assert sig_last_block.ndim == 2
    assert window.ndim == 1
    block_size = spectrum.shape[-1]
    num_channels = spectrum.shape[0]
    assert sig_last_block.shape[0] == num_channels
    assert window.shape[0] == block_size
    
    if overlap != block_size // 2:
        raise NotImplementedError
    if not util.is_power_of_2(block_size):
        raise NotImplementedError 

    sig = window[None,:] * np.fft.ifft(spectrum, axis=-1)
    
    sig[:,:overlap] += sig_last_block
    return sig