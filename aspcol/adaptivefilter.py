"""Implements several basic adaptive filters

* Least mean squares (LMS)
* Recursive least squares (RLS)
* Fast block LMS
* Fast block weighted LMS

References
----------
`[1] <https://link.springer.com/book/10.1007/978-3-030-29057-3>`_ P. S. R. Diniz, Adaptive filtering: algorithms and practical implementation. Cham: Springer International Publishing, 2020. doi: 10.1007/978-3-030-29057-3.

"""
import numpy as np
from abc import ABC, abstractmethod

import aspcore.filterclasses as fc
import aspcore.freqdomainfiltering as fdf
import aspcol.utilities as util


class AdaptiveFilterBase(ABC):
    def __init__(self, ir_len, num_in, num_out, filter_type=None):
        filter_dim = (num_in, num_out, ir_len)
        self.num_in = num_in
        self.num_out = num_out
        self.ir_len = ir_len

        if filter_type is not None:
            self.filt = filter_type(ir=np.zeros(filter_dim))
        else:
            self.filt = fc.FilterSum(ir=np.zeros(filter_dim))

        self.x = np.zeros((self.num_in, self.ir_len, 1))  # Column vector

        self.metadata = {
            "name" : self.__class__.__name__,
            "ir length" : self.ir_len, 
            "num input channels" : self.num_in,
            "num output channels" : self.num_out
        }


    def insert_in_signal(self, ref):
        assert ref.shape[-1] == 1
        self.x = np.roll(self.x, 1, axis=1)
        self.x[:, 0:1, 0] = ref

    def prepare(self):
        pass

    @property
    def ir(self):
        return self.filt.ir

    @abstractmethod
    def update(self):
        pass

    def process(self, signalToProcess):
        return self.filt.process(signalToProcess)

    def set_ir(self, new_ir):
        if new_ir.shape != self.filt.ir.shape:
            self.num_in = new_ir.shape[0]
            self.num_out = new_ir.shape[1]
            self.ir_len = new_ir.shape[2]
        self.filt.set_ir(new_ir)



class AdaptiveFilterFreq(ABC):
    def __init__(self, num_freq, num_in, num_out):
        assert num_freq % 2 == 0
        self.num_in = num_in
        self.num_out = num_out
        self.num_freq = num_freq

        self.filt = fc.FilterSum_Freqdomain(numFreq=num_freq, num_in=num_in, num_out=num_out)

    @abstractmethod
    def update(self):
        pass

    def process(self, sig_to_process):
        return self.filt.process(sig_to_process)
    
    def process_nosum(self, sig_to_process):
        return self.filt.process_nosum(sig_to_process)


class LMS(AdaptiveFilterBase):
    """Sample by sample processing, although it accepts block inputs
    - Dimension of filter is (input channels, output channels, IR length)

    - ir_len, num_in, num_out are integers
    - step_size is either a real positive scalar, or a generator that 
        outputs a real positive scalar for each sample
    - regularization is a real positive scalar
    - normalization is a string corresponding to one of the normalization options

    - If wait_until_initialized is True, the ir will not be updated until 
        the full buffer of reference signals is filled. 
    - filter_type is deprecated and will be removed later.
    """

    def __init__(
        self,
        ir_len,
        num_in,
        num_out,
        step_size,
        regularization,
        normalization="channel_independent",
        wait_until_initialized = False,
        filter_type=None,
    ):
        super().__init__(ir_len, num_in, num_out, filter_type)
        self.reg = regularization

        if isinstance(step_size, (int, float)):
            def step_size_gen():
                while True:
                    yield step_size
            self.step_size = step_size_gen()
        else: # already a generator
            self.step_size = step_size()

        if normalization == "channel_independent":
            self.norm_func = self._channel_indep_norm
        elif normalization == "channel_common":
            self.norm_func = self._channel_common_norm
        elif normalization == "none":
            self.norm_func = self._no_norm

        if wait_until_initialized:
            self.init_len = self.ir_len
        else:
            self.init_len = 0
        self.reinitialize()

        #self.metadata["step size"] = self.step_size
        self.metadata["regularization"] = self.reg
        self.metadata["normalization"] = normalization

    def reinitialize(self):
        self.phases = util.PhaseCounter({"init" : self.init_len, "processing" : np.inf})

    def _channel_indep_norm(self):
        return 1 / (self.reg + np.transpose(self.x, (0, 2, 1)) @ self.x)

    def _channel_common_norm(self):
        """
        if the two signals are identical, this is currently twice as large as _channel_indep_norm
        This allowed for step_size = 1 to be ideal step size with PSEQ training signal

        Old version was (difference is mean vs sum over channels)
        1 / (self.reg + np.mean(np.transpose(self.x, (0, 2, 1)) @ self.x))
        """
        return 1 / (self.reg + np.sum(np.transpose(self.x, (0, 2, 1)) @ self.x))
    
    def _no_norm(self):
        return 1

    def update(self, ref, desired):
        """Inputs should be of the shape (channels, num_samples)"""
        assert ref.shape[-1] == desired.shape[-1]
        num_samples = ref.shape[-1]
        desired_est = self.filt.process(ref)
        error = desired - desired_est

        for n in range(num_samples):
            self.insert_in_signal(ref[:, n : n + 1])
            if self.phases.current_phase_is("processing"):
                step_size = next(self.step_size)
                normalization = self.norm_func()
                self.filt.ir += (
                    step_size
                    * normalization
                    * np.squeeze(
                        error[None, :, None, n : n + 1] * self.x[:, None, :, :], axis=-1
                    )
                )
            self.phases.progress()

        return desired_est, error


class BlockLMS(AdaptiveFilterBase):
    """
    Block based processing and normalization
    Dimension of filter is (input channels, output channels, IR length)
    Normalizes with a scalar, common for all channels
    identical to FastBlockLMS with scalar normalization
    """

    def __init__(
        self, ir_len, num_in, num_out, stepSize, regularization=1e-3, filter_type=None
    ):
        super().__init__(ir_len, num_in, num_out, filter_type)
        self.mu = stepSize
        self.beta = regularization

    # def channelIndepNorm(self):
    #     return 1 / (self.beta + np.transpose(self.x,(0,2,1)) @ self.x)

    # def channelCommonNorm(self):
    #     return 1 / (self.beta + np.mean(np.transpose(self.x,(0,2,1)) @ self.x))

    def update(self, ref, error):
        """Inputs should be of the shape (channels, numSamples)"""
        assert ref.shape[-1] == error.shape[-1]
        numSamples = ref.shape[-1]

        grad = np.zeros_like(self.filt.ir)
        for n in range(numSamples):
            self.insert_in_signal(ref[:, n : n + 1])
            grad += np.squeeze(
                error[None, :, None, n : n + 1] * self.x[:, None, :, :], axis=-1
            )

        norm = 1 / (np.sum(ref ** 2) + self.beta)
        self.filt.ir += self.mu * norm * grad


class FastBlockLMS(AdaptiveFilterFreq):
    """Identical to BlockLMS when scalar normalization is used"""
    def __init__(
        self,
        block_size,
        num_in,
        num_out,
        step_size,
        regularization=1e-3,
        power_est_forget_factor=0.99,
        normalization="scalar",
    ):
        super().__init__(2 * block_size, num_in, num_out)
        self.mu = step_size
        self.beta = regularization
        self.block_size = block_size

        if normalization == "scalar":
            self.norm_func = self.scalar_normalization
        elif normalization == "freq_independent":
            self.ref_power_estimate = fc.MovingAverage(power_est_forget_factor, (2 * block_size, 1, 1))
            self.norm_func = self.freq_independent_normalization
        elif normalization == "channel_independent":
            self.norm_func = self.channel_independent_normalization
        else:
            raise ValueError
            
    def scalar_normalization(self, ref, freqRef):
        return 1 / (np.sum(ref[...,self.block_size:]**2) + self.beta)

    def freq_independent_normalization(self, ref, freqRef):
        self.ref_power_estimate.update(np.sum(np.abs(freqRef[:,:,None])**2, axis=(1,2), keepdims=True))
        return 1 / (self.ref_power_estimate.state + self.beta)
    
    def channel_independent_normalization(self, ref, freqRef):
        return 1 / (np.mean(np.abs(X)**2,axis=0, keepdims=True) + self.beta)


    def update(self, ref, error):
        """ref is two blocks, the latter of which 
            corresponds to the single error block"""
        assert ref.shape == (self.num_in, 2 * self.block_size)
        assert error.shape == (self.num_out, self.block_size)

        X = fdf.fft_transpose(ref)
        td_grad = fdf.correlate_euclidian_tf(error, X)
        gradient = fdf.fft_transpose(np.concatenate((td_grad, np.zeros_like(td_grad)),axis=-1))
        norm = self.norm_func(ref, X)
        
        self.filt.tf += self.mu * norm * gradient


class FastBlockWeightedLMS(FastBlockLMS):
    def __init__(
        self,
        blockSize,
        num_in,
        num_out,
        step_size,
        weight_matrix,
        regularization=1e-3,
        freq_indep_norm=False,
    ):
        super().__init__(
            blockSize, num_in, num_out, step_size, regularization, freq_indep_norm
        )
        self.weight_matrix = weight_matrix

    def update(self, ref, error):
        assert ref.shape == (self.num_in, 2 * self.block_size)
        assert error.shape == (self.num_out, self.block_size)

        padded_error = np.concatenate((np.zeros_like(error), error), axis=-1)
        E = np.fft.fft(padded_error, axis=-1).T[:, :, None]
        X = np.fft.fft(ref, axis=-1).T[:, :, None]

        gradient = self.weight_matrix @ E @ np.transpose(X.conj(), (0, 2, 1))
        td_grad = np.fft.ifft(gradient, axis=0)
        td_grad[self.block_size :, :, :] = 0
        gradient = np.fft.fft(td_grad, axis=0)

        norm = self.norm_func(X)
        self.ir += self.mu * norm * gradient



class RLS(AdaptiveFilterBase):
    """Dimension of filter is (input channels, output channels, IR length)"""

    def __init__(
        self, 
        ir_len, 
        num_in, 
        num_out, 
        forget_factor, 
        signal_power_est,
        wait_until_initialized = False,
        ):

        super().__init__(ir_len, num_in, num_out)
        self.flat_ir_len = self.ir_len * self.num_in

        self.forget_factor = forget_factor
        self.forget_inv = 1 - forget_factor
        self.signal_power_est = signal_power_est

        self.inv_corr = np.zeros((self.flat_ir_len, self.flat_ir_len))
        self.inv_corr[:, :] = np.eye(self.flat_ir_len) * signal_power_est
        self.gain = np.zeros((self.num_in*self.ir_len, 1))

        if wait_until_initialized:
            init_len = self.ir_len
        else:
            init_len = 0
        self.phases = util.PhaseCounter({"init" : init_len, "processing" : np.inf})

        self.metadata["forget factor"] = self.forget_factor
        self.metadata["signal power estimate"] = self.signal_power_est

    def update(self, ref, desired):
        assert ref.shape[-1] == desired.shape[-1]
        num_samples = ref.shape[-1]
        for i in range(num_samples):
            mic_est = self.filt.process(ref[:,i:i+1])
            error = desired[:,i:i+1] - mic_est
            self.insert_in_signal(ref[:, i:i+1])
            if self.phases.current_phase_is("processing"):
                X = self.x.reshape((-1, 1))
                #X = np.flip(self.sig["ls"][:,i-self.rir_len+1:i+1],axis=-1).reshape(-1,1)
                x_times_corr = self.inv_corr @ X

                self.gain = x_times_corr / (self.forget_factor + X.T @ x_times_corr)
                self.inv_corr = (self.inv_corr - self.gain @ x_times_corr.T) / self.forget_factor
                self.filt.ir += self.gain.reshape(self.num_in, 1, self.ir_len) * error[None,:,:]
            self.phases.progress()


