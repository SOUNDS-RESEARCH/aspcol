"""Implements autocorrelation and cross-correlation estimators, as well as related useful calculations. 

* Estimation of correlation function and covariance matrix of signal.  
* Sample covariance for vector-valued variable, including the non-zero mean case, even in a streaming processing.  
* Covariance estimation with optimal linear shrinkage [1] and almost optimal non-linear shrinkage [2],


References
----------
`[1] <doi.org/10.1109/TSP.2010.2053029>`_ Y. Chen, A. Wiesel, Y. C. Eldar, and A. O. Hero, “Shrinkage Algorithms for MMSE Covariance Estimation,” IEEE Trans. Signal Process., vol. 58, no. 10, pp. 5016–5029, Oct. 2010, doi: 10.1109/TSP.2010.2053029. 
`[2] <doi.org/10.5167/UZH-176887>`_ O. Ledoit and M. Wolf, “Quadratic shrinkage for large covariance matrices,” Dec. 2020, doi: 10.5167/UZH-176887. 
"""


import numpy as np
import scipy.signal as spsig

import aspcol.matrices as mat
import aspcol.filterclasses as fc


def cov_est_oas(sample_cov, n, verbose=False):
    """
    Implements the OAS covariance estimator with linear shrinkage from
    Shrinkage Algorithms for MMSE Covariance Estimation

    n is the number of sample vectors that the sample covariance is 
    constructed from

    Assumes Gaussian sample distribution
    """
    assert sample_cov.ndim == 2
    assert sample_cov.shape[0] == sample_cov.shape[1]
    p = sample_cov.shape[-1]
    tr_s = np.trace(sample_cov)
    tr_s_square = np.trace(sample_cov @ sample_cov)

    rho_num = ((-1) / p) *  tr_s_square + tr_s**2
    rho_denom = ((n - 1) / p) * (tr_s_square - (tr_s**2 / p))
    rho = rho_num / rho_denom
    rho_oas = min(rho, 1)

    reg_factor = rho_oas * tr_s / p
    cov_est = (1-rho_oas) * sample_cov + reg_factor * np.eye(p)

    if verbose:
        print(f"OAS covariance estimate is {1-rho_oas} * sample_cov + {reg_factor} * I")
    return cov_est

def cov_est_qis(sample_cov, n):
    """
    Implements the quadratic-inverse shrinkage, the non-linear shrinkage from 
    "Quadratic shrinkage for large covariance matrices" by Ledoit and Wolf. 

    Sample vectors in the sample covariance matrix should be i.i.d and zero mean. 
    
    Code adapted from https://github.com/pald22/covShrinkage by Patrick Ledoit
    It is here rewritten to use only numpy instead of pandas.

    Parameters
    ----------
    sample_cov : ndarray of shape (p,p)
        assumes the matrix is ensure to be symmetric positive semidefinite
    n  : int
        the sample size of the covariance matrix. If the sample mean was subtracted 
        (if the variables were not zero mean), the sample size should be adjusted to n-1.

    Returns
    -------
    cov_estimate : ndarray of shape (p,p)
    """
    p = sample_cov.shape[0]
    assert sample_cov.shape == (p,p)
    c = p/n
    h = (min(c**2,1/c**2)**0.35)/p**0.35

    eigvals, eigvec = np.linalg.eigh(sample_cov)
    eigvals = eigvals.real.clip(min=0)
    
    eigval_inv = 1 / eigvals[max(1,p-n+1)-1:p]
    ev_j = np.repeat(eigval_inv[:,None], min(p,n), axis=-1)
    ev_ji = ev_j - ev_j.T

    theta = np.mean(ev_j * ev_ji / (ev_ji**2 + ev_j**2 * h**2),axis = 0)
    Htheta = np.mean(h* ev_j**2 / (ev_ji**2 + ev_j**2 * h**2),axis = 0)
    Atheta2 = theta**2 + Htheta**2

    is_full_rank = p <= n
    if is_full_rank:
        delta = 1 / ((1-c)**2*eigval_inv+2*c*(1-c)*eigval_inv*theta \
                    +c**2*eigval_inv*Atheta2)
    else:
        delta0 = 1/((c-1)*np.mean(eigval_inv))
        delta = np.repeat(delta0, p-n)
        delta = np.concatenate((delta, 1/(eigval_inv*Atheta2)), axis=None)

    deltaQIS = np.diag(delta*(sum(eigvals)/sum(delta)))
    
    cov_estimate = eigvec @ deltaQIS @ eigvec.T.conj()
    return cov_estimate

class SampleCorrelation:
    """
    Estimates the correlation matrix between two random vectors 
    v(n) & w(n) by iteratively computing (1/N) sum_{n=0}^{N-1} v(n) w^T(n) 
    The goal is to estimate R = E[v(n) w^T(n)]

    If delay is supplied, it will calculate E[v(n) w(n-delay)^T]

    Only the internal state will be changed by calling update()
    In order to update self.corr_mat, get_corr() must be called
    """
    def __init__(self, forget_factor, size, delay=0, estimate_mean=False):
        """
        size : scalar integer, correlation matrix is size x size. 
                or tuple of length 2, correlation matrix is size
        forget_factor : scalar between 0 and 1. 
            1 is straight averaging, increasing time window
            0 will make matrix only dependent on the last sample
        """
        if not isinstance(size, (list, tuple, np.ndarray)):
            size = (size, size)
        assert len(size) == 2
        self.vec_len = size[0]
        self.corr_mat = np.zeros(size)
        #self.avg = fc.MovingAverage(forget_factor, size)
        self.corr = np.zeros(size)
        self._preallocated_update = np.zeros_like(self.corr)
        self._old_vec = np.zeros((size[1], 1))
        self.delay = delay
        self._saved_vecs = np.zeros((size[1], delay))
        self.estimate_mean = estimate_mean
        if self.estimate_mean:
            self.mean = fc.MovingAverage(forget_factor, (size[0], 1))
            self.mean2 = fc.MovingAverage(forget_factor, (size[1], 1))

        self.n = 0


    def update(self, vec1, vec2=None):
        """Updates the correlation matrix with a new sample vector
            If only one is provided, the autocorrelation is computed
            If two are provided, their cross-correlation is computed

            Both vec are ndarrays of shape (vec_dim, 1)

            For the recursive definition of covariance with sample mean, 
            take a look at 'Computing (co)variances recursively' - Thijs Knaap. 

            Without mean we calculate 1/N sum_{n=1}^{N} x_n y_n\*

            With mean we calculate 1/N sum_{n=1}^{N} (x_n - xbar_n)(y_n - ybar_n)\*
            where the sample mean is xbar_n = 1/n sum_{i=1}^{n} x_i. The recursive calculation
            is exact (apart from possible numerical differences), there is no additional assumptions. 

            Bessels correctyion (normalizing by 1 / (N-1)) is used, but is not necessarily desirable
                since it is not the lower MSE (but it is unbiased). For a normalization of 1/N, use the
                following code (the first index must be handled separately in this case as well)
            
                np.matmul(vec1 - self.mean.state, (vec2 - self.mean2.state).T, out=self._preallocated_update)
                self._preallocated_update \*= 1 / self.n
                self.corr \*= self.n / (self.n + 1)
        """
        if vec2 is None:
            vec2 = vec1

        if self.estimate_mean:
            self.mean.update(vec1)
            if vec2 is None:
                self.mean2 = self.mean
            else:
                self.mean2.update(vec2)

        if self.delay > 0:
            assert not self.estimate_mean #Check if the implementation makes sense for this combo
            idx = self.n % self.delay
            self._old_vec[...] = self._saved_vecs[:,idx:idx+1]
            self._saved_vecs[:,idx:idx+1] = vec2
            vec2 = self._old_vec

        if self.n >= self.delay:
            if self.estimate_mean:
                if self.n > 0:
                    np.matmul(vec1 - self.mean.state, (vec2 - self.mean2.state).T, out=self._preallocated_update)
                    self._preallocated_update *= (self.n+1) / (self.n**2)
                    self.corr *= (self.n - 1) / self.n
            else:
                np.matmul(vec1, vec2.T, out=self._preallocated_update)
                self._preallocated_update *= 1 / (self.n + 1)
                self.corr *= self.n / (self.n + 1)
            self.corr += self._preallocated_update
            #self.avg.update(self._preallocated_update)

        self.n += 1

    # def _update_with_sample_mean(self, vec1, vec2=None):
    #     """Updates the correlation matrix with a new sample vector
    #         If only one is provided, the autocorrelation is computed
    #         If two are provided, their cross-correlation is computed

    #         Both vec are ndarrays of shape (vec_dim, 1)

    #         For the recursive definition of covariance with sample mean, 
    #         take a look at
    #         'Computing (co)variances recursively' - Thijs Knaap
    #     """

    #     self.mean.update(vec1)
    #     if vec2 is None:
    #         vec2 = vec1
    #         self.mean2 = self.mean
    #     else:
    #         self.mean2.update(vec2)

    #     if self.delay != 0:
    #         raise NotImplementedError

    #     np.matmul(vec1 - self.mean.state, (vec2 - self.mean2.state).T, out=self._preallocated_update)
    #     self._preallocated_update *= (self.n + 1) / self.n #To get the new data normalized as (1/n instead of 1/(n+1))
    #     self.avg.update(self._preallocated_update)

    #     self.n += 1


    # def _update_zero_mean(self, vec1, vec2=None):
    #     """Update the correlation matrix with a new sample vector
    #         If only one is provided, the autocorrelation is computed
    #         If two are provided, the cross-correlation is computed
    #     """
    #     if vec2 is None:
    #         vec2 = vec1

    #     if self.delay > 0:
    #         idx = self.n % self.delay
    #         self._old_vec[...] = self._saved_vecs[:,idx:idx+1]
    #         self._saved_vecs[:,idx:idx+1] = vec2
    #         vec2 = self._old_vec

    #     if self.n >= self.delay:
    #         np.matmul(vec1, vec2.T, out=self._preallocated_update)
    #         self.avg.update(self._preallocated_update)
    #     self.n += 1

    def get_corr(self, autocorr=False, est_method="scm", pos_def=False):
        """Returns the correlation matrix and stores it in self.corr_mat
        
            Will ensure positive semi-definiteness and hermitian-ness if autocorr is True
            If pos_def=True it will even ensure that the matrix is positive definite. 
        
            est_method can be 'scm', 'oas' or 'qis'
        """
        if not autocorr:
            assert est_method == "scm"
            self.corr_mat[...] = self.corr
            #if self.estimate_mean:
            #    self.corr_mat *= self.n / (self.n-1)
            return self.corr_mat

        num_samples = self.n-1 if self.estimate_mean else self.n
        if est_method == "scm":
            self.corr_mat[...] = self.corr
            #if self.estimate_mean:
            #    self.corr_mat *= self.n / (self.n-1)
        elif est_method == "oas":
            self.corr_mat[...] = cov_est_oas(self.corr, num_samples, verbose=True)
        elif est_method == "qis":
            #print(self.n)
            self.corr_mat[...] = cov_est_qis(self.corr, num_samples)
        else:
            raise ValueError("Invalid est_method name")
        
        self.corr_mat = mat.ensure_hermitian(self.corr_mat)
        if pos_def:
            self.corr_mat = mat.ensure_pos_def_adhoc(self.corr_mat, verbose=True)
        else:
            self.corr_mat = mat.ensure_pos_semidef(self.corr_mat)
        return self.corr_mat
        



def sample_correlation(data, data2=None, estimate_mean=False):
    """
    data is a matrix of size (data_dim, num_samples)
    data2 is an optional matrix of size (data_dim2, num_samples)
    the cross-correlation is calulcated if this is supplied

    if estimate_mean is True, the sample mean is calculated xbar = 1/N sum_{n=1}^{N} x_n
    where x_n is the nth column of the data matrix
    the correlation is 1/(N-1) sum_{n=1}^{N} (x_n - xbar)(y_n - ybar)^H

    if estimate_mean is False, the data is assumed to be zero-mean
    the correlation is 1/N sum_{n=1}^{N} x_n y_n^H
    
    """
    if data2 is None:
        data2 = data

    assert data.ndim == 2
    assert data2.ndim == 2
    assert data.shape[1] == data2.shape[1]
    N = data.shape[1]
    data_dim = data.shape[0]
    data_dim2 = data2.shape[0]

    if estimate_mean:        
        if N == 1:
            scm = np.zeros((data_dim, data_dim2))
        else:
            centering_matrix = np.eye(N) - (1/N) * np.ones((N, N))
            scm = data @ centering_matrix @ data2.T
            scm *= 1 / (N-1)
    else:    
        scm = data @ data2.T
        scm *= 1 / N
    return scm









class Autocorrelation:
    """
    Autocorrelation is defined as r_m1m2(i) = E[s_m1(n) s_m2(n-i)]
    the returned shape is (num_channels, num_channels, max_lag)
    r[m1, m2, i] is positive entries, r_m1m2(i)
    r[m2, m1, i] is negative entries, r_m1m2(-i)
    """
    def __init__(self, forget_factor, max_lag, num_channels):
        """
        size : scalar integer, correlation matrix is size x size. 
        forget_factor : scalar between 0 and 1. 
        1 is straight averaging, increasing time window
        0 will make matrix only dependent on the last sample
        """
        self.forget_factor = forget_factor
        self.max_lag = max_lag
        self.num_channels = num_channels

        self.corr = fc.MovingAverage(forget_factor, (num_channels, num_channels, max_lag))
        self.corr_mat = np.zeros((self.num_channels*self.max_lag, self.num_channels*self.max_lag))

        self._buffer = np.zeros((self.num_channels, self.max_lag-1))

    def update(self, sig):
        num_samples = sig.shape[-1]
        if num_samples == 0:
            return
        padded_sig = np.concatenate((self._buffer, sig), axis=-1)
        self.corr.update(autocorr(padded_sig, self.max_lag), count_as_updates=num_samples)

        self._buffer[...] = padded_sig[:,sig.shape[-1]:]

    def get_corr_mat(self, max_lag=None, new_first=True, pos_def=False):
        """
        shorthands: L=max_lag, M=num_channels, r(i)=r_m1m2(i)

        R = E[s(n) s(n)^T] =    [R_11 ... R_1M  ]
                                [               ]
                                [               ]
                                [R_M1 ... R_MM  ]

        R_m1m2 = E[s_m1(n) s_m2(n)^T]

        == With new_first==True the vector is defined as
        s_m(n) = [s_m(n) s_m(n-1) ... s_m(n-max_lag+1)]^T

        R_m1m2 =    [r(0)   r(1) ... r(L-1) ]
                    [r(-1)                  ]
                    [                  r(1) ]
                    [r(-L+1) ... r(-1) r(0) ]

        == With new_first==False, the vector is defined as 
        s_m(n) = [s_m(n-max_lag+1) s_m(n-max_lag+2) ... s_m(n)]^T

        R_m1m2 =    [r(0) r(-1) ...   r(-L+1) ]
                    [r(1)                     ]
                    [                  r(-1)  ]
                    [r(L-1) ...   r(1) r(0)   ]
        """
        if max_lag is None:
            max_lag = self.max_lag
        current_corr_est = self.corr.state[...,:max_lag]

        if new_first:
            self.corr_mat = corr_matrix_from_autocorrelation(current_corr_est)
        else:
            self.corr_mat = mat.block_transpose(corr_matrix_from_autocorrelation(current_corr_est), max_lag)

        self.corr_mat = mat.ensure_hermitian(self.corr_mat)
        if pos_def:
            self.corr_mat = mat.ensure_pos_def_adhoc(self.corr_mat, verbose=True)
        else:
            self.corr_mat = mat.ensure_pos_semidef(self.corr_mat)
        return self.corr_mat



def autocorr(sig, max_lag, normalize=True):
    num_channels = sig.shape[0]
    padded_len = sig.shape[-1]
    num_samples = padded_len - max_lag + 1
    r = np.zeros((num_channels, num_channels, max_lag))
    for ch1 in range(num_channels):
        for ch2 in range(num_channels):
            for i in range(max_lag-1, padded_len):
                r[ch1, ch2, :] += sig[ch1,i] * np.flip(sig[ch2, i-max_lag+1:i+1])
    if normalize:
        r /= num_samples
    return r


def multichannel_acf_from_independent_acf(acf_2d):
    """
    Takes a (num_channels, corr_len) autocorrelation function
        and returns a (num_channels, num_channels, corr_len) autocorrelation function
        which includes the cross-correlations between the different channels,
        which are assumed to be 0. 
    
    """
    assert acf_2d.ndim == 2
    num_channels = acf_2d.shape[0]
    corr_len = acf_2d.shape[-1]
    acf_3d = np.zeros((num_channels, num_channels, corr_len))
    for ch in range(num_channels):
        acf_3d[ch,ch,:] = acf_2d[ch,:]
    return acf_3d

    

def corr_matrix_from_autocorrelation(corr):
    """The correlation is of shape (num_channels, num_channels, max_lag)
        corr[i,j,k] = E[x_i(n)x_j(n-k)], for k = 0,...,max_lag-1
        That means corr[i,j,k] == corr[j,i,-k]

        The output is a correlation matrix of shape (num_channels*max_lag, num_channels*max_lag)
        Each block R_ij is defined as
        [r_ij(0) ... r_ij(max_lag-1)  ]
        [                             ]
        [r_ij(-max_lag+1) ... r_ij(0) ]

        which is part of the full matrix as 
        [R_11 ... R_1M  ]
        [               ]
        [R_M1 ... R_MM  ]

        So that the full R = E[X(n) X^T(n)], 
        where X(n) = [X^T_1(n), ... , X^T_M]^T
        and X_i = [x_i(n), ..., x_i(n-max_lag)]^T
    """
    corr_mat = mat.block_of_toeplitz(np.moveaxis(corr, 0,1), corr)
    return corr_mat
    

def corr_matrix(seq1, seq2, lag1, lag2):
    """Computes the cross-correlation matrix from two signals. See definition
        in Autocorrelation class or corr_matrix_from_autocorrelation.
    
        seq1 has shape (sumCh, numCh1, numSamples) or (numCh1, numSamples)
        seq2 has shape (sumCh, numCh2, numSamples) or (numCh2, numSamples)
    
        Outputs a correlation matrix of size 
        (numCh1*lag1, numCh2*lag2)
        
        Sums over the first dimension"""
    assert seq1.ndim == seq2.ndim
    if seq1.ndim == 2:
        seq1 = np.expand_dims(seq1, 0)
        seq2 = np.expand_dims(seq2, 0)
    assert seq1.shape[0] == seq2.shape[0]
    assert seq1.shape[2] == seq2.shape[2]
    sum_ch = seq1.shape[0]
    seq_len = seq1.shape[2]

    if seq_len < max(lag1, lag2):
        pad_len = max(lag1, lag2) - seq_len
        seq1 = np.pad(seq1, ((0,0), (0,0), (0,pad_len)))
        seq2 = np.pad(seq2, ((0,0), (0,0), (0,pad_len)))

    R = np.zeros((seq1.shape[1]*lag1, seq2.shape[1]*lag2))
    for i in range(sum_ch):
        R += _corr_matrix(seq1[i,...], seq2[i,...], lag1, lag2) / sum_ch
    return R 

def _corr_matrix(seq1, seq2, lag1, lag2):
    """seq1 has shape (num_channels_1, numSamples)
        seq2 has shape (num_channels_2, numSamples)
    
        Outputs a correlation matrix of size 
        (num_channels_1*lag1, num_channels_2*lag2)"""
    assert seq1.ndim == seq2.ndim == 2
    assert seq1.shape[-1] == seq2.shape[-1]
    seqLen = seq1.shape[-1]
    num_channels1 = seq1.shape[0]
    num_channels2 = seq2.shape[0]

    corr = np.zeros((num_channels1, num_channels2, 2*seqLen-1))
    for i in range(num_channels1):
        for j in range(num_channels2):
            corr[i,j,:] = spsig.correlate(seq1[i,:], seq2[j,:], mode="full")
    corr /= seqLen
    corrMid = seqLen - 1
    R = np.zeros((seq1.shape[0]*lag1, seq2.shape[0]*lag2))
    for c1 in range(seq1.shape[0]):
        for l1 in range(lag1):
            rowIdx = c1*lag1 + l1
            R[rowIdx,:] = corr[c1, :, corrMid-l1:corrMid-l1+lag2].ravel()
    return R

def is_autocorr_mat(ac_mat, verbose=False):
    assert ac_mat.ndim == 2
    square = ac_mat.shape[0] == ac_mat.shape[1]
    herm = mat.is_hermitian(ac_mat)
    psd = mat.is_pos_semidef(ac_mat)

    if verbose:
        print(f"Is square: {square}")
        print(f"Is hermitian: {herm}")
        print(f"Is positive semidefinite: {psd}")

    return all((square, herm, psd))

def is_autocorr_func(func, verbose=False):
    """
    An autocorrelation function should be of shape
    (num_channels, num_channels, max_lag)

    Assumes the autocorrelation is real valued, from a real-valued stochastic process

    The evenness property is inherent in the representation as 
        func = [r(0), r(1), ..., r(max_lag-1)], as only one side is recorded
    """
    if func.ndim == 1:
        func = func[None,:]
    if func.ndim == 2:
        new_func = np.zeros((func.shape[0], func.shape[0], func.shape[1]), dtype=func.dtype)
        for i in range(func.shape[0]):
            new_func[i,i,:] = func[i,:]
        func = new_func
    assert func.ndim == 3
    assert func.shape[0] == func.shape[1]
    num_channels = func.shape[0]

    max_at_zero = True
    for i in range(num_channels):
        if np.any(func[i,i,0] < func[i,i,1:]):
            max_at_zero = False

    symmetric = True
    for i in range(num_channels): 
        for j in range(num_channels):
            if not np.allclose(func[i,j,:], func[j,i,:]):
                symmetric = False

    if verbose:
        print(f"Is symmetric: {symmetric}")
        print(f"Is max at zero: {max_at_zero}")

    return all((symmetric, max_at_zero))

def _func_is_symmetric(func):
    raise NotImplementedError
    assert func.ndim == 3
    assert func.shape[0] == func.shape[1]
    num_ch = func.shape[0]
    for i in range(num_ch):
        for j in range(num_ch):
            np.allclose(func[i,j,:])


def periodic_autocorr(seq):
    """seq is a single channel sequence of shape (1, period_length), 
    that should be a single period of a periodic signal

    This function calculates the periodic autocorrelation
    
    returns autocorr of shape (period_length)
    """
    assert seq.ndim == 2
    assert seq.shape[0] == 1
    num_samples = seq.shape[1]
    autocorr = np.zeros((num_samples))
    for shift in range(num_samples):
        autocorr[shift] = np.sum(seq[0,:num_samples-shift] * seq[0,shift:])
        autocorr[shift] += np.sum(seq[0,num_samples-shift:] * seq[0,:shift])
    autocorr /= num_samples
    return autocorr


def get_filter_for_autocorrelation(autocorr):
    """
    autocorr is of shape (num_channels, max_lag)
        
    computes a filter h(i) such that y(n) has the autocorrelation provided
        if y(n) = h(i) * x(n), where * is convolution, and x(n) is white

    returns an IR of shape (num_channels, max_lag*2-1)
    """
    ac_full = np.concatenate((autocorr, np.flip(autocorr[:,1:],axis=-1)), axis=-1)
    psd = np.fft.fft(ac_full, axis=-1)
    psd = np.real_if_close(psd)
    assert np.allclose(np.imag(psd), 0) 
    freq_func = np.sqrt(psd)
    ir = np.real_if_close(np.fft.ifft(freq_func, axis=-1))
    assert np.allclose(np.imag(ir), 0)  
    return ir













































def autocorrelation_old(sig, max_lag, interval):
    """
    I'm not sure I trust this one. Write some unit tests 
    against Autocorrelation class first. But the corr_matrix function
    works, so this shouldn't be needed. 

    Returns the autocorrelation of a multichannel signal

    corr[j,k,i] = r_jk(i) = E[s_j(n)s_k(n-i)]
    output is for positive indices, i=0,...,max_lag-1
    for negative correlation values for r_jk, use r_kj instead
    Because r_jk(i) = r_kj(-i)
    """
    #Check if the correlation is normalized
    num_channels = sig.shape[0]
    corr = np.zeros((num_channels, num_channels, max_lag))

    for i in range(num_channels):
        for j in range(num_channels):
            corr[i,j,:] = spsig.correlate(np.flip(sig[i,interval[0]-max_lag+1:interval[1]]), 
                                            np.flip(sig[j,interval[0]:interval[1]]), "valid")
    # corr /= interval[1] - interval[0] #+ max_lag - 1
    return corr

# These should be working correctly, but are written only for testing purposes.
# Might be removed at any time and moved to test module. 

def autocorr_ref_spsig(sig, max_lag):
    num_channels = sig.shape[0]
    num_samples = sig.shape[-1]
    sig = np.pad(sig, ((0,0), (max_lag-1, 0)))
    r = np.zeros((num_channels, num_channels, max_lag))
    for ch1 in range(num_channels):
        for ch2 in range(num_channels):
                r[ch1, ch2, :] = spsig.correlate(np.flip(sig[ch1,:]), 
                                            np.flip(sig[ch2,max_lag-1:]), "valid")
    r /= num_samples
    return r


def autocorr_ref(sig, max_lag):
    num_channels = sig.shape[0]
    num_samples = sig.shape[-1]
    padded_len = num_samples + max_lag - 1
    sig = np.pad(sig, ((0,0), (max_lag-1, 0)))
    r = np.zeros((num_channels, num_channels, max_lag))
    for ch1 in range(num_channels):
        for ch2 in range(num_channels):
            for i in range(max_lag-1, padded_len):
                r[ch1, ch2, :] += sig[ch1,i] * np.flip(sig[ch2, i-max_lag+1:i+1])
    r /= num_samples
    return r

def corr_mat_new_first_ref(sig, max_lag):
    num_channels = sig.shape[0]
    num_samples = sig.shape[-1]
    padded_len = num_samples + max_lag - 1
    sig = np.pad(sig, ((0,0), (max_lag-1, 0)))
    R = np.zeros((max_lag*num_channels, max_lag*num_channels))
    for i in range(max_lag-1, padded_len):
        vec = np.flip(sig[:,i-max_lag+1:i+1], axis=-1).reshape(-1,1)
        R+= vec @ vec.T
    R /= num_samples
    return R

def corr_mat_old_first_ref(sig, max_lag):
    num_channels = sig.shape[0]
    num_samples = sig.shape[-1]
    padded_len = num_samples + max_lag - 1
    sig = np.pad(sig, ((0,0), (max_lag-1, 0)))
    R = np.zeros((max_lag*num_channels, max_lag*num_channels))
    for i in range(max_lag-1, padded_len):
        vec = sig[:,i-max_lag+1:i+1].reshape(-1,1)
        R+= vec @ vec.T
    R /= num_samples
    return R