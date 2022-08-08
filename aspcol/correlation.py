import numpy as np
import numexpr as ne
import scipy.signal as spsig
import scipy.linalg as splin

import ancsim.signal.matrices as mat
import ancsim.signal.filterclasses as fc


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
    
    n is the sample size of the covariance matrix. If the sample mean was subtracted 
        (if the variables were not zero mean), the sample size should be adjusted to n-1.

    Code slightly adapted, but mostly taken straight from
    https://github.com/pald22/covShrinkage by Patrick Ledoit
    
    """
    #Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
    #    None, np.nan or int
    #Post-Condition: Sigmahat dataframe is returned

    #Set df dimensions
    N = Y.shape[0]                                              #num of columns
    p = Y.shape[1]                                                 #num of rows

    #default setting
    if (k is None or math.isnan(k)):
        Y = Y.sub(Y.mean(axis=0), axis=1)                               #demean
        k = 1

    #vars
    n = N-k                                      # adjust effective sample size
    c = p/n                                               # concentration ratio

    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    sample = (sample+sample.T)/2                              #make symmetrical

    #Spectral decomp
    lambda1, u = np.linalg.eigh(sample)            #use Cholesky factorisation 
    #                                               based on hermitian matrix
    lambda1 = lambda1.real.clip(min=0)              #reset negative values to 0
    dfu = pd.DataFrame(u,columns=lambda1)   #create df with column names lambda
    #                                        and values u
    dfu.sort_index(axis=1,inplace = True)              #sort df by column index
    lambda1 = dfu.columns                              #recapture sorted lambda

    #COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2,1/c**2)**0.35)/p**0.35                   #smoothing parameter
    invlambda = 1/lambda1[max(1,p-n+1)-1:p]  #inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl['lambda'] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values,min(p,n))]          #like  1/lambda_j
    Lj = pd.DataFrame(Lj.to_numpy())                        #Reset column names
    Lj_i = Lj.subtract(Lj.T)                    #like (1/lambda_j)-(1/lambda_i)

    theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj)*h**2)).mean(axis = 0)          #smoothed Stein shrinker
    Htheta = Lj.multiply(Lj*h).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj)*h**2)).mean(axis = 0)                    #its conjugate
    Atheta2 = theta**2+Htheta**2                         #its squared amplitude

    if p<=n:               #case where sample covariance matrix is not singular
        delta = 1 / ((1-c)**2*invlambda+2*c*(1-c)*invlambda*theta \
                    +c**2*invlambda*Atheta2)    #optimally shrunk eigenvalues
        delta = delta.to_numpy()
    else:
        delta0 = 1/((c-1)*np.mean(invlambda.to_numpy())) #shrinkage of null 
        #                                                 eigenvalues
        delta = np.repeat(delta0,p-n)
        delta = np.concatenate((delta, 1/(invlambda*Atheta2)), axis=None)

    deltaQIS = delta*(sum(lambda1)/sum(delta))                  #preserve trace
    
    temp1 = dfu.to_numpy()
    temp2 = np.diag(deltaQIS)
    temp3 = dfu.T.to_numpy().conjugate()
    #reconstruct covariance matrix
    sigmahat = pd.DataFrame(np.matmul(np.matmul(temp1,temp2),temp3))
    return sigmahat



class SampleCorrelation:
    """
    Estimates the correlation matrix between two random vectors 
    v(n) & w(n) by iteratively computing (1/N) sum_{n=0}^{N-1} v(n) w^T(n) 
    The goal is to estimate R = E[v(n) w^T(n)]

    If delay is supplied, it will calculate E[v(n) w(n-delay)^T]

    Only the internal state will be changed by calling update()
    In order to update self.corr_mat, get_corr() must be called
    """
    def __init__(self, forget_factor, size, delay=0):
        """
        size : scalar integer, correlation matrix is size x size. 
                or tuple of length 2, correlation matrix is size
        forget_factor : scalar between 0 and 1. 
            1 is straight averaging, increasing time window
            0 will make matrix only dependent on the last sample
        """
        if not isinstance(size, (list, tuple, np.ndarray)):
            size = (size, size)
        self.corr_mat = np.zeros(size)
        self.avg = fc.MovingAverage(forget_factor, size)
        self._preallocated_update = np.zeros_like(self.avg.state)
        self._old_vec = np.zeros((size[1], 1))
        self.delay = delay
        self._saved_vecs = np.zeros((size[1], delay))
        self.n = 0

    def update(self, vec1, vec2=None):
        """Update the correlation matrix with a new sample vector
            If only one is provided, the autocorrelation is computed
            If two are provided, the cross-correlation is computed
        """
        if vec2 is None:
            vec2 = vec1

        if self.delay > 0:
            idx = self.n % self.delay
            self._old_vec[...] = self._saved_vecs[:,idx:idx+1]
            self._saved_vecs[:,idx:idx+1] = vec2
            vec2 = self._old_vec

        if self.n >= self.delay:
            np.matmul(vec1, vec2.T, out=self._preallocated_update)
            self.avg.update(self._preallocated_update)
        self.n += 1

    def get_corr(self, autocorr=False, est_method="plain", pos_def=False):
        """Returns the correlation matrix and stores it in self.corr_mat
        
            Will ensure positive semi-definiteness and hermitian-ness if autocorr is True
            If pos_def=True it will even ensure that the matrix is positive definite. 
        
            est_method can be 'oas' or 'plain'
        """
        if not autocorr:
            self.corr_mat[...] = self.avg.state
            return self.corr_mat

        if est_method == "plain":
            self.corr_mat[...] = self.avg.state
        elif est_method == "oas":
            self.corr_mat[...] = cov_est_oas(self.avg.state, self.n, verbose=True)
        else:
            raise ValueError("Invalid est_method name")
        
        self.corr_mat = mat.ensure_hermitian(self.corr_mat)
        if pos_def:
            self.corr_mat = mat.ensure_pos_def_adhoc(self.corr_mat, verbose=True)
        else:
            self.corr_mat = mat.ensure_pos_semidef(self.corr_mat)
        return self.corr_mat
        

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
        if new_first:
            self.corr_mat = corr_matrix_from_autocorrelation(self.corr.state)
        else:
            self.corr_mat = mat.block_transpose(corr_matrix_from_autocorrelation(self.corr.state), max_lag)

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
    return all((symmetric, max_at_zero))

def _func_is_symmetric(func):
    raise NotImplementedError
    assert func.ndim == 3
    assert func.shape[0] == func.shape[1]
    num_ch = func.shape[0]
    for i in range(num_ch):
        for j in range(num_ch):
            np.allclose(func[i,j,:])



def cos_similary(vec1, vec2):
    """Computes <vec1, vec2> / (||vec1|| ||vec2||)
        which is cosine of the angle between the two vectors. 
        1 is paralell vectors, 0 is orthogonal, and -1 is opposite directions

        If the arrays have more than 1 axis, it will be flattened.
    """
    assert vec1.shape == vec2.shape
    vec1 = np.ravel(vec1)
    vec2 = np.ravel(vec2)
    norms = np.linalg.norm(vec1) *np.linalg.norm(vec2)
    if norms == 0:
        return np.nan
    ip = vec1.T @ vec2
    return ip / norms

def corr_matrix_distance(mat1, mat2):
    """Computes the correlation matrix distance, as defined in:
    Correlation matrix distaince, a meaningful measure for evaluation of 
    non-stationary MIMO channels - Herdin, Czink, Ozcelik, Bonek
    
    0 means that the matrices are equal up to a scaling
    1 means that they are maximally different (orthogonal in NxN dimensional space)
    """
    assert mat1.shape == mat2.shape
    norm1 = np.linalg.norm(mat1, ord="fro", axis=(-2,-1))
    norm2 = np.linalg.norm(mat2, ord="fro", axis=(-2,-1))
    if norm1 * norm2 == 0:
        return np.nan
    return 1 - np.trace(mat1 @ mat2) / (norm1 * norm2)


def covariance_distance_riemannian(mat1, mat2):
    """
    Computes the covariance matrix distance as proposed in 
        A Metric for Covariance Matrices - Wolfgang Förstner, Boudewijn Moonen
        http://www.ipb.uni-bonn.de/pdfs/Forstner1999Metric.pdf

    It is the distance of a canonical invariant Riemannian metric on the space 
        Sym+(n, R) of real symmetric positive definite matrices. 

    When the metric of the space is the fisher information metric, this is the 
        distance of the space. See COVARIANCE CLUSTERING ON RIEMANNIAN MANIFOLDS
        FOR ACOUSTIC MODEL COMPRESSION - Shinohara, Masukp, Akamine

    Invariant to affine transformations and inversions. 
    It is a distance measure, so 0 means equal and then it goes to infinity
        and the matrices become more unequal.

    """
    eigvals = splin.eigh(mat1, mat2, eigvals_only=True)
    return np.sqrt(np.sum(np.log(eigvals)**2))


def covariance_distance_kl_divergence(mat1, mat2):
    """
    It is the Kullback Leibler divergence between two Gaussian
        distributions that has mat1 and mat2 as their covariance matrices. 
        Assumes both of these distributions has zero mean

    It is a distance measure, so 0 means equal and then it goes to infinity
        and the matrices become more unequal.
    
    """
    assert mat1.shape == mat2.shape
    assert mat1.shape[0] == mat1.shape[1]
    assert mat.ndim == 2
    N = mat1.shape[0]
    eigvals = splin.eigh(mat1, mat2, eigvals_only=True)

    det1 = splin.det(mat1)
    det2 = splin.det(mat2)
    common_trace = np.sum(eigvals)
    return np.sqrt((np.log(det2 / det1) + common_trace - N) / 2)



def periodic_autocorr(seq):
    """seq is a single channel sequence of shape (1, period_length), 
    that should be a single period of a periodic signal

    This function calculates the periodic autocorrelation
    
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
















def autocorrelation(sig, max_lag, interval):
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