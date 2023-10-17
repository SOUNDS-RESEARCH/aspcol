"""Implements algorithms related to low-rank tensor approximations of impulse responses

* Decomposes and reconstructs any impulse response with singular value decomposition or canonical polyadic decomposition for a low-rank approximation [1,2]
* Implements a low-cost convolution by directly using the low-rank representation [3,4]

References
----------
`[1] <doi.org/10.23919/EUSIPCO54536.2021.9616075>`_ M. Jälmby, F. Elvander, and T. van Waterschoot, “Low-rank tensor modeling of room impulse responses,” 
in 2021 29th European Signal Processing Conference (EUSIPCO), Aug. 2021, pp. 111–115. doi: 10.23919/EUSIPCO54536.2021.9616075.
`[2] <doi.org/10.1109/TASLP.2018.2842146>`_ C. Paleologu, J. Benesty, and S. Ciochină, “Linear system identification based on a Kronecker product decomposition,” 
IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 10, pp. 1793–1808, Oct. 2018, doi: 10.1109/TASLP.2018.2842146.
`[3] <doi.org/10.1109/ICASSP.2013.6637632>`_ J. Atkins, A. Strauss, and C. Zhang, “Approximate convolution using partitioned truncated singular value decomposition filtering,” 
in 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, May 2013, pp. 176–180. doi: 10.1109/ICASSP.2013.6637632.
`[4] <doi.org/10.1109/ICASSP49357.2023.10095908>`_ M. Jälmby, F. Elvander, and T. van Waterschoot, “Fast low-latency convolution by low-rank tensor approximation,” 
in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes, Greece, Jun. 2023.
"""
import numpy as np
import numba as nb
import itertools as it
import tensorly.decomposition as td

def reconstruct_ir(ir_decomp, out=None):
    """
    Takes a decomposition of an IR and reconstructs the IR. The decomposition
    is assumed to be of the form (ir1, ir2, ir3, ...), where the i:th entry has
    shape (..., rank, I_i). 

    Parameters
    ----------
    ir_decomp : list or tuple of ndarrays, where the i:th array has shape (..., rank, I_i)
        where the resulting reconstructed IR has length prod_{i=0}^{len(ir_decomp)} I_i
    out : ndarray of same shape as return value
        will place the return value in the supplied array. Use to avoid repeated memory allocations

    Returns
    -------
    ir : ndarray of shape (..., I) 
    
    """
    broadcast_dim = ir_decomp[0].shape[:-2]
    rank = ir_decomp[0].shape[-2]
    num_dims = len(ir_decomp)
    assert np.all([np.allclose(broadcast_dim, ird.shape[:-2]) for ird in ir_decomp])
    assert np.all([rank == ird.shape[-2] for ird in ir_decomp])
    ir_len = np.prod([ird.shape[-1] for ird in ir_decomp])
    if out is None:
        ir = np.zeros((*broadcast_dim, ir_len))
    else:
        ir = out
        ir.fill(0)
        assert ir.shape == (*broadcast_dim, ir_len)

    if num_dims == 2:
        for idxs in it.product(*[range(d) for d in broadcast_dim]):
            for r in range(rank):
                ir[idxs + (slice(None),)] += np.kron(ir_decomp[1][idxs + (r, slice(None))], ir_decomp[0][idxs + (r, slice(None))])
    else:
        for idxs in it.product(*[range(d) for d in broadcast_dim]):
            for r in range(rank):
                res = np.ones(1)
                for d in range(num_dims):
                    res = np.kron(ir_decomp[d][idxs + (r, slice(None))], res)
                ir[idxs + (slice(None),)] += res       
    return ir

def decompose_ir(ir, dims, rank):
    """
    Decomposes an IR into a Kronecker / Tensor product decomposition of rank rank. 
    The IR is assumed to be of shape (..., ir_len), where the product of dims must equal ir_len. 

    Parameters
    ----------
    ir : ndarray of shape (..., ir_len)
    dims : tuple of ints
        the product of these values must equal ir_len
    rank : int

    Return
    ------
    ir_decomp : tuple of ndarrays which are shorter IRs, the i:th entry has shape (..., rank, dims[i])
    len(decomposed_ir) == len(dims)
    
    """
    num_dims = len(dims)
    ir_len = ir.shape[-1]
    assert np.prod(dims) == ir_len


    if num_dims == 1:
        assert rank == 1
        assert dims[0] == ir.shape[-1]
        return (ir[...,None,:],)
    if num_dims == 2:
        #ir = np.squeeze(ir)
        broadcast_dim = ir.shape[:-1]
        ir = np.moveaxis(ir.reshape(*broadcast_dim, dims[1], dims[0]), -2, -1)

        u, s, vh = np.linalg.svd(ir)
        ir_1 = np.sqrt(s[...,:rank,None]) * np.moveaxis(u,-2,-1)[...,:rank,:]
        ir_2 = np.sqrt(s[...,:rank,None]) * vh[...,:rank,:]
        return (ir_1, ir_2)
    else:
        ir = np.squeeze(ir)
        ir = ir.reshape(dims[2], dims[1], dims[0]).T
        ir_decomp = td.parafac(ir, rank)
        return ir_decomp.factors
        # def decomp_3d(ir, dims, rank):
        #     ir = np.squeeze(ir)
        #     ir = ir.reshape(dims[2], dims[1], dims[0]).T
        #     ir_decomp = td.parafac(ir, rank)
        #     return ir_decomp.factors



def create_filter(
    ir=None, 
    num_in=None, 
    num_out=None,
    rank = None, 
    ir_len=None
    ):
    """
    Returns the appropriate low rank filter. 
    
    The use of the filters are identical to the filter classes of aspcore.filterclasses. 
    Under an assumption that the keywords were default, meaning sum_over_input=True, 
    broadcast_dim=None, dynamic=False

    Either ir must be provided or num_in, num_out, rank, and ir_len. In the
    latter case, the filter coefficients are initialized to zero.  

    Parameters
    ----------
    ir : tuple or list of length 2 or 3, of ndarrays of shape (num_in, num_out, rank, ir_len[i])
        So the first three axes must be the same length, but they can differ in ir_len
    num_in : int
    num_out : int
    rank : int
    ir_len : 2-tuple or 3-tuple of ints

    Returns
    -------
    filter : Appropriate filter class, LowRankFilter2D or LowRankFilter3D
    """
    if all([val is not None for val in (num_in, num_out, rank, ir_len)]):
        assert ir is None
        ir = [np.zeros((num_in, num_out, rank, ir_len_i)) for ir_len_i in ir_len]

    if ir is not None:
        if len(ir) == 2:
            return LowRankFilter2D(*ir)
        elif len(ir) == 3:
            return LowRankFilter3D(*ir)
        else:
            raise ValueError("Incorrect ir supplied")


spec_lr2d = [
    ('ir1', nb.float64[:,:,:,:]),          
    ('ir2', nb.float64[:,:,:,:]),
    ('num_in', nb.int32),
    ('num_out', nb.int32),
    ('rank', nb.int32),
    ('ir_len1', nb.int32),
    ('ir_len2', nb.int32),
    ('tot_ir_len', nb.int32),
    ('buffer', nb.float64[:,:]),
    ('dly_len', nb.int32),
    ('dly_counter', nb.int32[:,:,:]),
    ('delay_line', nb.float64[:,:,:,:]),
]
@nb.experimental.jitclass(spec_lr2d)
class LowRankFilter2D:
    def __init__ (self, ir1, ir2):
        """
        Computes the linear convolution directly using a low-rank representation
        of the impulse response. Is equivalent to using a conventional filter
        using the impulse response from reconstruct_ir((ir1, ir2, ir3)), but this
        one is faster if the rank is low enough. 

        Parameters
        ----------
        ir1 : ndarray of shape (num_in, num_out, rank, ir_len1)
        ir2 : ndarray of shape (num_in, num_out, rank, ir_len2)
            Corresponds to output from decompose_ir
        
        """
        #assert all([individual_ir.ndim == 4 for individual_ir in (ir1, ir2)])
        #assert ir1.shape[:3] == ir2.shape[:3]

        self.ir1 = ir1
        self.ir2 = ir2
        self.num_in = ir1.shape[0]
        self.num_out = ir1.shape[1]
        self.rank = ir1.shape[2]
        #self.ir_len = [individual_ir.shape[3] for individual_ir in (ir1, ir2)]
        self.ir_len1 = ir1.shape[3]
        self.ir_len2 = ir2.shape[3]
        
        self.tot_ir_len = self.ir_len1 * self.ir_len2 #np.prod(self.ir_len)
        self.buffer = np.zeros((self.num_in, self.tot_ir_len - 1))

        self.dly_len = self.tot_ir_len
        #self.dly_counter = 0
        self.dly_counter = np.zeros((self.num_in, self.num_out, self.rank), dtype=nb.int32)
        self.delay_line = np.zeros((self.num_in, self.num_out, self.rank, self.tot_ir_len+self.dly_len))
        #self.filters = [[fc.create_filter(self.ir[0][ch_in:ch_in+1, :,r,:]) for ch_in in range(self.num_in)] for r in range(self.rank)]

    def process(self, sig):
        """
        Parameters
        ----------
        sig : ndarray of shape (num_in, num_samples)

        Returns
        -------
        out_sig : ndarray of shape (num_out, num_samples)
        
        """
        num_samples = sig.shape[1]

        buffered_sig = np.concatenate((self.buffer, sig), axis=-1)
        out_sig = np.zeros((self.num_out, num_samples))

        temp_vec = np.zeros(self.ir_len2)
        for ch_in in range(self.num_in):
            for ch_out in range(self.num_out):
                for r in range(self.rank):
                    for i in range(num_samples):
                        start_idx = i + self.tot_ir_len - 1
                        dly_idx = self.tot_ir_len+self.dly_counter[ch_in, ch_out, r]-1

                        sig2 = buffered_sig[ch_in,start_idx-self.ir_len1+1:start_idx+1]
                        result = np.sum(np.flip(sig2)*self.ir1[ch_in,ch_out,r,:])
                        self.delay_line[ch_in, ch_out, r, dly_idx] = result

                        for j in range(self.ir_len2):
                            temp_vec[j] = self.delay_line[ch_in,ch_out,r, dly_idx-j*self.ir_len1]

                        new_val = np.sum(temp_vec * self.ir2[ch_in,ch_out,r,:])
                        out_sig[ch_out,i] += new_val

                        self.dly_counter[ch_in, ch_out, r] += 1
                        if self.dly_counter[ch_in, ch_out, r] % self.dly_len == 0:
                            self.dly_counter[ch_in, ch_out, r] = 0
                            self.delay_line[ch_in,ch_out,r,:self.tot_ir_len] = self.delay_line[ch_in,ch_out,r,self.dly_len:]


        self.buffer[...] = buffered_sig[:, buffered_sig.shape[-1] - self.tot_ir_len + 1 :]
        return out_sig
    
    def process_nosum(self, sig):
        """
        Parameters
        ----------
        sig : ndarray of shape (num_in, num_samples)

        Returns
        -------
        out_sig : ndarray of shape (num_out, num_samples)
        
        """
        num_samples = sig.shape[1]

        buffered_sig = np.concatenate((self.buffer, sig), axis=-1)
        out_sig = np.zeros((self.num_in, self.num_out, num_samples))

        temp_vec = np.zeros(self.ir_len2)
        for ch_in in range(self.num_in):
            for ch_out in range(self.num_out):
                for r in range(self.rank):
                    for i in range(num_samples):
                        start_idx = i + self.tot_ir_len - 1
                        dly_idx = self.tot_ir_len+self.dly_counter[ch_in, ch_out, r]-1

                        sig2 = buffered_sig[ch_in,start_idx-self.ir_len1+1:start_idx+1]
                        result = np.sum(np.flip(sig2)*self.ir1[ch_in,ch_out,r,:])
                        self.delay_line[ch_in, ch_out, r, dly_idx] = result

                        for j in range(self.ir_len2):
                            temp_vec[j] = self.delay_line[ch_in,ch_out,r, dly_idx-j*self.ir_len1]

                        new_val = np.sum(temp_vec * self.ir2[ch_in,ch_out,r,:])
                        out_sig[ch_in,ch_out,i] += new_val

                        self.dly_counter[ch_in, ch_out, r] += 1
                        if self.dly_counter[ch_in, ch_out, r] % self.dly_len == 0:
                            self.dly_counter[ch_in, ch_out, r] = 0
                            self.delay_line[ch_in,ch_out,r,:self.tot_ir_len] = self.delay_line[ch_in,ch_out,r,self.dly_len:]


        self.buffer[...] = buffered_sig[:, buffered_sig.shape[-1] - self.tot_ir_len + 1 :]
        return out_sig



spec_lr3d = [
    ('ir1', nb.float64[:,:,:,:]),          
    ('ir2', nb.float64[:,:,:,:]),
    ('ir3', nb.float64[:,:,:,:]),
    ('num_in', nb.int32),
    ('num_out', nb.int32),
    ('rank', nb.int32),
    ('ir_len1', nb.int32),
    ('ir_len2', nb.int32),
    ('ir_len3', nb.int32),
    ('tot_ir_len', nb.int32),
    ('buffer', nb.float64[:,:]),
    ('dly_len', nb.int32),
    ('dly_counter', nb.int32[:,:,:]),
    ('delay_line1', nb.float64[:,:,:,:]),
    ('delay_line2', nb.float64[:,:,:,:]),
]
@nb.experimental.jitclass(spec_lr3d)
class LowRankFilter3D:
    def __init__ (self, ir1, ir2, ir3):
        """
        Computes the linear convolution directly using a low-rank representation
        of the impulse response. Is equivalent to using a conventional filter
        using the impulse response from reconstruct_ir((ir1, ir2, ir3)), but this
        one is faster if the rank is low enough. 

        Parameters
        ----------
        ir1 : ndarray of shape (num_in, num_out, rank, ir_len1)
        ir2 : ndarray of shape (num_in, num_out, rank, ir_len2)
        ir3 : ndarray of shape (num_in, num_out, rank, ir_len2)
            Corresponds to the decomposed ir from decompose_ir
        
        """
        self.ir1 = ir1
        self.ir2 = ir2
        self.ir3 = ir3
        self.num_in = ir1.shape[0]
        self.num_out = ir1.shape[1]
        self.rank = ir1.shape[2]
        #self.ir_len = [individual_ir.shape[3] for individual_ir in (ir1, ir2)]
        self.ir_len1 = ir1.shape[3]
        self.ir_len2 = ir2.shape[3]
        self.ir_len3 = ir3.shape[3]
        
        self.tot_ir_len = self.ir_len1 * self.ir_len2 * self.ir_len3 #np.prod(self.ir_len)
        self.buffer = np.zeros((self.num_in, self.tot_ir_len - 1))

        self.dly_len = self.tot_ir_len
        self.dly_counter = np.zeros((self.num_in, self.num_out, self.rank), dtype=nb.int32)
        self.delay_line1 = np.zeros((self.num_in, self.num_out, self.rank, self.tot_ir_len+self.dly_len))
        self.delay_line2 = np.zeros((self.num_in, self.num_out, self.rank, self.tot_ir_len+self.dly_len))
        #self.filters = [[fc.create_filter(self.ir[0][ch_in:ch_in+1, :,r,:]) for ch_in in range(self.num_in)] for r in range(self.rank)]


    def process(self, sig):
        """
        Parameters
        ----------
        sig : ndarray of shape (num_in, num_samples)

        Returns
        -------
        out_sig : ndarray of shape (num_out, num_samples)
        
        """
        num_samples = sig.shape[1]

        buffered_sig = np.concatenate((self.buffer, sig), axis=-1)
        out_sig = np.zeros((self.num_out, num_samples))

        temp_vec1 = np.zeros(self.ir_len2)
        temp_vec2 = np.zeros(self.ir_len3)
        for ch_in in range(self.num_in):
            for ch_out in range(self.num_out):
                for r in range(self.rank):
                    for i in range(num_samples):
                        start_idx = i + self.tot_ir_len - 1
                        dly_idx = self.tot_ir_len+self.dly_counter[ch_in, ch_out, r]-1

                        sig2 = buffered_sig[ch_in,start_idx-self.ir_len1+1:start_idx+1]
                        result = np.sum(np.flip(sig2)*self.ir1[ch_in,ch_out,r,:])
                        self.delay_line1[ch_in, ch_out, r,dly_idx] = result

                        for j in range(self.ir_len2):
                            temp_vec1[j] = self.delay_line1[ch_in,ch_out,r, dly_idx - j*self.ir_len1]

                        new_val = np.sum(temp_vec1 * self.ir2[ch_in,ch_out,r,:])
                        self.delay_line2[ch_in, ch_out, r,dly_idx] = new_val
                        
                        skip = self.ir_len1*self.ir_len2
                        for j in range(self.ir_len3):
                            temp_vec2[j] = self.delay_line2[ch_in,ch_out,r, dly_idx-j*skip]

                        new_val = np.sum(temp_vec2 * self.ir3[ch_in,ch_out,r,:])
                        out_sig[ch_out,i] += new_val

                        self.dly_counter[ch_in, ch_out, r] += 1
                        if self.dly_counter[ch_in, ch_out, r] % self.dly_len == 0:
                            self.dly_counter[ch_in, ch_out, r] = 0
                            self.delay_line1[ch_in,ch_out,r,:self.tot_ir_len] = self.delay_line1[ch_in,ch_out,r,self.dly_len:]
                            self.delay_line2[ch_in,ch_out,r,:self.tot_ir_len] = self.delay_line2[ch_in,ch_out,r,self.dly_len:]

        self.buffer[...] = buffered_sig[:, buffered_sig.shape[-1] - self.tot_ir_len + 1 :]
        return out_sig
    
    def process_nosum(self, sig):
        """
        Parameters
        ----------
        sig : ndarray of shape (num_in, num_samples)

        Returns
        -------
        out_sig : ndarray of shape (num_out, num_samples)
        
        """
        num_samples = sig.shape[1]

        buffered_sig = np.concatenate((self.buffer, sig), axis=-1)
        out_sig = np.zeros((self.num_in, self.num_out, num_samples))

        temp_vec1 = np.zeros(self.ir_len2)
        temp_vec2 = np.zeros(self.ir_len3)
        for ch_in in range(self.num_in):
            for ch_out in range(self.num_out):
                for r in range(self.rank):
                    for i in range(num_samples):
                        start_idx = i + self.tot_ir_len - 1
                        dly_idx = self.tot_ir_len+self.dly_counter[ch_in, ch_out, r]-1

                        sig2 = buffered_sig[ch_in,start_idx-self.ir_len1+1:start_idx+1]
                        result = np.sum(np.flip(sig2)*self.ir1[ch_in,ch_out,r,:])
                        self.delay_line1[ch_in, ch_out, r,dly_idx] = result

                        for j in range(self.ir_len2):
                            temp_vec1[j] = self.delay_line1[ch_in,ch_out,r, dly_idx - j*self.ir_len1]

                        new_val = np.sum(temp_vec1 * self.ir2[ch_in,ch_out,r,:])
                        self.delay_line2[ch_in, ch_out, r,dly_idx] = new_val
                        
                        skip = self.ir_len1*self.ir_len2
                        for j in range(self.ir_len3):
                            temp_vec2[j] = self.delay_line2[ch_in,ch_out,r, dly_idx-j*skip]

                        new_val = np.sum(temp_vec2 * self.ir3[ch_in,ch_out,r,:])
                        out_sig[ch_in, ch_out,i] += new_val

                        self.dly_counter[ch_in, ch_out, r] += 1
                        if self.dly_counter[ch_in, ch_out, r] % self.dly_len == 0:
                            self.dly_counter[ch_in, ch_out, r] = 0
                            self.delay_line1[ch_in,ch_out,r,:self.tot_ir_len] = self.delay_line1[ch_in,ch_out,r,self.dly_len:]
                            self.delay_line2[ch_in,ch_out,r,:self.tot_ir_len] = self.delay_line2[ch_in,ch_out,r,self.dly_len:]

        self.buffer[...] = buffered_sig[:, buffered_sig.shape[-1] - self.tot_ir_len + 1 :]
        return out_sig