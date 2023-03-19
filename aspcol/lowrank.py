import numpy as np
import itertools as it
import tensorly.decomposition as td

def reconstruct_ir(ir_decomp, out=None):
    """
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
    Parameters
    ----------
    ir : ndarray of shape (..., ir_len)
    dims : tuple of ints
        the product of these values must equal ir_len
    rank : int

    Return
    ------
    tuple of ndarrays which are shorter IRs, the i:th entry has shape (..., rank, dims[i])
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



class LowRankFilter:
    def __init__ (self, ir):
        """
        Parameters
        ----------
        ir : tuple of ndarray of shape (num_in, num_out, rank, ir_len)
            ir_len can be different for different tuple entries
            Corresponds to output from decompose_ir
            For now, can only be length 2, higher dimensions has not
            been implemented yet. 
        
        
        """
        assert all([individual_ir.ndim == 4 for individual_ir in ir])
        if len(ir) > 2:
            raise NotImplementedError
        self.ir = ir
        self.num_in = ir[0].shape[0]
        self.num_out = ir[0].shape[1]
        self.rank = ir[0].shape[2]
        self.ir_len = [individual_ir.shape[3] for individual_ir in ir]
        self.tot_ir_len = np.prod(self.ir_len)
        self.buffer = np.zeros((self.num_in, self.tot_ir_len - 1))

    def process(self, sig):
        """
        Parameters
        ----------
        sig : ndarray of shape (num_in, num_samples)

        Returns
        -------
        out_sig : ndarray of shape (num_out, num_samples)
        
        """
        assert sig.ndim == 2
        assert sig.shape[0] == self.num_in
        num_samples = sig.shape[1]

        buffered_sig = np.concatenate((self.buffer, sig), axis=-1)
        out_sig = np.zeros((self.num_out, num_samples))

        for ch_in in range(self.num_in):
            for ch_out in range(self.num_out):
                for r in range(self.rank):
                    for i in range(num_samples):
                        start_idx = i + self.tot_ir_len - 1
                        intermed_sig = np.zeros(self.ir_len[1])
                        for j in range(self.ir_len[1]):
                            #chunk = buffered_sig[ch_in, start_idx-j*self.ir_len[1]:start_idx-j*self.ir_len[1]-self.ir_len[0]:-1],
                            chunk = np.flip(buffered_sig[ch_in, start_idx-j*self.ir_len[0]-self.ir_len[0]+1 : start_idx-j*self.ir_len[0]+1], axis=-1)
                            intermed_sig[j] = np.sum(self.ir[0][ch_in, ch_out, r,:] * chunk, axis=-1)
                        out_sig[ch_out,i] += np.sum(self.ir[1][ch_in, ch_out, r,:] * intermed_sig, axis=-1)

        #sig = np.moveaxis(sig.reshape(self.num_in, self.ir_len[1], self.ir_len[0]), 1, 2)
        #out_sig = np.sum(self.ir[0][:,:,:,None,:] @ sig[:,None,None,:,:], axis=0) #shape num_out, rank, 1, ir_len[1]
        #out_sig = out_sig @ self.ir[]

        self.buffer[...] = buffered_sig[:, buffered_sig.shape[-1] - self.tot_ir_len + 1 :]
        return out_sig