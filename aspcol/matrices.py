import numpy as np
import scipy.signal as spsig
import scipy.linalg as splin

def to_length(ar, length, axis=-1):
    """Either truncates or pads ndarray at the end with zeros, 
        so that the length along axis is equal to length."""
    end = min(length, ar.shape[axis])
    slices = [slice(None) for _ in range(ar.ndim)]
    slices[axis] = slice(0,end)
    ar = ar[tuple(slices)]

    pad_tuple = [(0,0) for _ in range(ar.ndim)]
    pad_tuple[axis] = (0, length-end)
    ar = np.pad(ar, pad_tuple)
    return ar

def block_diag(arrays):
    """Creates a block diagonal matrix
        - arrays is a list of ndarrays
        - The operation is performed independently for the first dimension. 
        The length of the first dimension must naturally be the same.
        - arr.ndim must be 3 or more
        (to make a block_diag with ndim=2, just use scipy.block_diag)"""
    assert all([ar.ndim == arrays[0].ndim for ar in arrays])
    assert all([ar.shape[:-2] == arrays[0].shape[:-2] for ar in arrays])
    assert all([ar.dtype == arrays[0].dtype for ar in arrays])
    tot_cols = np.sum([ar.shape[-1] for ar in arrays])
    tot_rows = np.sum([ar.shape[-2] for ar in arrays])
    bc_dim = arrays[0].shape[:-2]
    
    diag_mat = np.zeros(np.concatenate((bc_dim, [tot_rows, tot_cols])), dtype=arrays[0].dtype)

    idx = np.zeros(2, dtype=int)
    for ar in arrays:
        idx_shift = ar.shape[-2:]
        diag_mat[...,idx[0]:idx[0]+idx_shift[0], 
                    idx[1]:idx[1]+idx_shift[1]] = ar

        idx += idx_shift
    return diag_mat

def block_of_toeplitz(block_of_col, block_of_row=None):
    """ block_of_col and block_of_row are of shapes
        (M, N, K_col) and (M, N, K_row)
        
        The values in the last axis for each (m,n) block is turned
        into a toeplitz matrix. The output is a single matrix of shape
        (M*K_col N*K_row)

        If only block_of_col is supplied, the toeplitz 
        matrices are constructed symmetric
    """
    if block_of_row is None:
        if np.iscomplexobj(block_of_col):
            block_of_row = np.conj(block_of_col)
        else:
            block_of_row = block_of_col

    assert block_of_col.shape[:2] == block_of_row.shape[:2]
    len_col = block_of_col.shape[2]
    len_row = block_of_row.shape[2]

    block_mat = np.zeros((block_of_col.shape[0]*len_col, 
                        block_of_col.shape[1]*len_row), 
                        dtype=block_of_col.dtype)
    for m in range(block_of_col.shape[0]):
        for n in range(block_of_col.shape[1]):
            block_mat[m*len_col:(m+1)*len_col, 
                      n*len_row:(n+1)*len_row] = splin.toeplitz(block_of_col[m,n,:], 
                                                                   block_of_row[m,n,:])
    return block_mat

def block_transpose(matrix, block_size, out=None):
    """
    Transposes each block individually in a block matrix. 
    block_size is an integer defining the size of the square blocks. 
    Requires square blocks, but not a square block matrix. 

    B = [
        A_11^T, ..., A_1r^T

        A_r1^T, ..., A_rc^T
    ]

    if out is supplied, the transposed matrix will be written into its memory
    """
    assert matrix.shape[0] % block_size == 0
    assert matrix.shape[1] % block_size == 0
    num_rows = matrix.shape[0] // block_size
    num_cols = matrix.shape[1] // block_size

    if out is not None:
        transposed_mat = out
    else:
        transposed_mat = np.zeros((matrix.shape[1], matrix.shape[0]))

    for r in range(num_rows):
        for c in range(num_cols):
            transposed_mat[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size] = \
                    matrix[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size].T
    return transposed_mat



def block_diag_multiply(mat, block_left=None, block_right=None, out_matrix=None):
    """
    block_left is size (a, a)
    block_right is size (a, a)
    mat is size (ab, ab) for some integer b
    
    performs the operation block_diag_left @ mat @ block_diag_right, 
    where block_diag is a matrix with multiple copies of block on the diagonal
    block_diag = I_b kron block
    """
    if block_left is not None and block_right is not None:
        assert all([block_left.shape[0] == b for b in (*block_left.shape, *block_right.shape)])
        block_size = block_left.shape[0]
    elif block_left is not None:
        assert block_left.shape[0] == block_left.shape[1]
        block_size = block_left.shape[0]
    elif block_right is not None:
        assert block_right.shape[0] == block_right.shape[1]
        block_size = block_right.shape[0]
    else:
        raise ValueError("Neither block_left or block_right provided")
    
    assert mat.shape[0] == mat.shape[1]
    mat_size = mat.shape[0]
    assert mat_size % block_size == 0
    num_blocks = mat_size // block_size
    
    if out_matrix is None:
        if any(np.issubdtype(m.dtype, np.complexfloating) for m in (mat, block_left, block_right)):
            dtp = complex
        else:
            dtp = float
        out_matrix = np.zeros((mat_size, mat_size), dtype=dtp)

    if block_left is not None and block_right is not None:
        for i in range(num_blocks):
            for j in range(num_blocks):
                out_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                    block_left @ mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] @ block_right
    elif block_left is not None:
        for i in range(num_blocks):
            for j in range(num_blocks):
                out_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                    block_left @ mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
    elif block_right is not None:
        for i in range(num_blocks):
            for j in range(num_blocks):
                out_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                    mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] @ block_right

    return out_matrix



def broadcast_func(mat, func, *args, out_shape=None, dtype=float, **kwargs):
    """
    Applies the same function to each matrix in the array
    mat is of shape (*tuple, a, b)
        it can also be a tuple of multiple matrices of the same shape
    func is applied to the matrices in the last two axes

    out_shape is the shape of the return value for a single matrix.
    It must be provided if the output is not a scalar

    output is shape (*tuple, out_shape)
    """
    if isinstance(mat, (tuple, list)):
        assert all([mat[0].shape == m.shape for m in mat])
        mat_lst = mat
        mat = mat[0]
    else:
        mat_lst = (mat,)

    assert mat.ndim >= 2
    if mat.ndim == 2:
        return func(*mat_lst, *args, **kwargs)

    broadcast_shape = mat.shape[:-2]
    if out_shape is None:
        out_shape = broadcast_shape
    else:
        out_shape = (*broadcast_shape, *out_shape)

    out = np.empty(out_shape, dtype=dtype)
    for i in np.ndindex(broadcast_shape):
        input_arg = (m[(*i,slice(None), slice(None))] for m in mat_lst)
        out[(*i, Ellipsis)] = func(*input_arg, *args, **kwargs)
    return out


def apply_blockwise(mat, func, out_shape, *args, num_blocks=None, block_size=None, separate_axis=False, dtype=float, **kwargs):
    """
    Applies the same function to each block in the block matrix mat. 
    Assumes that the matrix mat is square, and each block is square. 

    Either num_blocks or block_size must be supplied. 

    the func should return a matrix of shape out_shape

    if separate_axis is False, out_shape must be length-2 tuple
        , or the keyword 'same', in which case the returned blocks are same size as the input
        the output will be of shape (num_blocks*out_shape[0], num_blocks*out_shape[1])
    if separate_axis is True, out_shape can be a scalar value or a tuple of any length
        the output will be of shape (*out_shape, num_blocks, num_blocks)

    """
    assert mat.ndim == 2
    assert mat.shape[0] == mat.shape[1]
    mat_size = mat.shape[0]
    if num_blocks is not None:
        assert block_size is None
        block_size = mat_size // num_blocks
    elif block_size is not None:
        assert num_blocks is None
        num_blocks = mat_size // block_size
    assert block_size * num_blocks == mat_size

    if out_shape == "same":
        out_shape = (block_size, block_size)
    elif not isinstance(out_shape, (tuple, list, np.ndarray)):
        out_shape = (out_shape,)

    if separate_axis:
        raise NotImplementedError
    else:
        out = np.zeros((num_blocks * out_shape[0], num_blocks * out_shape[1]), dtype=dtype)
        for i in range(num_blocks):
            for j in range(num_blocks):
                out[i*out_shape[0]:(i+1)*out_shape[0], j*out_shape[1]:(j+1)*out_shape[1]] = \
                    func(mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size], *args, **kwargs)
    return out

# def broadcast_func(mat, func, *args, out_shape=None, dtype=float, **kwargs):
#     """
#     Applies the same function to each matrix in the array
#     mat is of shape (*tuple, a, b).
#     func is applied to the matrices in the last two axes

#     out_shape is the shape of the return value for a single matrix.
#     It must be provided if the output is not a scalar

#     output is shape (*tuple, out_shape)
#     """
#     assert mat.ndim >= 2
#     if mat.ndim == 2:
#         return func(mat, *args, **kwargs)

#     broadcast_shape = mat.shape[:-2]
#     if out_shape is None:
#         out_shape = broadcast_shape
#     else:
#         out_shape = (*broadcast_shape, *out_shape)

#     out = np.empty(out_shape, dtype=dtype)
#     for i in np.ndindex(broadcast_shape):
#         out[(*i, Ellipsis)] = func(mat[(*i,slice(None), slice(None))], *args, *kwargs)
#     return out


def is_hermitian(mat):
    return broadcast_func(mat, _is_hermitian, dtype=bool)

def _is_hermitian(mat):
    assert mat.ndim == 2
    if mat.shape[0] != mat.shape[1]:
        return False
    return np.allclose(mat, mat.conj().T)

def is_hermitian_hardcoded(mat):
    """If ndim > 2, then the array is interpreted as an array of
    multiple matrices

    if mat has shape (a,a) then a single boolean is returned
    
    if mat has shape (*tpl, a, a), then a boolean array of shape
    tpl is returned, with the truth value for each indivudal matrix 
    """
    assert mat.ndim >= 2
    if mat.shape[-1] != mat.shape[-2]:
        return False
    if mat.ndim == 2:
        return np.allclose(mat, mat.conj().T)

    out = np.empty(mat.shape[:-2], dtype=bool)
    for i in np.ndindex(out.shape):
        out[i] = np.allclose(mat[(*i,slice(None), slice(None))], 
                            mat[(*i,slice(None), slice(None))].conj().T)
    return out


# def _is_hermitian(mat):
#     assert mat.ndim == 2
#     if mat.shape[0] != mat.shape[1]:
#         return False
#     return np.allclose(mat, mat.conj().T)

def is_pos_semidef(mat):
    return broadcast_func(mat, _is_pos_semidef, dtype=bool)
    
def _is_pos_semidef(mat):
    """
    Assumes mat is hermitian without checking
    """
    assert mat.ndim == 2
    evs = splin.eigh(mat, eigvals_only=True)
    return np.all(evs >= 0)

def is_pos_def(mat):
    return broadcast_func(mat, _is_pos_def, dtype=bool)

def _is_pos_def(mat):
    """
    Assumes mat is hermitian without checking.
    Implementation should be changed to attempt 
    to perform cholesky decomp. instead
    """
    assert mat.ndim == 2
    evs = splin.eigh(mat, eigvals_only=True)
    return np.all(evs > 0)


def ensure_hermitian(mat, overwrite_ok=True):
    assert mat.ndim == 2
    if not is_hermitian(mat):
        if overwrite_ok:
            mat += mat.conj().T
            mat /= 2
        else:
            mat = (mat + mat.conj().T) / 2
    return mat

def ensure_pos_semidef(mat):
    """
    Ensures matrix is positive semidefinite
        now uses ensure_pos_def_adhoc. Change implementation
        if there is a better way to ensure positive semidefiniteness

    Assumes without checking that mat is hermitian
    If you are unsure, use ensure_hermitian first. 
    """
    assert mat.ndim == 2
    if _is_pos_semidef(mat):
        return mat

    # The following seems sensible but appears to be awful numerically
    # evs, vec = splin.eigh(mat)
    # evs[evs < 0] = 0
    # new_mat = vec @ np.diag(evs) @ vec.T.conj()
    return ensure_pos_def_adhoc(mat)

def ensure_pos_def_adhoc(mat, start_reg=-12, verbose=False):
    return broadcast_func(mat, _ensure_pos_def_adhoc, start_reg=start_reg, verbose=verbose, out_shape=mat.shape[-2:], dtype=mat.dtype)

def _ensure_pos_def_adhoc(mat, start_reg=-12, verbose=False):
    """
    Adds a scaled identity matrix to the matrix in
    order to ensure positive definiteness. Starts by 
    adding I * 10**(start_reg), and increases the 
    scaling by an order of magnitude each time it fails
    to be positive definite.
    
    Checks for positive definiteness by attempting to compute 
    a cholesky decomposition.
    """
    assert mat.ndim == 2
    reg = start_reg
    if _is_pos_def(mat):
        if verbose:
            print(f"Matrix was already positive definite")
        return mat

    while True:
        new_mat = mat + 10**(reg) * np.eye(mat.shape[0])
        if _is_pos_def(new_mat):
            break
        else:
            reg += 1
    if verbose:
        print(f"To ensure positive definiteness, identity matrix scaled by 10**{reg} was added")
    return new_mat