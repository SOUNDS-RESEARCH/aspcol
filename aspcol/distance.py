"""A collection of distance measures for different types of quantities

Any array: 
* Mean square error

Vectors:   
* Angular distance
* Cosine similarity 

PSD matrices:  
* Correlation matrix distance [1]
* Affine invariant Riemannian metric [2]
* Kullback Leibler divergence between zero-mean Gaussian densities described by the compared matrices [3]

References
----------
`[1] <https://doi.org/10.1109/VETECS.2005.1543265>`_ M. Herdin, N. Czink, H. Ozcelik, and E. Bonek, 'Correlation matrix distance, a meaningful measure for evaluation of non-stationary MIMO channels,' in 2005 IEEE 61st Vehicular Technology Conference, May 2005, pp. 136-140 Vol. 1. doi: 10.1109/VETECS.2005.1543265.
`[2] <(doi.org/10.1007/978-3-662-05296-9_31>`_ W. Förstner and B. Moonen, 'A metric for covariance matrices,' in Geodesy-The Challenge of the 3rd Millennium, E. W. Grafarend, F. W. Krumm, and V. S. Schwarze, Eds., Berlin, Heidelberg: Springer Berlin Heidelberg, 2003, pp. 299–309. doi: 10.1007/978-3-662-05296-9_31.  
`[3] <https://web.stanford.edu/~jduchi/projects/general_notes.pdf>`_ J. Duchi, 'Derivations for Linear Algebra and Optimization'
"""
import numpy as np
import scipy.linalg as splin


def mse(var1, var2):
    """The normalized mean square error

    Normalized by the second variable

    Parameters
    ----------
    var1 : np.ndarray of any shape
        First variable
    var2 : np.ndarray of the same shape as var1
        Second variable. Cannot be zero, as it is used as the denominator
    
    Returns
    -------
    mse : float
        The normalized mean square error    
    """
    return np.sum(np.abs(var1 - var2)**2) / np.sum(np.abs(var2)**2)



#============== FOR VECTORS ======================
def angular_distance(vec1, vec2, sign_invariant=False):
    """A distance metric based on the cosine similary, that retains the
        scale invariant property, but is also a proper distance metric

    Parameters
    ----------
    vec1 : np.ndarray of shape (N,)
        First vector
    vec2 : np.ndarray of shape (N,)
        Second vector
    sign_invariant : bool, optional
        if True, the angle is first adjusted to a range between 1 and 0
        meaning that parallell vectors and opposite vectors are both considered to be the same
        If False, the same shape but opposite signs gives maximum distance

    Returns
    -------
    ang_dist : float
        The angular distance between the two vectors, in the range [0, 1]
    """
    sim = cos_similary(vec1, vec2)
    if sign_invariant:
        sim = np.abs(sim)
    return np.arccos(sim) / np.pi

def cos_similary(vec1, vec2):
    """Computes <vec1, vec2> / (||vec1|| ||vec2||) which is cosine of the angle between the two vectors. 
        1 is paralell vectors, 0 is orthogonal, and -1 is opposite directions

    Parameters
    ----------
    vec1 : np.ndarray of any shape
        First vector. If the arrays have more than 1 axis, it will be flattened.
    vec2 : np.ndarray of same shape as vec1
        Second vector. If the arrays have more than 1 axis, it will be flattened.

    Returns
    -------
    cos_sim : float
        The cosine similarity between the two vectors
    """
    assert vec1.shape == vec2.shape
    vec1 = np.ravel(vec1)
    vec2 = np.ravel(vec2)
    norms = np.linalg.norm(vec1) *np.linalg.norm(vec2)
    if norms == 0:
        return np.nan
    ip = vec1.T @ vec2
    return ip / norms

def spatial_similarity(vec1, vec2):
    """Measures the spatial similarity between two vectors. Also known as the modal assurance criterion (MAC).

    1 is identical, 0 is fully dissimilar

    Implements |p^H q|^2 / (||p||^2 ||q||^2)
    
    Parameters
    ----------
    vec1 : np.ndarray of shape (..., N)
        First vector
    vec2 : np.ndarray of shape (..., N)
        Second vector

    Returns
    -------
    sim : float or ndarray of shape (...)
        The spatial similarity between the two vectors
    
    References
    ----------
    (25) in M. Hahmann and E. Fernandez-Grande, “A convolutional plane wave model for sound field reconstruction.” Aug. 24, 2022.
    """
    assert vec1.shape == vec2.shape

    denom = np.linalg.norm(vec1, axis=-1)**2 * np.linalg.norm(vec2, axis=-1)**2
    return np.abs(np.sum(vec1.conj() * vec2, axis=-1))**2 / denom
    #norm2 = np.linalg.norm(vec2, ord=2, axis=-1)**2



#=============== FOR COVARIANCE MATRICES =============

def corr_matrix_distance(mat1, mat2):
    """Computes the correlation matrix distance
    
    0 means that the matrices are equal up to a scaling
    1 means that they are maximally different (orthogonal in NxN dimensional space)

    Parameters
    ----------
    mat1 : np.ndarray of shape (..., N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (..., N, N)
        Second covariance matrix, should be symmetric and positive definite

    References
    ----------
    Correlation matrix distaince, a meaningful measure for evaluation of 
    non-stationary MIMO channels - Herdin, Czink, Ozcelik, Bonek
    """
    assert mat1.shape == mat2.shape
    norm1 = np.linalg.norm(mat1, ord="fro", axis=(-2,-1))
    norm2 = np.linalg.norm(mat2, ord="fro", axis=(-2,-1))
    if norm1 * norm2 == 0:
        return np.array(np.nan)
    return np.real_if_close(1 - np.trace(mat1 @ mat2) / (norm1 * norm2))


def covariance_distance_riemannian(mat1, mat2):
    """
    Computes the covariance matrix distance

    Parameters
    ----------
    mat1 : np.ndarray of shape (N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (N, N)
        Second covariance matrix, should be symmetric and positive definite

    Returns
    -------
    dist : float
        The distance between the two matrices
    
    Notes
    -----
    It is the distance of a canonical invariant Riemannian metric on the space 
    Sym+(n, R) of real symmetric positive definite matrices. 

    Invariant to affine transformations and inversions. 
    It is a distance measure, so 0 means equal and then it goes to infinity
    and the matrices become more unequal.

    When the metric of the space is the fisher information metric, this is the 
    distance of the space. See COVARIANCE CLUSTERING ON RIEMANNIAN MANIFOLDS
    FOR ACOUSTIC MODEL COMPRESSION - Shinohara, Masukp, Akamine

    References
    ----------
    A Metric for Covariance Matrices - Wolfgang Förstner, Boudewijn Moonen
    http://www.ipb.uni-bonn.de/pdfs/Forstner1999Metric.pdf
    """
    assert mat1.shape == mat2.shape
    assert mat1.shape[0] == mat1.shape[1]
    assert mat1.ndim == 2
    eigvals = splin.eigh(mat1, mat2, eigvals_only=True)
    return np.real_if_close(np.sqrt(np.sum(np.log(eigvals)**2)))


def covariance_distance_kl_divergence(mat1, mat2):
    """The Kullback Leibler divergence between two Gaussian
    distributions that has mat1 and mat2 as their covariance matrices. 

    Assumes both of these distributions has zero mean.

    It is a distance measure, so 0 means equal and then it goes to infinity
    and the matrices become more unequal.

    Parameters
    ----------
    mat1 : np.ndarray of shape (N, N)
        First covariance matrix, should be symmetric and positive definite
    mat2 : np.ndarray of shape (N, N)
        Second covariance matrix, should be symmetric and positive definite
    
    Returns
    -------
    dist : float
        The distance between the two matrices
    
    """
    assert mat1.shape == mat2.shape
    assert mat1.shape[0] == mat1.shape[1]
    assert mat1.ndim == 2
    N = mat1.shape[0]
    eigvals = splin.eigh(mat1, mat2, eigvals_only=True)

    det1 = splin.det(mat1)
    det2 = splin.det(mat2)
    common_trace = np.sum(eigvals)
    return np.real_if_close(np.sqrt((np.log(det2 / det1) + common_trace - N) / 2))

