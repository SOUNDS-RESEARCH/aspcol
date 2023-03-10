import numpy as np
import scipy.signal as spsig
import scipy.linalg as splin

import aspcol.matrices as mt

"""
If nothing else is stated, a matrix has the shape (num_taps, mat_dim1, mat_dim2)
For a parahermitian matrix the shape is (num_taps, mat_dim, mat_dim). 

It is required that num_taps is odd, so that A[t_c + t,:,:] for t_c= num_taps//2+1 
gives z^-t delay. Then A[t_c,:,:] is the zero-delay taps. 

"""

# ==================== GENERAL POLYNOMIAL MATRIX FUNCTIONS ===================
def paraconjugate(mat):
    return np.moveaxis(np.flip(mat, axis=0), 1, 2).conj()

def is_parahermitian(mat):
    return np.allclose(mat, paraconjugate(mat))

def is_paraunitary(mat):
    if mat.shape[1] != mat.shape[2]:
        return False
    return np.allclose(np.eye(mat.shape[1]), matmul(mat, paraconjugate(mat)))

def matmul(mat1, mat2):
    assert mat1.shape[2] == mat2.shape[1]
    assert mat1.dtype == mat2.dtype
    num_taps = mat1.shape[0] + mat2.shape[0] - 1
    outer_dim_l = mat1.shape[1]
    outer_dim_r = mat2.shape[2]
    inner_dim = mat1.shape[2]

    output_mat = np.zeros((num_taps, outer_dim_l, outer_dim_r), dtype=mat1.dtype)
    for i in range(outer_dim_l):
        for j in range(outer_dim_r):
            for k in range(inner_dim):
                output_mat[:,i,j] += np.convolve(mat1[:,i,k],  mat2[:,k,j], "full")
    return output_mat


#================== IMPLEMENTATION OF SBR2 ================================
def pevd_sbr2(R, tolerance, max_iter, trim_param):
    """
    R is a parahermitian matrix, 
        R.ndim == 3
        R.shape == (num_taps, mat_dim, mat_dim)
    
        tolerance is small positive real value
        max_iter is positive integer
        trim_param is positive real value close to 1
    """
    assert R.ndim == 3
    assert is_parahermitian(R)
    num_taps = R.shape[0]
    mat_dim = R.shape[-1]
    center = num_taps // 2
    R = R.astype(complex)

    max_val = 1+tolerance
    iter = 0
    Hp = np.zeros_like(R, dtype=R.dtype)
    Hp[center, :, :] = np.eye(mat_dim)
    #gs = []
    max_val_seq = []
    sq_norm_seq = []
    #N4 = np.linalg.norm(R)**2
    while max_val > tolerance and iter < max_iter:
        # 1) Find max off-diagonal element
        max_val_coord, max_val = off_diag_search(R)
        #print("j, k, tau, g, ||R||^2: ", j, k, tau, g, np.linalg.norm(R)**2)
        max_val_seq.append(max_val)
        sq_norm_seq.append(np.linalg.norm(R)**2)
        if max_val > tolerance:
            iter += 1
            B = delay_matrix(max_val_coord[2], max_val_coord[0], mat_dim=mat_dim, num_taps=num_taps, dtype=R.dtype)
            # Btilde, B = _delay_matrix_(k, tau, p=R.shape[0], max_tau=(R.shape[-1] - 1) // 2)
            Rp = matmul(matmul(B, R), paraconjugate(B))
            Hp = matmul(B, Hp)

            theta, phi = _get_rotation_angles(Rp, max_val_coord[1], max_val_coord[2])
            Q = rotation_matrix(mat_dim, max_val_coord[1], max_val_coord[2], theta, phi)

            Rp = matmul(matmul(Q, Rp), paraconjugate(Q))
            Hp = matmul(Q, Hp)

            R = trim(Rp, trim_param)
            num_taps = R.shape[0]
            # Hp = trim(Hp, 0.99, N4)
    return Hp, max_val_seq, sq_norm_seq


def delay_matrix(k, t, mat_dim, num_taps, dtype=float):
    """ Create an elementary delay polynomial matrix
    Arguments:
            k (int): The row/column number  to apply the delay
            t (int): The number of units time to delay by
            p (int): The dimension of the (square) pxp lag-zero matrix
            max_tau (int): The numbers of signals to locate
    """
    #assert np.abs(t) <= max_tau, "Delay time too long for depth of array"
    assert num_taps % 2 == 1
    assert np.abs(t) <= num_taps // 2 
    #num_taps = 2*max_tau + 1
    center = num_taps // 2
    B = np.zeros((num_taps, mat_dim, mat_dim), dtype=dtype)
    B[center, :, :] = np.eye(mat_dim)
    B[center, k, k] = 0
    B[center+t, k, k] = 1
    # Btilde = np.zeros((num_taps, p, p), dtype=complex)
    # Btilde[t_c, :, :] = np.eye(p)
    # Btilde[t_c, k, k] = 0
    # Btilde[t_c-t, k, k] = 1
    return B


def rotation_matrix(p, j, k, theta, phi):
    c = np.cos(theta)
    s = np.sin(theta)

    Q = np.zeros((1, p, p), dtype=complex)
    Q[0, :, :] = np.eye(p)
    Q[0, j, j] = c
    Q[0, j, k] = s*np.exp(1j*phi)
    Q[0, k, j] = -s*np.exp(-1j*phi)
    Q[0, k, k] = c

    # QH = np.zeros((p, p, 2*max_tau+1), dtype=complex)
    # QH[:, :, 0] = np.eye(p)
    # QH[j, j, 0] = cs
    # QH[j, k, 0] = -ss*np.exp(1j*phi)
    # QH[k, j, 0] = ss*np.exp(-1j*phi)
    # QH[k, k, 0] = cs
    return Q

def _get_rotation_angles(mat, j, k):
    num_taps = mat.shape[0]
    center = num_taps // 2

    phi = np.angle(mat[center, j, k])
    theta = np.arctan2(2*np.abs(mat[center, j, k]), np.real_if_close(mat[center, j, j]-mat[center, k, k])) / 2

    # cs = np.cos()
    # ss = np.sin(np.arctan2((2*np.abs(X[j, k, 0])), (X[j, j, 0].real-X[k, k, 0].real))/2)
    return theta, phi

def off_diag_search(mat):
    assert mat.ndim == 3
    assert mat.shape[1] == mat.shape[2]
    num_taps = mat.shape[0]
    mat_dim = mat.shape[2]
    center = num_taps // 2
    #maxtau = (X.shape[-1]-1)//2
    max_val_coords = [np.nan, np.nan, np.nan]
    max_val = 0
    for j in range(mat_dim):
        for k in range(j+1, mat_dim):
            for t in range(num_taps):
                val = np.abs(mat[t, j, k])
                if val > max_val:
                    max_val_coords = [t, j, k]
                    max_val = val

    if max_val_coords[0] > center:
        max_val_coords[0] = max_val_coords[0]-mat.shape[-1]
    return max_val_coords, max_val

def trim(R, mu):
    num_taps = R.shape[0]
    center = num_taps // 2
    #max_tau = (R.shape[-1]-1)//2
    sq_norm_orig = np.linalg.norm(R)**2

    #tod = -1
    num_samples_trim = 0
    for i in range(center):
        D = R[i:-i, :, :]
        sq_norm_trim = np.linalg.norm(D)**2
        if sq_norm_trim >= (1-mu)*sq_norm_orig:
            #tod += 1
            num_samples_trim = i
        else:
            break
    if num_samples_trim > 0:
        R = R[num_samples_trim:-num_samples_trim,:,:]
    return R





#===========================Everything below is copied from AlexW335 ========================

def __SBR2(R, delta, maxiter):
    R0 = R.copy()
    g = 1+delta
    iter = 0
    Hp = np.zeros_like(R, dtype='complex128')
    Hp[:, :, 0] = np.eye(R.shape[0])
    gs = []
    r2s = []
    N4 = np.linalg.norm(R)**2
    while g > delta and iter < maxiter:
        # 1) Find max off-diagonal element
        j, k, tau, g = off_diag_search(R)
        print("j, k, tau, g, ||R||^2: ", j, k, tau, g, np.linalg.norm(R)**2)
        gs.append(g)
        r2s.append(np.linalg.norm(R)**2)
        if g > delta:
            iter += 1
            B, Btilde = delay_matrix(k, tau, p=R.shape[0], max_tau=(R.shape[-1]-1)//2)
            # Btilde, B = _delay_matrix_(k, tau, p=R.shape[0], max_tau=(R.shape[-1] - 1) // 2)
            Rpt = fftpmm(B, R)
            Rp = fftpmm(Rpt, Btilde)
            Hp = fftpmm(B, Hp)
            Q, QH = rotation_matrix(j, k, Rp)
            Rt = fftpmm(Q, Rp)
            Rp = fftpmm(Rt, QH)
            Hp = fftpmm(Q, Hp)
            R = trim(Rp, 0.99, N4)
            # Hp = trim(Hp, 0.99, N4)
    return Hp, gs, r2s

def __MS_SBR2(R, delta, maxiter):
    N4 = np.linalg.norm(R)**2
    g = 1+delta
    iter = 0
    Hp = np.zeros_like(R, dtype='complex128')
    Hp[:, :, 0] = np.eye(R.shape[0])
    gs = []
    r2s = []
    Rp = R
    while iter < maxiter and g > delta:
        # Locate and shift max off-diagonal elements
        Li = 0
        blacklist = []
        prs = []
        Rps = []
        while g > delta and len(blacklist) < R.shape[0]-1:
            j, k, tau, g = off_diag_search(R, blacklist=blacklist)
            print("j, k, tau, g, ||R||^2: ", j, k, tau, g, np.linalg.norm(R)**2)
            prs.append((j, k))
            blacklist.append(j)
            blacklist.append(k)
            if g > delta:
                P, Ph = delay_matrix(k, tau, p=R.shape[0], max_tau=(R.shape[-1]-1)//2)
                Rp = fftpmm(P, fftpmm(Rp, Ph))
                Hp = fftpmm(P, Hp)
                Li += 1
        # Perform a sequence of Jacobi rotations
        for (j, k) in prs:
            Q, Qh = rotation_matrix(j, k, Rp)
            Rp = fftpmm(Q, fftpmm(Rp, Qh))
            Hp = fftpmm(Q, Hp)
        iter += 1
        print(iter)
        r2s.append(np.linalg.norm(Rp)**2)
        Rp = trim(Rp, 0.99, N4)
    return Hp, gs, r2s

def __trim(R, mu, N4):
    max_tau = (R.shape[-1]-1)//2
    tod = -1
    for idx in np.arange(max_tau):
        D = R[:, :, max_tau-idx:max_tau+idx+2]
        if np.linalg.norm(D)**2 <= (1-mu)*N4:
            tod += 1
        else:
            break
    if tod >= 0:
        R[:, :, max_tau-tod:max_tau+tod+2] = np.zeros((R.shape[0], R.shape[1], (tod+1)*2))
    return R

def __paraconjugate(X):
    Xtmp = X[:, :, 1:]
    Xtmp = np.flip(Xtmp, -1)
    Xtmp = np.concatenate((X[:, :, 0:1], Xtmp), -1)
    Xtmp = np.conj(np.transpose(Xtmp, (1, 0, 2)))
    return Xtmp

def __off_diag_search(X, blacklist=()):
    # 1) Find dominant off-diagonal element
    maxtau = (X.shape[-1]-1)//2
    coords = None
    Xcoords = 0
    for j in [x for x in np.arange(X.shape[0]) if x not in blacklist]:
        for k in [y for y in np.arange(j+1, X.shape[1]) if y not in blacklist]:
            for t in np.arange(X.shape[-1]):
                if np.abs(X[j, k, t]) > Xcoords:
                    coords = [j, k, t]
                    Xcoords = np.abs(X[j, k, t])
    if coords[-1] > maxtau:
        coords[-1] = coords[-1]-X.shape[-1]
    return coords[0], coords[1], coords[2], Xcoords


def __rotation_matrix(j, k, X):
    p = X.shape[0]
    max_tau = (X.shape[-1]-1)//2
    phi = np.angle(X[j, k, 0])

    cs = np.cos(np.arctan2((2*np.abs(X[j, k, 0])), (X[j, j, 0].real-X[k, k, 0].real))/2)
    ss = np.sin(np.arctan2((2*np.abs(X[j, k, 0])), (X[j, j, 0].real-X[k, k, 0].real))/2)

    Q = np.zeros((p, p, 2*max_tau+1), dtype='complex128')
    Q[:, :, 0] = np.eye(p)
    Q[j, j, 0] = cs
    Q[j, k, 0] = ss*np.exp(1j*phi)
    Q[k, j, 0] = -ss*np.exp(-1j*phi)
    Q[k, k, 0] = cs
    # print("Q: ",Q[:,:,0])
    QH = np.zeros((p, p, 2*max_tau+1), dtype='complex128')
    QH[:, :, 0] = np.eye(p)
    QH[j, j, 0] = cs
    QH[j, k, 0] = -ss*np.exp(1j*phi)
    QH[k, j, 0] = ss*np.exp(-1j*phi)
    QH[k, k, 0] = cs
    return Q, QH

def __fftpmm(H, R):
    """Algorithm 1 taken from Redif, S., & Kasap, S. (2015). Novel Reconfigurable Hardware Architecture for Polynomial Matrix Multiplications. Tvlsi, 23(3), 454–465."""
    (p, _, N) = H.shape
    Hfft = np.apply_along_axis(fft_pack.fft, -1, H, n=N)
    Rfft = np.apply_along_axis(fft_pack.fft, -1, R, n=N)
    reHfft = Hfft.real
    imHfft = Hfft.imag
    reRfft = Rfft.real
    imRfft = Rfft.imag
    reCfft = np.zeros((p, p, N))
    imCfft = np.zeros((p, p, N))

    for ix in np.arange(p):
        for jx in np.arange(p):
            for kx in np.arange(2*p):
                for t in np.arange(N):
                    if kx < p:
                        reCfft[ix, jx, t] += reHfft[ix, kx, t]*reRfft[kx, jx, t]
                        imCfft[ix, jx, t] += reHfft[ix, kx, t]*imRfft[kx, jx, t]
                    else:
                        reCfft[ix, jx, t] -= imHfft[ix, kx-p, t]*imRfft[kx-p, jx, t]
                        imCfft[ix, jx, t] += imHfft[ix, kx-p, t]*reRfft[kx-p, jx, t]
    # for ix in np.arange(p):
    #     for jx in np.arange(p):
    #         for kx in np.arange(2*p):
    #             for t in np.arange(N):
    #                 if kx < p:
    #                     imCfft[ix, jx, t] += reHfft[ix, kx, t]*imRfft[kx, jx, t]
    #                 else:
    #                     imCfft[ix, jx, t] += imHfft[ix, kx-p, t]*reRfft[kx-p, jx, t]
    Cfft = reCfft+1.0j*imCfft
    C = np.apply_along_axis(fft_pack.ifft, -1, Cfft, n=N)
    # plt.plot(abs(C.flatten()))
    # plt.show()
    # C.real[abs(C.real) < 1.0e-8] = 0
    # C.imag[abs(C.imag) < 1.0e-8] = 0
    # plt.plot(abs(C.flatten()))
    # plt.show()
    return C


def print_polynomial(z):
    prst = ""
    maxt = (len(z)-1)//2
    for pos in np.arange(maxt):
        if z[pos]!=0:
            if z[pos]==1:
                prst += "z^{} + ".format(pos)
            else:
                prst += "{}z^{} + ".format(z[pos], pos)
    if z[maxt-1]!=0:
        if z[maxt-1]==1:
            prst += "z + "
        else:
            prst += "{}z + ".format(z[maxt-1])
    if z[maxt]!=0:
        prst += "{} + ".format(z[maxt])
    for pos in np.arange(maxt+1, len(z)):
        if z[pos]!=0:
            if z[pos]==1:
                prst += "z^-{} + ".format(pos)
            else:
                prst += "{}z^{} + ".format(z[pos], maxt-pos)

    print(prst[:-2])

def polynom_steervec(mics, theta, max_tau=100, sound_speed=343.1):
    incidentdir = np.array([-np.cos(theta), -np.sin(theta)])
    incidentdir.shape = (2, 1)
    tau = np.dot(mics, incidentdir)/sound_speed
    Az = np.sinc(np.tile(np.arange(-max_tau, max_tau+1), (4, 1))-tau)
    Az *= np.blackman(Az.shape[-1])
    Az = np.flip(np.roll(Az, max_tau), -1)
    return Az