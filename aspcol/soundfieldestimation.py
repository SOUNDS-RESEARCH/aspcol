import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special





def estimate_soundfield(p):
    """Implements the soundfield estimation method of
    Ueno, Koyama, Saruwatari - Directionally weighted wave field 
    estimation exploiting prior information on source direction
    

    Parameters
    ----------


    Returns
    -------

    """
    pass




def coefEstOprGen(posEst, orderEst, posMic, orderMic, coefMic, k, reg=1e-3):
    """Generate operator to estimate expansion coefficients of spherical wavefunctions from measurement vectors
    - N. Ueno, S. Koyama, and H. Saruwatari, “Sound Field Recording Using Distributed Microphones Based on 
      Harmonic Analysis of Infinite Order,” IEEE SPL, DOI: 10.1109/LSP.2017.2775242, 2018.
    Parameters
    ------
    posEst: Position of expansion center for estimation
    orderEst: Maximum order for estimation
    poMic: Microphone positions
    orderMic: Maximum order of microphone directivity
    coefMic: Expansion coefficients of microphone directivity
    Returns
    ------
    Operator for estimation
    (Expansion coefficeints are estimated by multiplying with measurement vectors)
    """
    numMic = posMic.shape[0]
    if np.isscalar(k):
        numFreq = 1
        k = np.array([k])
    else:
        numFreq = k.shape[0]

    Xi = np.zeros((numFreq, (orderEst+1)**2, numMic), dtype=complex)
    Psi = np.zeros((numFreq, numMic, numMic), dtype=complex)
    for ff in np.arange(numFreq):
        print('Frequency: %d/%d' % (ff, numFreq))
        for j in np.arange(numMic):
            T = trjmat3d(orderEst, orderMic, posEst[0, 0]-posMic[j, 0], posEst[0, 1]-posMic[j, 1], posEst[0, 2]-posMic[j, 2], k[ff])
            Xi[ff, :, j] = T @ coefMic[:, j]
            Psi[ff, j, j] = coefMic[:, j].conj().T @ coefMic[:, j]
            for i in np.arange(j, numMic):
                T = trjmat3d(orderMic, orderMic, posMic[i, 0]-posMic[j, 0], posMic[i, 1]-posMic[j, 1], posMic[i, 2]-posMic[j, 2], k[ff])
                Psi[ff, i, j] = coefMic[:, i].conj().T @ T @ coefMic[:, j]
                Psi[ff, j, i] = Psi[ff, i, j].conj()
    eigPsi, _ = np.linalg.eig(Psi)
    regPsi =  eigPsi[:,0] * reg
    Psi_inv = np.linalg.inv(Psi + regPsi[:,None,None] * np.eye(numMic, numMic)[None, :, :])
    coefEstOpr = Xi @ Psi_inv

    return coefEstOpr



def trjmat3d(order1, order2, x, y, z, k):
    """Translation operator in 3D

    Taken directly from https://github.com/sh01k/MeshRIR/blob/main/example/sf_func.py
    """
    if np.all([x, y, z] == 0):
        T = np.eye((order1+1)**2, (order2+1)**2)
        return T
    else:
        order = order1 + order2
        n, m = sph_harm_nmvec(order)
        P = sf_int_basis3d(n, m, x, y, z, k)
        T = np.zeros(((order1+1)**2, (order2+1)**2), dtype=complex)

        icol = 0
        for n in np.arange(0, order2+1):
            for m in np.arange(-n, n+1):
                irow = 0
                for nu in np.arange(0, order1+1):
                    for mu in np.arange(-nu, nu+1):
                        l = np.arange((n+nu), max( [np.abs(n-nu), np.abs(m-mu)] )-1, -2)
                        G = np.zeros(l.shape)
                        for ll in np.arange(0, l.shape[0]):
                            G[ll] = gaunt_coef(n, m, nu, -mu, l[ll])
                        T[irow, icol] = np.sqrt(4.*np.pi) * 1j**(nu-n) * (-1.)**m * np.sum( 1j**(l) * P[l**2 + l - (mu-m)] * G )
                        irow = irow + 1
                icol = icol+1
        return T

def gaunt_coef(l1, m1, l2, m2, l3):
    """Gaunt coefficients

    Taken directly from https://github.com/sh01k/MeshRIR/blob/main/example/sf_func.py
    """
    m3 = -m1 - m2
    l = int((l1 + l2 + l3) / 2)

    t1 = l2 - m1 - l3
    t2 = l1 + m2 - l3
    t3 = l1 + l2 - l3
    t4 = l1 - m1
    t5 = l2 + m2

    tmin = max([0, max([t1, t2])])
    tmax = min([t3, min([t4, t5])])

    t = np.arange(tmin, tmax+1)
    gl_tbl = np.array(special.gammaln(np.arange(1, l1+l2+l3+3)))
    G = np.sum( (-1.)**t * np.exp( -np.sum( gl_tbl[np.array([t, t-t1, t-t2, t3-t, t4-t, t5-t])] )  \
                                  +np.sum( gl_tbl[np.array([l1+l2-l3, l1-l2+l3, -l1+l2+l3, l])] ) \
                                  -np.sum( gl_tbl[np.array([l1+l2+l3+1, l-l1, l-l2, l-l3])] ) \
                                  +np.sum( gl_tbl[np.array([l1+m1, l1-m1, l2+m2, l2-m2, l3+m3, l3-m3])] ) * 0.5 ) ) \
        * (-1.)**( l + l1 - l2 - m3) * np.sqrt( (2*l1+1) * (2*l2+1) * (2*l3+1) / (4*np.pi) )
    return G





def sph_harm_nmvec(order, rep=None):
    """Vectors of spherical harmonic orders and degrees
    Returns (order+1)**2 size vectors of n and m
    n = [0, 1, 1, 1, ..., order, ..., order]^T
    m = [0, -1, 0, 1, ..., -order, ..., order]^T
    Parameters
    ------
    order: Maximum order
    rep: Same vectors are copied as [n, .., n] and [m, ..., m]
    Returns
    ------
    n, m: Vectors of orders and degrees

    Taken directly from https://github.com/sh01k/MeshRIR/blob/main/example/sf_func.py
    """
    n = np.array([0])
    m = np.array([0])
    for nn in np.arange(1, order+1):
        nn_vec = np.tile([nn], 2*nn+1)
        n = np.append(n, nn_vec)
        mm = np.arange(-nn, nn+1)
        m = np.append(m, mm)
    if rep is not None:
        n = np.tile(n[:, None], (1, rep))
        m = np.tile(m[:, None], (1, rep))
    return n, m


def sf_int_basis3d(n, m, x, y, z, k):
    """Spherical wavefunction for interior sound field in 3D
    
    Parameters
    ------
    n, m: orders and degrees
    x, y, z: Position in Cartesian coordinates
    k: Wavenumber
    Returns
    ------
    sqrt(4pi) j_n(kr) Y_n^m(phi,theta)
    (Normalized so that 0th order coefficient corresponds to pressure)
    Taken directly from https://github.com/sh01k/MeshRIR/blob/main/example/sf_func.py
    """
    phi, theta, r = cart2sph(x, y, z)
    J = special.spherical_jn(n, k * r)
    Y = special.sph_harm(m, n, phi, theta)
    f = np.sqrt(4*np.pi) * J * Y
    return f



def sph2cart(phi, theta, r):
    """Conversion from spherical to Cartesian coordinates
    Parameters
    ------
    phi, theta, r: Azimuth angle, zenith angle, distance
    Returns
    ------
    x, y, z : Position in Cartesian coordinates

    Taken directly from https://github.com/sh01k/MeshRIR/blob/main/example/sf_func.py
    should be replaced with util.spherical2cart
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cart2sph(x, y, z):
    """Conversion from Cartesian to spherical coordinates
    Parameters
    ------
    x, y, z : Position in Cartesian coordinates
    Returns
    ------
    phi, theta, r: Azimuth angle, zenith angle, distance

    Taken directly from https://github.com/sh01k/MeshRIR/blob/main/example/sf_func.py
    should be replaced with util.cart2spherical
    """
    r_xy = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(r_xy, z)
    r = np.sqrt(x**2 + y**2 + z**2)
    return phi, theta, r