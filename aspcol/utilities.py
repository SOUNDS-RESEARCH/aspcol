import numpy as np
import datetime
import time
from functools import wraps
import numpy as np

# only scalar for now
def is_power_of_2(x):
    return is_integer(np.log2(x))

def is_integer(x):
    if np.all(np.isclose(x, x.astype(int))):
        return True
    return False

def cart2pol(x, y):
    r = np.hypot(x, y)
    angle = np.arctan2(y, x)
    return (r, angle)

def pol2cart(r, angle):
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return (x, y)

def cart2spherical(cartCoord):
    """cartCoord is shape = (numPoints, 3)
        returns (r, angle), defined as in spherical2cart"""
    r = np.linalg.norm(cartCoord, axis=1)
    raise NotImplementedError

def spherical2cart(r, angle):
    """r is shape (numPoints, 1) or (numPoints)
        angle is shape (numPoints, 2)
        angle[:,0] is theta
        angle[:,1] is phi
        theta is normal polar coordinate angle, 0 is x direction, pi/2 is y direction
        phi is azimuth, 0 is z direction, pi is negative z direction"""
    numPoints = r.shape[0]
    cartCoord = np.zeros((numPoints,3))
    cartCoord[:,0] = np.squeeze(r) * np.cos(angle[:,0]) * np.sin(angle[:,1])
    cartCoord[:,1] = np.squeeze(r) * np.sin(angle[:,0]) * np.sin(angle[:,1])
    cartCoord[:,2] = np.squeeze(r) * np.cos(angle[:,1])
    return cartCoord

def getSmallestCoprime(N):
    assert N > 2 #don't have to deal with 1 and 2 at this point
    for i in range(2,N):
        if np.gcd(i,N):
            return i

def nextDivisible(divisor, minValue):
    """Gives the smallest integer divisible by divisor, 
        that is strictly larger than minValue"""
    rem = (minValue + divisor) % divisor
    return minValue + divisor - rem

def db2mag(db):
    return 10 ** (db / 20)

def mag2db(amp):
    return 20 * np.log10(amp)

def db2pow(db):
    return 10 ** (db / 10)

def pow2db(power):
    return 10 * np.log10(power)