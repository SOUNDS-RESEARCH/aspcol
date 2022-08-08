import numpy as np
import scipy.spatial.distance as distfuncs
import scipy.special as special
import scipy.signal as sig
import numba as nb

import ancsim.utilities as util
import ancsim.signal.filterdesign as fd
import ancsim.integration.montecarlo as mc
import ancsim.integration.pointgenerator as gen
import ancsim.signal.filterclasses as fc
import ancsim.soundfield.roomimpulseresponse as rir


def kernelHelmholtz2d(points1, points2, waveNum):
    """points1 is shape (numPoints1, 2)
        points2 is shape (numPoints2, 2)
        waveNum is shape (numFreqs)

        returns shape (numFreqs, numPoints1, numPoints2)
    """
    distMat = distfuncs.cdist(points1, points2)
    return special.j0(distMat[None,:,:] * waveNum[:,None,None])

def kernelHelmholtz3d(points1, points2, waveNum):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)

        returns shape (numFreqs, numPoints1, numPoints2)
    """
    distMat = distfuncs.cdist(points1, points2)
    return special.spherical_jn(0, distMat[None,:,:] * waveNum[:,None,None])

def kernelDirectional3d(points1, points2, waveNum, angle, beta):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)
        angle is tuple (theta, phi) defined as in util.spherical2cart
        
        returns shape (numFreqs, numPoints1, numPoints2)
    """
    rDiff = points1[:,None,:] - points2[None,:,:]
    angleFactor = beta * util.spherical2cart(np.ones((1,1)), np.array(angle)[None,:])[None,None,...]
    posFactor = 1j * waveNum[:,None,None,None] * rDiff[None,...]
    return special.spherical_jn(0, 1j*np.sqrt(np.sum((angleFactor + posFactor)**2, axis=-1)))

@nb.njit
def kernelDirectionalVec3d(points1, points2, waveNum, directionVec, beta):
    """points1 is shape (numPoints1, 3)
        points2 is shape (numPoints2, 3)
        waveNum is shape (numFreqs)
        directionVec is shape (numAngles, 3), where each (1,3) is a unit vector
        
        returns shape (numFreqs, numAngles, numPoints1, numPoints2)
    
        Identical to kernelDirectional3d, but accepts the direction
        in the form of a unit vector instead of angles. 
    """
    angleTerm = 1j * beta * directionVec.reshape((1,-1,1,1,directionVec.shape[-1]))
    posTerm = waveNum.reshape((-1,1,1,1,1)) * (points1.reshape((1,1,-1,1,points1.shape[-1])) - points2.reshape((1,1,1,-1,points2.shape[-1])))
    return np.sinc(np.sqrt(np.sum((angleTerm - posTerm)**2, axis=-1)) / np.pi)


# def kernelDirectionalVec3d(points1, points2, waveNum, directionVec, beta):
#     """points1 is shape (numPoints1, 3)
#         points2 is shape (numPoints2, 3)
#         waveNum is shape (numFreqs)
#         directionVec is shape (numAngles, 3), where each (1,3) is a unit vector
        
#         returns shape (numFreqs, numAngles, numPoints1, numPoints2)
    
#         Identical to kernelDirectional3d, but accepts the direction
#         in the form of a unit vector instead of angles. 
#     """
#     #angleFactor = beta * util.spherical2cart(np.ones((1,1)), np.array(angle)[None,:])[None,None,...]
#     angleTerm = 1j * beta * directionVec[None, :,None, None, :]
#     posTerm = waveNum[:,None,None,None,None] * (points1[None,None,:,None,:] - points2[None,None,None,:,:])
#     return special.spherical_jn(0, np.sqrt(np.sum((angleTerm - posTerm)**2, axis=-1)))



def kernelReciprocal3d(points1, points2, waveNum):
    """points is a tuple (micPoints, srcPoints),
        where micPoints is ndarray of shape (numMic, 3)
        srcPoints is ndarray of shape (numSrc, 3)

        waveNum is shape (numFreq)
        output has shape (numFreq, numMic1*numSrc1, numMic2*numSrc2)
        they are placed according to micIdx+numMics*srcIdx

        When flattened, the index for microphone m, speaker l is m+l*M. i.e.
        the microphone index changes faster. 

    From the paper: Kernel interpolation of acoustic transfer 
                function between regions considering reciprocity"""
    waveNum = waveNum[:,None,None,None,None]
    micDist = distfuncs.cdist(points1[0], points2[0])[None,None,:,None,:]
    srcDist = distfuncs.cdist(points1[1], points2[1])[None,:,None,:,None]
    mixDist1 = distfuncs.cdist(points1[0], points2[1])[None,None,:,:,None]
    mixDist2 = distfuncs.cdist(points1[1], points2[0])[None,:,None,None,:]

    kVal = 0.5 * (special.spherical_jn(0, waveNum * micDist) * \
                        special.spherical_jn(0, waveNum * srcDist)) + \
                        (special.spherical_jn(0, waveNum * mixDist1) * \
                        special.spherical_jn(0, waveNum * mixDist2))
    kVal = np.reshape(kVal, kVal.shape[:3]+(-1,))
    kVal = np.reshape(kVal, (kVal.shape[0], -1,kVal.shape[-1]))
    return kVal


def getKernelWeightingFilter(kernelFunc, regParam, micPos, integralDomain, 
                                mcSamples, numFreq, samplerate, c, *args):
    """Calculates kernel weighting filter A(w) in frequency domain
        
        kernelFunc is one of kernelHelmholtz3d, kernelHelmholtz2d, kernelDirectional3d as defined in this module
        regParam is positive scalar
        micPos is ndarray with shape (numPositions, spatialDimension)
        integralDomain is subclass of Region object, found in soundfield/geometry module
        mcSamples is integer, how many monte carlo samples to be drawn for integration
        samplerate is integer
        c is speed of sound
        *args are arguments needed for kernel function except points1, points2, waveNum
        
        For both diffuse and directional kernel P^H = P, so the hermitian tranpose should not do anything
        It is left in place in case a kernel function in the future changes that identity. """
    freqs = fd.getFrequencyValues(numFreq, samplerate)
    waveNum = 2 * np.pi * freqs / c

    def integrableFunc(r):
        kappa = kernelFunc(r, micPos, waveNum, *args)
        kappa = np.transpose(kappa,(0,2,1))
        return kappa.conj()[:,:,None,:] * kappa[:,None,:,:]

    numMics = micPos.shape[0]
    K = kernelFunc(micPos, micPos, waveNum, *args)
    P = np.linalg.pinv(K + regParam * np.eye(numMics))

    integralValue = mc.integrate(integrableFunc, integralDomain.sample_points, mcSamples, integralDomain.volume)
    weightingFilter = np.transpose(P,(0,2,1)).conj() @ integralValue @ P

    weightingFilter = fd.insertNegativeFrequencies(weightingFilter, even=True)
    return weightingFilter



def getKRRParameters(kernelFunc, regParam, outputArg, dataArg, *args):
    """Calculates parameter vector or matrix given a kernel function for Kernel Ridge Regression.
    Both dataArg and outputArg should be formatted as (numPoints, pointDimension)
    kernelFunc should return args as (numfreq, numPoints1, numPoints2)
    returns params of shape (numFreq, numOutPoints, numDataPoints)"""
    K = kernelFunc(dataArg, dataArg, *args)
    Kreg = K + regParam * np.eye(K.shape[-1])
    kappa = np.transpose(kernelFunc(outputArg, dataArg, *args), (0,2,1))

    params = np.transpose(np.linalg.solve(Kreg, kappa), (0, 2, 1))
    return params

# def getKRRParameters(kernelFunc, regParam, outputArg, dataArg, *args):
#     """Calculates parameter vector or matrix given a kernel function for Kernel Ridge Regression.
#     Both dataArg and outputArg should be formatted as (numPoints, pointDimension)
#     kernelFunc should return args as (numfreq, numPoints1, numPoints2)
#     returns params of shape (numFreq, numOutPoints, numDataPoints)"""
#     dataDim = dataArg.shape[0]
#     K = kernelFunc(dataArg, dataArg, *args)
#     Kreg = K + regParam * np.eye(dataDim)
#     kappa = kernelFunc(dataArg, outputArg, *args)

#     params = np.transpose(np.linalg.solve(Kreg, kappa), (0, 2, 1))
#     return params


def soundfieldInterpolationFIR(
    toPoints, fromPoints, irLen, regParam, numFreq, spatialDims, samplerate, c
):
    """Convenience function for calculating the time domain causal FIR interpolation filter
    from a set of points to a set of points. """
    assert numFreq > irLen
    freqFilter = soundfieldInterpolation(
        toPoints, fromPoints, numFreq, regParam, spatialDims, samplerate, c
    )
    kiFilter,_ = fd.firFromFreqsWindow(freqFilter, irLen)
    return kiFilter

def soundfieldInterpolation(
    toPoints, fromPoints, numFreq, regParam, spatialDims, samplerate, c
):
    """ Convenience function for calculating the frequency domain interpolation filter
    from a set of points to a set of points. """
    if spatialDims == 3:
        kernelFunc = kernelHelmholtz3d
    elif spatialDims == 2:
        kernelFunc = kernelHelmholtz2d
    else:
        raise ValueError

    assert numFreq % 2 == 0

    freqs = fd.getFrequencyValues(numFreq, samplerate)#[:, None, None]
    waveNum = 2 * np.pi * freqs / c
    ipParams = getKRRParameters(kernelFunc, regParam, toPoints, fromPoints, waveNum)
    ipParams = fd.insertNegativeFrequencies(ipParams, even=True)
    return ipParams




class ATFKernelInterpolator():
    """Uses the method kernel interpolation with reciprocity (by ribeiro)
        It can interpolate the source positions as well. 
        
        Currently implemented only for interpolating between different speakers using the 
        same set of microphones"""
    def __init__(self, kiFromSpeakerPos, kiToSpeakerPos, micPos, regParam, kiFiltLen, atfLen, numFreq, samplerate, c, atfDelay=0):
        assert kiFiltLen % 2 == 1
        self.kiFiltLen = kiFiltLen
        self.kiDly = self.kiFiltLen // 2
        self.atfLen = atfLen

        self.kiFromSpeakerPos = kiFromSpeakerPos
        self.kiToSpeakerPos = kiToSpeakerPos
        self.micPos = micPos

        self.numSpeakerFrom = self.kiFromSpeakerPos.shape[0]
        self.numSpeakerTo = self.kiToSpeakerPos.shape[0]
        self.numMics = self.micPos.shape[0]

        waveNum = 2 * np.pi * fd.getFrequencyValues(numFreq, samplerate) / c
        # kiTF = getKRRParameters(kernelReciprocal3d, regParam, 
        #                     (self.micPos, self.kiFromSpeakerPos),
        #                     (self.micPos, self.kiToSpeakerPos),
        #                     waveNum)
        kiTF = getKRRParameters(kernelReciprocal3d, regParam, 
                            (self.micPos, self.kiToSpeakerPos),
                            (self.micPos, self.kiFromSpeakerPos),
                            waveNum)
        kiTF = fd.insertNegativeFrequencies(kiTF, even=True)
        
        kiIR,_ = fd.firFromFreqsWindow(kiTF, self.kiFiltLen)
        kiIR = np.transpose(kiIR, (1,0,2))
        #kiIR = np.pad(kiIR, ((0,0),(0,0),(0,self.kiDly+100)))
        self.kiFilt = fc.createFilter(ir=kiIR)

        self.directCompFrom = rir.irRoomImageSource3d(self.kiFromSpeakerPos, self.micPos, 
                                        [10,10,5], [0,0,0], self.atfLen-atfDelay, 0, samplerate, c)
        #self.directCompFrom = np.moveaxis(self.directCompFrom, 0,1)
        self.directCompFrom = np.pad(self.directCompFrom, ((0,0),(0,0),(atfDelay,0)))
        #self.directCompFrom = self.directCompFrom.reshape((-1, self.directCompFrom.shape[-1]))


        self.directCompTo = rir.irRoomImageSource3d(self.kiToSpeakerPos, self.micPos, 
                                        [10,10,5], [0,0,0], self.atfLen-atfDelay, 0, samplerate, c)
        self.directCompTo = np.pad(self.directCompTo, ((0,0),(0,0),(atfDelay,0)))
        #self.directCompTo = self.directCompTo.reshape((-1, self.directCompTo.shape[-1]))


        #self.kiFilt = FilterSum_Freqdomain(ir=np.concatenate((kiIR, np.zeros(kiIR.shape[:-1]+(self.blockSize-kiIR.shape[-1],))),axis=-1))

    def process(self, kiFromIr):
        assert kiFromIr.shape == (self.numSpeakerFrom, self.numMics, self.atfLen)
        self.kiFilt.buffer.fill(0)
        reverbComp = kiFromIr - self.directCompFrom
        reverbComp = reverbComp.reshape(-1, self.atfLen)
        reverbComp = np.pad(reverbComp, ((0,0),(0,self.kiDly)))

        interpolatedIr = self.kiFilt.process(reverbComp)
        truncIPIr = interpolatedIr[...,self.kiDly:]
        truncIPIr = truncIPIr.reshape(self.numSpeakerTo, self.numMics, self.atfLen)
        truncIPIr += self.directCompTo
        return truncIPIr

    def process_alt(self, ir):
        assert ir.shape == (self.numSpeakerFrom, self.numMics, self.atfLen)
        ir -= self.directCompFrom
        ir = ir.reshape(-1, ir.shape[-1])
        ir_ip = np.zeros((self.numMics*self.numSpeakerTo, self.atfLen+self.kiFiltLen-1))
        for inIdx in range(self.kiFilt.ir.shape[0]):
            for outIdx in range(self.kiFilt.ir.shape[1]):
                ir_ip[outIdx,:] += np.convolve(self.kiFilt.ir[inIdx, outIdx, :], ir[inIdx,:], "full")
        ir_ip = ir_ip.reshape(self.numSpeakerTo, self.numMics, self.atfLen+self.kiFiltLen-1)
        ir_ip = ir_ip[...,self.kiDly:-self.kiDly]
        ir_ip += self.directCompTo
        return ir_ip


# def kiFilter(kernelFunc, regParam, toPoints, fromPoints, numFreq, samplerate, c, *args):
#     """ Convenience function for calculating the frequency domain interpolation filter
#     from a set of points to a set of points. Uses any kernel, in contrast to soundfieldInterpolation
#     Returns frequency domain coefficients. Use with filterdesign.firFromFreqsWindow() for FIR filter"""
#     assert numFreq
#     freqs = fd.getFrequencyValues(numFreq, samplerate)#[:, None, None]
#     waveNum = 2 * np.pi * freqs / c
#     ipParams = getKRRParameters(kernelFunc, regParam, toPoints, fromPoints, waveNum, *args)
#     ipParams = fd.insertNegativeFrequencies(ipParams, even=True)
#     return ipParams





def analyticKernelWeightingDisc2d(errorMicPos, freq, regParam, truncOrder, radius, c):
    """Analytic solution of the kernel interpolation weighting filter integral
        in the frequency domain for a disc in 2D. """
    if isinstance(freq, (int, float)):
        freq = np.array([freq])
    if len(freq.shape) == 1:
        freq = freq[:, np.newaxis, np.newaxis]
    waveNumber = 2 * np.pi * freq / c
    K = special.j0(waveNumber * distfuncs.cdist(errorMicPos, errorMicPos))
    P = np.linalg.pinv(K + regParam * np.eye(K.shape[-1]))
    S = getS(truncOrder, waveNumber, errorMicPos)
    Gamma = getGamma(truncOrder, waveNumber, radius)
    A = (
        np.transpose(P.conj(), (0, 2, 1))
        @ np.transpose(S.conj(), (0, 2, 1))
        @ Gamma
        @ S
        @ P
    )
    return A

def getGamma(maxOrder, k, R):
    matLen = 2 * maxOrder + 1
    diagValues = smallGamma(np.arange(-maxOrder, maxOrder + 1), k, R)
    gamma = np.zeros((diagValues.shape[0], matLen, matLen))

    gamma[:, np.arange(matLen), np.arange(matLen)] = diagValues
    return gamma

def smallGamma(mu, k, R):
    Jfunc = special.jv((mu - 1, mu, mu + 1), k * R)
    return np.pi * (R ** 2) * ((Jfunc[:, 1, :] ** 2) - Jfunc[:, 0, :] * Jfunc[:, 2, :])

def getS(maxOrder, k, positions):
    r, theta = util.cart2pol(positions[:, 0], positions[:, 1])

    mu = np.arange(-maxOrder, maxOrder + 1)[:, np.newaxis]
    S = special.jv(mu, k * r) * np.exp(theta * mu * (-1j))
    return S
