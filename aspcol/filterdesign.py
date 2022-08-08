import numpy as np
import scipy.signal as signal
import ancsim.signal.freqdomainfiltering as fdf
import itertools as it


# =======================================================================


def getFrequencyValues(numFreq, samplerate):
    """Get the frequency values of all positive frequency bins in Hz
    numFreq is the number of frequency bins INCLUDING negative frequencies
    If numFreq is even, it will return (numFreq/2)+1 values
    If numFreq is odd, it will return (numFreq+1)/2 values
    use np.fft.fftfreq to double check values when implementing. 
    """
    if numFreq % 2 == 0:
        return (samplerate / (numFreq)) * np.arange(numFreq // 2 + 1)
    elif numFreq % 2 == 1:
        raise NotImplementedError
    else:
        raise ValueError


def insertNegativeFrequencies(freqSignal, even):
    """To be used in conjunction with getFrequencyValues
    Inserts all negative frequency values under the
    assumption of conjugate symmetry, i.e. a real impulse response.
    Parameter even: boolean indicating if an even or odd number
    of bins is desired. This must correspond to numFreq value
    set in getFrequencyValues
    
    Frequencies must be on axis=0"""
    if even:
        return np.concatenate(
            (freqSignal, np.flip(freqSignal[1:-1, :, :].conj(), axis=0)), axis=0
        )
    else:
        raise NotImplementedError

def firFromFreqsWindow(freqFilter, irLen, twoSided=True):
    """Use this over the other window methods,
    as they might be wrong. freqFilter is with both positive
    and negative frequencies.

    Makes FIR filter from frequency values.
    Works only for odd impulse response lengths
    Uses hamming window"""
    assert irLen % 1 == 0
    assert freqFilter.shape[0] % 2 == 0
    if twoSided:
        #halfLen = irLen // 2
        midPoint = freqFilter.shape[0] // 2
        timeFilter = np.real(fdf.ifftWithTranspose(freqFilter))
        #truncError = calcTruncationError(fullTimeFilter, irLen, twoSided)
        timeFilter = np.concatenate((timeFilter[...,-midPoint:], timeFilter[...,:midPoint]), axis=-1)

        #truncFilter = timeFilter[..., midPoint-halfLen:midPoint+halfLen+1]
        truncFilter, truncError = truncateFilter(timeFilter, irLen, True)
        
        # timeFilter = np.concatenate(
        #     (fullTimeFilter[..., -halfLen:], fullTimeFilter[..., : halfLen + 1]), axis=-1
        # )


        # ONLY TEMPORARILY COMMENTED. THE WINDOW CODE HERE REALLY WORKS. 
        # ADD A BOOLEAN ARGUMENT INSTEAD
        #truncFilter = truncFilter * signal.windows.hamming(irLen).reshape(
        #    (1,) * (truncFilter.ndim - 1) + truncFilter.shape[-1:]
        #)
        
    else:
        raise NotImplementedError

    return truncFilter, truncError

def truncateFilter(ir, irLen, twoSided):
    if twoSided:
        assert irLen % 2 == 1
        assert ir.shape[-1] % 2 == 0
        halfLen = irLen // 2
        midPoint = ir.shape[-1] // 2
        truncFilter = ir[..., midPoint-halfLen:midPoint+halfLen+1]

        truncPower = np.sum(ir[...,:midPoint-halfLen]**2) + np.sum(ir[...,midPoint+halfLen+1:]**2)
        totalPower = np.sum(ir**2)
        relTruncError = 10 * np.log10(truncPower / totalPower)
    else:
        raise NotImplementedError
    return truncFilter, relTruncError

    

# def firFromFreqsWindow_works(freqFilter, irLen, twoSided=True):
#     """Use this over the other window methods,
#     as they might be wrong. freqFilter is with both positive
#     and negative frequencies.

#     Makes FIR filter from frequency values.
#     Works only for odd impulse response lengths
#     Uses hamming window"""
#     assert irLen % 1 == 0
#     if twoSided:
#         halfLen = irLen // 2
#         midPoint = freqFilter.shape[0] // 2

#         fullTimeFilter = np.real(fdf.ifftWithTranspose(freqFilter))
#         timeFilter = np.concatenate(
#             (fullTimeFilter[..., -halfLen:], fullTimeFilter[..., : halfLen + 1]), axis=-1
#         )
#         timeFilter = timeFilter * signal.windows.hamming(irLen).reshape(
#             (1,) * (timeFilter.ndim - 1) + timeFilter.shape[-1:]
#         )
        
#     else:
#         raise NotImplementedError

#     return timeFilter, truncError


def calcTruncationError(ir, irLen, twoSided=True):
    if twoSided:
        assert irLen % 2 == 1
        halfLen = irLen // 2
        midPoint = ir.shape[-1]
        truncPower = np.sum(ir[...,:midPoint-halfLen]**2) + np.sum(ir[...,midPoint+halfLen:]**2)
        totalPower = np.sum(ir**2)
        relTruncError = 10 * np.log10(truncPower / totalPower)
    else:
        raise NotImplementedError
    return relTruncError
    

def minTruncatedLength(ir, twosided=True, maxRelTruncError=1e-3):
    """Calculates the minimum length you can truncate a filter to.
    ir has shape (..., irLength)
    if twosided, the ir will be assumed centered in the middle.
    The filter can be multidimensional, the minimum length will
    be calculated independently for all impulse responses, and
    the longest length chosen. The relative error is how much of the
    power of the impulse response that is lost by truncating."""
    irLen = ir.shape[-1]
    irShape = ir.shape[:-1]

    totalEnergy = np.sum(ir ** 2, axis=-1)
    energyNeeded = (1 - maxRelTruncError) * totalEnergy

    if twosided:
        centerIdx = irLen // 2
        casualSum = np.cumsum(ir[..., centerIdx:] ** 2, axis=-1)
        noncausalSum = np.cumsum(np.flip(ir[..., :centerIdx] ** 2, axis=-1), axis=-1)
        energySum = casualSum
        energySum[..., 1:] += noncausalSum
    else:
        energySum = np.cumsum(ir ** 2, axis=-1)

    enoughEnergy = energySum > energyNeeded[..., None]
    truncIndices = np.zeros(irShape, dtype=int)
    for indices in it.product(*[range(dimSize) for dimSize in irShape]):
        truncIndices[indices] = np.min(
            np.nonzero(enoughEnergy[indices + (slice(None),)])[0]
        )

    reqFilterLength = np.max(truncIndices)
    if twosided:
        reqFilterLength = 2 * reqFilterLength + 1
    return reqFilterLength










# ==============================================================================
# GENERATES TIME DOMAIN FILTER FROM FREQUENCY DOMAIN FILTERS
# SHOULD BE CHECKED OUT/TESTED BEFORE USED. I DONT FULLY TRUST THEM
# def tdFilterFromFreq(
#     freqSamples, irLen, posFreqOnly=True, method="window", window="hamming"
# ):
#     numFreqs = freqSamples.shape[-1]
#     if posFreqOnly:
#         if numFreqs % 2 == 0:
#             raise NotImplementedError
#             freqSamples = np.concatenate(
#                 (freqSamples, np.flip(freqSamples, axis=-1)), axis=-1
#             )
#         else:
#             freqSamples = np.concatenate(
#                 (freqSamples, np.flip(freqSamples[:, :, 1:], axis=-1)), axis=-1
#             )

#     if method == "window":
#         assert irLen < freqSamples.shape[-1]
#         tdFilter = tdFilterWindow(freqSamples, irLen, window)
#     elif method == "minphase":
#         assert irLen < 2 * freqSamples.shape[-1] - 1
#         tdFilter = tdFilterMinphase(freqSamples, irLen, window)

#     return tdFilter


# def tdFilterWindow(freqSamples, irLen, window):
#     ir = np.fft.ifft(freqSamples, axis=-1)
#     ir = np.concatenate(
#         (ir[:, :, -irLen // 2 + 1 :], ir[:, :, 0 : irLen // 2 + 1]), axis=-1
#     )

#     if np.abs(np.imag(ir)).max() / np.abs(np.real(ir)).max() < 10 ** (-7):
#         ir = np.real(ir)

#     ir *= signal.get_window(window, (irLen), fftbins=False)[None, None, :]
#     return ir


# def tdFilterMinphase(freqSamples, irLen, window):
#     ir = np.fft.ifft(freqSamples ** 2, axis=-1)
#     ir = np.concatenate((ir[:, :, -irLen + 1 :], ir[:, :, 0:irLen]), axis=-1)
#     ir = ir * (
#         signal.get_window(window, (irLen * 2 - 1), fftbins=False)[None, None, :] ** 2
#     )

#     if np.abs(np.imag(ir)).max() / np.abs(np.real(ir)).max() < 10 ** (-7):
#         ir = np.real(ir)

#     tdFilter = np.zeros((ir.shape[0], ir.shape[1], irLen))
#     for i in range(ir.shape[0]):
#         for j in range(ir.shape[1]):
#             tdFilter[i, j, :] = signal.minimum_phase(ir[i, j, :])
#     return tdFilter

