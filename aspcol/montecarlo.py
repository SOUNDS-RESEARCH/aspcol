import numpy as np
import ancsim.utilities as util



def integrateMp(
    func, pointGenerator, totNumSamples, totalVolume, numPerIter=5, verbose=False
):
    """Uses mcIntegrate but paralellized"""
    import dill
    dill.settings["recurse"] = True
    import pathos.multiprocessing as mp
    from pathos.helpers import freeze_support


    ncpu = mp.cpu_count()
    # ncpu = 2
    integrationSamples = int(np.ceil(totNumSamples / ncpu))
    intArgs = [
        [func for _ in range(ncpu)],
        [pointGenerator for _ in range(ncpu)],
        [integrationSamples for _ in range(ncpu)],
        [totalVolume for _ in range(ncpu)],
        [numPerIter for _ in range(ncpu)],
        [verbose for _ in range(ncpu)],
    ]

    with mp.ProcessingPool(ncpu) as p:
        integral = p.map(integrate, *intArgs)

    integral = np.mean(np.stack(integral, axis=-1), axis=-1)
    return integral

def integrate_fast(
    func, pointGenerator, totNumSamples, totalVolume, *args, numPerIter=50,
):
    """identical to integrate, but is meant to be simpler."""
    numBlocks = int(np.ceil(totNumSamples / numPerIter))
    testVal = func(pointGenerator(1), *args)
    outDims = np.squeeze(testVal, axis=-1).shape
    integralVal = np.zeros(outDims, dtype=testVal.dtype)

    for i in range(numBlocks):
        fVals = func(pointGenerator(numPerIter), *args)
        integralVal = (integralVal * i + np.mean(fVals, axis=-1)) / (i + 1)
    integralVal *= totalVolume
    return integralVal



def integrate(
    func, pointGenerator, totNumSamples, totalVolume, numPerIter=50, verbose=False, *args
):
    """pointGenerator should return np array, [numPoints, numSpatialDimensions]
    func should return np array [funcDims, numPoints],
    where funcDims can be any number of dimensions (in a multidimensional array sense)"""
    print("Starting MC Integration")
    samplesPerIter = numPerIter
    numBlocks = int(np.ceil(totNumSamples / samplesPerIter))
    outDims = np.squeeze(func(pointGenerator(1), *args), axis=-1).shape
    integralVal = np.zeros(outDims)

    for i in range(numBlocks):
        points = pointGenerator(samplesPerIter)
        fVals = func(points, *args)

        newIntVal = (integralVal * i + np.mean(fVals, axis=-1)) / (i + 1)
        print("Block ", i)
        if verbose:
            diagnostics(newIntVal, integralVal, i)

        integralVal = newIntVal
    integralVal *= totalVolume
    print("Finished!!")
    return integralVal


def diagnostics(newVal, oldVal, blockIdx):
    # print("Block ", blockIdx)
    change = newVal - oldVal
    relChange = change / oldVal

    meansquarechange = np.mean(np.square(newVal - oldVal))
    meanabschange = np.mean(np.abs(newVal - oldVal))
    print("Mean Square Change: " + "{:2.12E}".format(meansquarechange))
    print("Mean Abs Change: " + "{:2.12E}".format(meanabschange))

    maxAbsChange = np.max(np.abs(change))
    maxAbsIdx = np.argmax(np.abs(change))
    maxAbsChangeVal = np.abs(newVal.flatten()[maxAbsIdx])
    maxAbsChangeRel = np.abs(relChange.flatten()[maxAbsIdx])
    print(
        "Max Abs change: "
        + "{:2.12E}".format(maxAbsChange)
        + "     Its Value: "
        + "{:2.12E}".format(maxAbsChangeVal)
        + "     rel change: "
        + "{:2.12E}".format(maxAbsChangeRel)
    )

    maxRelChange = np.max(np.abs(relChange))
    maxRelIdx = np.argmax(np.abs(relChange))
    maxRelChangeVal = np.abs(newVal.flatten()[maxRelIdx])
    maxRelChangeAbs = np.abs(change.flatten()[maxRelIdx])
    print(
        "Max rel change: "
        + "{:2.12E}".format(maxRelChange)
        + "     Its Value: "
        + "{:2.12E}".format(maxRelChangeVal)
        + "     abs change "
        + "{:2.12E}".format(maxRelChangeAbs)
    )

    avgRelChange = np.mean(np.abs(relChange))
    avgAbsChange = np.mean(np.abs(change))
    print(
        "Avg rel change: "
        + "{:2.12E}".format(avgRelChange)
        + "     Avg abs change: "
        + "{:2.12E}".format(avgAbsChange)
    )

    maxVal = np.max(np.abs(newVal))
    print("Max Value: " + "{:2.12E}".format(maxVal))
    print()
    return maxRelChange
