import ancsim.integration.montecarlo as mc
import ancsim.integration.pointgenerator as gen
import numpy as np

# def test_multiprocessing():
#     vol = 1
#     def f(r):
#         return np.sum(r, axis=0)[None,:]
#     pointGen = gen.block((1,1), 1)

#     np.random.seed(0)
#     val = mc.integrate(f, pointGen, 1000, vol)

#     np.random.seed(0)
#     valmp = mc.integrateMp(f, pointGen, 1000, vol)
#     assert valmp == val

# def test_constant_scalar_integral():
#     def f(r):
#         return np.ones((1,1))

#     def pointGen(numSamples):
#         points = np.random.rand(10, numSamples)
#         return points
#     vol = 1
#     val = mc.integrate(f, pointGen, 1000, vol)
#     assert val == 1


def test_constant_scalar_integral():
    rDim = 10
    domainLength = 2

    def f(r):
        return np.ones((1, r.shape[-1]))

    def pointGen(numSamples):
        points = np.random.rand(rDim, numSamples) * domainLength
        return points

    vol = domainLength ** rDim
    val = mc.integrate(f, pointGen, 1000, vol)
    assert np.array_equal(val, np.array([vol]))


# def test_circle_centerofgravity():
#     print("Test circle center of gravity: ")

#     def f(r):
#         return r

#     radius = 10

#     def pointGen(numSamples):
#         r = np.sqrt(np.random.rand(numSamples)) * radius
#         angle = np.random.rand(numSamples) * 2 * np.pi
#         x = r * np.cos(angle)
#         y = r * np.sin(angle)
#         return np.stack((x, y), axis=0)

#     vol = radius ** 2 * np.pi
#     for numSamples in [1e2, 1e3, 1e4]:
#         val = mc.integrate(f, pointGen, numSamples, vol)
#         print("Actual: ", val, " Expected: ", 0, "Number of Samples: ", numSamples)
#     print()
