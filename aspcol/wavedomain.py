import numpy as np

import matplotlib.pyplot as plt
import ancsim.signal.filterdesign as fd
import ancsim.soundfield.roomimpulseresponse as rir


def waveTransformCircularMicArray(micPos, numHarmonics):
    """As described in SPM paper. Inverse transform is pseudoinverse."""
    numMics = micPos.shape[0]

    angles = np.arange(numMics) * 2 * np.pi / numMics
    calcAngles = np.arctan(micPos[:, 1] / micPos[:, 0])
    assert np.allclose(angles % np.pi, (calcAngles + 1e-6) % np.pi) - 1e-6

    transformMatrix = (
        np.exp(
            -1j * angles[None, :] * np.arange(-numHarmonics, numHarmonics + 1)[:, None]
        )
        / numMics
    )
    transformMatrix = transformMatrix[None, :, :]
    return transformMatrix, np.linalg.pinv(transformMatrix)


def waveTransformSpeakerArray(speakerPos, micPos, numHarmonics, numFreq, samplerate, c):
    """Assumes all speakers are placed on z=0"""
    assert numFreq % 2 == 0
    numMics = micPos.shape[0]

    freqs = fd.getFrequencyValues(numFreq, samplerate)

    G = rir.tfPointSource3d(speakerPos, micPos, freqs)
    T2, _ = waveTransformCircularMicArray(micPos, numHarmonics)
    T1 = T2 @ np.transpose(G, (0, 2, 1))
    T1 /= numMics
    return T1, np.linalg.pinv(T1)
