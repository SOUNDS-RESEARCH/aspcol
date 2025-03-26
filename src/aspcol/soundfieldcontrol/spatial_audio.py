import numpy as np
import aspcore.filter as fc

# ================== SPATIAL AUDIO ===============================
def pressure_matching_sa(Hv, Hc):
    """Frequency domain pressure matching to generate loudspeaker signals from virtual source signals

    v is the virtual source sound pressure (num_freq, num_virtual_src, 1)
    
    Parameters
    ----------
    Hv : ndarray of shape (num_freq, num_mic, num_virtual_src)
        transfer functions from virtual sources to control points 
    Hc : ndarray of shape (num_freq, num_mic, num_ls)
        transfer functions from loudspeakers to control points 

    Returns
    -------
    beamformer w : ndarray of shape (num_freq, num_ls, num_virtual_source) 
        which should be applied to the virutal source sound pressure 
        as w @ v, where v : (num_freq, num_virtual_src, 1)

    Notes
    -----
    The beamformer is calculated as w = (H_c^H H_c)^{-1} H_c^H H_v. Definition can be found in (15) in [brunnstromSound2023]

    References
    ----------
    [brunnstromSound2023] J. Brunnström, T. van Waterschoot, and M. Moonen, “Sound zone control for arbitrary sound field reproduction methods,” in European Signal Processing Conference (EUSIPCO), Helsinki, Finland, Sep. 2023.
    """
    w = np.linalg.solve(np.moveaxis(Hc.conj(),-1,-2) @ Hc, np.moveaxis(Hc.conj(),-1,-2) @ Hv)
    return w

class PressureMatchingWOLA:
    """
    Sound field reproduction with pressure matching in the WOLA domain
    
    """
    def __init__(self, audio_src, Hv, Hc, block_size):
        self.ctrl_freq = self.update(Hv, Hc)
        self.audio_src = audio_src
        self.block_size = block_size

        self.wola = fc.WOLA(self.audio_src.num_channels, 1, 2*self.block_size, block_size)

        self.num_real_freq = Hv.shape[0]
        assert Hc.shape[0] == self.num_real_freq
        
    def get_samples(self):
        audio = self.audio_src.get_samples(self.block_size)
        self.wola.analysis(audio)
        ls_sig = self.ctrl_freq @ np.moveaxis(self.wola.spectrum, -1, 0)
        return ls_sig

    def update(self, Hv, Hc):
        self.ctrl_freq = pressure_matching_sa(Hv, Hc)
