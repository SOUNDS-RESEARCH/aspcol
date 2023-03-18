import numpy as np
import datetime
import time
from functools import wraps
import numpy as np
import copy

import aspcore.filterclasses as fc



def power_of_filtered_signal(src, ir, num_samples):
    """
    Returns the average power of the signal after 
        propagating through an impulse response
        rir should be (num_sources, num_recievers, rir_len)

    For stochastic signals, it is an estimate made from
        num_samples. 
    For periodic signals, set num_samples to the period
        for the true average power (not an estimate)

    Returns a non-negative array of shape (num_recievers,) 
        which is the average power for each channel 
    """
    assert ir.ndim == 3
    ir_len = ir.shape[-1]
    src_copy = copy.deepcopy(src)
    in_sig = src_copy.get_samples(num_samples+ir_len-1)

    filt = fc.create_filter(ir)
    filt_sig = filt.process(in_sig)
    filt_sig = filt_sig[...,ir_len-1:]
    avg_pow = np.mean(filt_sig**2, axis=-1)
    return avg_pow

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

def cart2spherical(cart_coord):
    """cartCoord is shape = (numPoints, 3)
        returns (r, angle), defined as in spherical2cart"""
    r = np.linalg.norm(cart_coord, axis=1)
    raise NotImplementedError

def spherical2cart(r, angle):
    """r is shape (numPoints, 1) or (numPoints)
        angle is shape (numPoints, 2)
        angle[:,0] is theta
        angle[:,1] is phi
        theta is normal polar coordinate angle, 0 is x direction, pi/2 is y direction
        phi is azimuth, 0 is z direction, pi is negative z direction"""
    num_points = r.shape[0]
    cart_coord = np.zeros((num_points,3))
    cart_coord[:,0] = np.squeeze(r) * np.cos(angle[:,0]) * np.sin(angle[:,1])
    cart_coord[:,1] = np.squeeze(r) * np.sin(angle[:,0]) * np.sin(angle[:,1])
    cart_coord[:,2] = np.squeeze(r) * np.cos(angle[:,1])
    return cart_coord

def get_smallest_coprime(N):
    assert N > 2 #don't have to deal with 1 and 2 at this point
    for i in range(2,N):
        if np.gcd(i,N):
            return i

def next_divisible(divisor, min_value):
    """Gives the smallest integer divisible by divisor, 
        that is strictly larger than minValue"""
    rem = (min_value + divisor) % divisor
    return min_value + divisor - rem

def db2mag(db):
    return 10 ** (db / 20)

def mag2db(amp):
    return 20 * np.log10(amp)

def db2pow(db):
    return 10 ** (db / 10)

def pow2db(power):
    return 10 * np.log10(power)


def simplify_ratio(a : int, b : int):
    """
    simplifies the ratio a/b into the simplest possible
    ratio where both numerator and denominator are integers
    """
    d = np.gcd(a,b)
    while d != 1:
        a = a // d
        b = b // d
        d = np.gcd(a,b)
    return a,b


def measure(name):
    """
        edited, originally from JBirdVegas, stackoverflow
    """
    def measure_internal(func):
        @wraps(func)
        def _time_it(*args, **kwargs):
            start = int(round(time.time() * 1000))
            try:
                return func(*args, **kwargs)
            finally:
                end_ = int(round(time.time() * 1000)) - start
                print(name + f" execution time: {end_ if end_ > 0 else 0} ms")
        return _time_it
    return measure_internal




def block_process_idxs(num_samples : int, block_size : int, overlap : int, start_idx=0):
    """Yields the starting index for each block, for block processing a signal

    Parameters
    ----------
    num_samples : int
        total number of samples for the signal that should be processed
    block_size : int
        the size of each of the blocks
    overlap : int
        the amount each block should be overlapped at the output
    start_idx : int
        can be supplied if the processing should start at another place
        of the original signal than idx = 0

    Yields
    -------
    idx : int
        can be used to get your block as signal[..., idx:idx + block_size]
    """
    assert 0 <= overlap < block_size
    assert 0 <= start_idx < num_samples
    hop = block_size - overlap
    #left_in_block = block_size - start_idx

    #indices = []
    
    sample_counter = start_idx
    while sample_counter+block_size < num_samples:
        #block_len = min(num_samples - sample_counter, left_in_block)
        yield sample_counter 
        #indices.append(sample_counter)


        sample_counter += hop

        #


# def calcBlockSizes(numSamples, startIdx, blockSize):
#     leftInBlock = blockSize - startIdx
#     sampleCounter = 0
#     blockSizes = []
#     while sampleCounter < numSamples:
#         blockLen = np.min((numSamples - sampleCounter, leftInBlock))
#         blockSizes.append(blockLen)
#         sampleCounter += blockLen
#         leftInBlock -= blockLen
#         if leftInBlock == 0:
#             leftInBlock = blockSize
#     return blockSizes




class PhaseCounter:
    """
    An index counter to keep track of non-overlapping continous phases
    
    Example:
    A processor needs the first 2000 samples for an initialization, 
    then must wait 5000 samples before beginning the real processing step.
    The class can then be used by providing
    phase_def = {
        'init' : 2000,
        'wait' : 5000,
        'process' : np.inf
    }
    and then checking if phase_counter.phase == 'init'
    or if phase_counter.current_phase_is('init'):
    
    The number is how many samples that each phase should be
    The first phase will start at sample 0.

    np.inf represents an infinite length
    This should naturally only be used for the last phase
    If all phases has finished, the phase will be None. 

    first_sample will be True on the first sample of each phase,
    allowing running one-time functions in each phase

    Extended implementation to blocksize != 1 can be done later
    """
    def __init__(self, phase_lengths, verbose=False):
        assert isinstance(phase_lengths, dict)
        self.phase_lengths = phase_lengths
        self.verbose = verbose
        self.phase = None
        self.first_sample = True
        

        #phase_lengths = {name : length for name, length in self.phase_lengths.items() if length != 0}
        #phase_lengths = {name : length for name, length in self.phase_lengths.items()}
        
        #phase_idxs = [i for i in self.phase_lengths.values() if i != 0]
        self.phase_lengths = {name : i if i >= 0 else np.inf for name, i in self.phase_lengths.items()}
        #assert all([i != 0 for i in p_len])
        self.start_idxs = np.cumsum(list(self.phase_lengths.values())).tolist()
        self.start_idxs = [i if np.isinf(i) else int(i) for i in self.start_idxs]
        self.start_idxs.insert(0,0)

        self.phase_names = list(self.phase_lengths.keys())
        if self.start_idxs[-1] < np.inf:
            self.phase_names.append(None)
        else:
            self.start_idxs.pop()

        self.start_idxs = {phase_name:start_idx for phase_name, start_idx in zip(self.phase_names, self.start_idxs)}

        self._phase_names = [phase_name for phase_name, phase_len in self.phase_lengths.items() if phase_len > 0]
        self._start_idxs = [start_idx for start_idx, phase_len in zip(self.start_idxs.values(), self.phase_lengths.values()) if phase_len > 0]

        self.idx = 0
        self.next_phase()

    def next_phase(self):
        if self.verbose:
            print(f"Changed phase from {self.phase}")
            
        self.phase = self._phase_names.pop(0)
        self._start_idxs.pop(0)
        if len(self._start_idxs) == 0:
            self._start_idxs.append(np.inf)
        self.first_sample = True
        
        if self.verbose:
            print(f"to {self.phase}")

    def progress(self):
        self.idx += 1
        if self.idx >= self._start_idxs[0]:
            self.next_phase()
        else:
            self.first_sample = False

    def current_phase_is(self, phase_name):
        return self.phase == phase_name



class EventCounter:
    """
    An index counter to keep track of events that should 
    only happen every x samples

    event_def is a dictionary with all event
    each entry is 'event_name' : (frequency, offset)

    Example:
    event_counter = EventCounter({'event_1' : (256,0), 'event_2' : (1,0), 'event_3' : (1024,256)})
    event_2 will happen every sample, event_1 every 256 samples
    First at sample 256 all three events will happen simultaneouly. 

    To be used as:
    if 'event_name' in event_counter.event:
        do_thing()

    """
    def __init__(self, event_def):
        self.event_def = event_def
        self.event = []

        self.freq = {name : freq for name, (freq, offset) in event_def.items()}
        self.offset = {name : offset for name, (freq, offset) in event_def.items()}

        self.idx = 0

    def add_event(self, name, freq, offset):
        self.event_def[name] = (freq, offset)

    def check_events(self):
        self.event = []
        for name, (freq, offset) in self.event_def.items():
            if (self.idx - offset) % freq == 0:
                self.event.append(name)

    def progress(self):
        self.idx += 1 
        self.check_events()
