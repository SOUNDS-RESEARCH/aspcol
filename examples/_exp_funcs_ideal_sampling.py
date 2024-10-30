import numpy as np
import pathlib
import json
import scipy.signal as spsig

import aspcore.pseq as pseq
import aspcore.filterdesign as fd
import aspcore.fouriertransform as ft

from aspsim.simulator import SimulatorSetup

import aspsim.room.trajectory as tr
import aspsim.room.generatepoints as gp
import aspsim.room.region as reg
import aspsim.signal.sources as sources
import aspsim.diagnostics.diagnostics as dg
import aspsim.saveloadsession as sls


import dynamicsfutilities as dsu


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def generate_constant_speed_trajectory(position, velocity, samplerate, target_speed, num_samples, tolerance = 0.05):
    """
    
    target speed : int
        target speed in meters per second

    """
    all_pos = np.zeros((num_samples, 3))
    all_pos[0,:] = position(0)

    target_speed_per_sample = target_speed / samplerate

    t = 0
    for n in range(1, num_samples):
        last_pos = all_pos[n-1,:]
        v = np.linalg.norm(velocity(t))
        timestep = target_speed_per_sample / v

        candidate_pos = position(t+timestep)
        speed = np.linalg.norm(candidate_pos - last_pos)

        if np.abs(speed - target_speed_per_sample) / target_speed_per_sample > tolerance:
            if speed > target_speed_per_sample:
                t_low = 0
                t_high = timestep
            else:
                t_low = timestep
                t_high = 2 * timestep
                while np.linalg.norm(position(t+ t_high) - last_pos) <= target_speed_per_sample:
                    t_high = 2* t_high
            timestep = pos_bifurcation(t_low, t_high, t, position, last_pos, target_speed_per_sample, tolerance)
            candidate_pos = position(t+timestep)


        t += timestep
        all_pos[n,:] = candidate_pos

    return all_pos


def pos_bifurcation(low, high, offset, pos, previous_pos, desired_val, tolerance):
    speed = -1000

    while np.abs(speed - desired_val) / desired_val > tolerance:
        t = (low + high) / 2
        p = pos(offset + t)

        speed = np.linalg.norm(p - previous_pos) 

        if speed <= desired_val:
            low = t
        else:
            high = t
    return t

class LissajousTrajectoryConstantSpeed():
    def __init__(self, amplitude, freq, center, samplerate, speed_factor, num_samples):
        """pos_func is a function, which takes a time_index in samples and outputs a position"""
        self.amplitude = amplitude
        self.freq = freq
        assert self.freq.shape == (1,3)
        #self.period_len = period_len
        self.center = center
        assert self.center.shape == (1,3)
        self.phase_offset = np.array([[0, np.pi/2, np.pi/2]])

        self.samplerate = samplerate
        self.speed_factor = speed_factor
        self.num_samples = num_samples
        #self.pos = np.full((1,3), np.nan)

        self.all_pos = generate_constant_speed_trajectory(self.r, self.velocity, self.samplerate, speed_factor, num_samples)

    def r(self, t):
        return self.center + self.amplitude * np.cos(2 * np.pi * t * self.freq / self.samplerate + self.phase_offset)
    
    def velocity(self, t): 
        return -self.amplitude * 2 * np.pi * self.freq * np.sin(2 * np.pi * t * self.freq / self.samplerate + self.phase_offset) / self.samplerate

    def current_pos(self, time_idx):
        return self.all_pos[time_idx:time_idx+1,:]
            

    # def gen_pos(self, num_samples):
    #     all_pos = np.zeros((num_samples, 3))
    #     t = 0
    #     for n in range(num_samples):
    #         all_pos[n,:] = self.r(t)

    #         v = np.linalg.norm(self.velocity(t))
    #         t += 1 / (self.samplerate * v)
                    
    #     return all_pos

    def plot(self, ax, symbol, name):
        pass

class LissajousTrajectory():
    def __init__(self, amplitude, freq, center):
        """pos_func is a function, which takes a time_index in samples and outputs a position"""
        self.amplitude = amplitude
        self.freq = freq
        assert self.freq.shape == (1,3)
        #self.period_len = period_len
        self.center = center
        assert self.center.shape == (1,3)
        self.phase_offset = np.array([[0, np.pi/2, np.pi/2]])
        #self.pos = np.full((1,3), np.nan)

    def current_pos(self, time_idx):
            return self.center + self.amplitude * np.cos(2 * np.pi * time_idx * self.freq + self.phase_offset)

        #return self.pos_func(time_idx)

    def plot(self, ax, symbol, name):
        pass



def generate_signals_3d(sr, rt60=0.1, num_mic = 20, seq_len_frac_of_sec = 2):
    rng = np.random.default_rng(10)
    side_len = 0.3
    #num_eval = 200
    num_mic = num_mic
    seq_len = sr // seq_len_frac_of_sec

    center = np.zeros((1,3))
    
    #eval_region = reg.Disc(radius, (0,0), (0.1, 0.1))
    eval_region = reg.Cuboid((side_len, side_len, side_len), (0,0,0), (0.03, 0.03, 0.03))
    pos_eval = eval_region.equally_spaced_points()

    high_res = 0.002
    low_res = 0.05
    image_region = reg.Rectangle((side_len, side_len), (0,0,0), (low_res, low_res))
    pos_image = image_region.equally_spaced_points()

    #pos_mic = np.zeros((num_mic, 3))
    pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))
    pos_src = np.array([[3,0,0]])

    setup = SimulatorSetup(pathlib.Path(__file__).parent.joinpath("figs"))
    setup.sim_info.samplerate = sr
    

    speed_factor = 0.5
    tot_trajectory_samples = num_mic * seq_len
    freq_factors = np.array([[3,4,2]])
    #trajectory = LissajousTrajectory(side_len/2, speed_factor * freq_factors / sr, center)
    #traj_pos = np.concatenate([trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0)

    trajectory = LissajousTrajectoryConstantSpeed(side_len/2, speed_factor * freq_factors / sr, center, sr, speed_factor, tot_trajectory_samples)
    traj_pos = np.concatenate([trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0)

    speed = np.linalg.norm(traj_pos[1:,:] - traj_pos[:-1,:], axis=-1) * sr
    #speed2 = np.linalg.norm(traj_pos2[1:,:] - traj_pos2[:-1,:], axis=-1) * sr


    initial_delay = seq_len
    post_delay = 0

    setup.sim_info.tot_samples = initial_delay + seq_len + post_delay
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 = rt60
    setup.sim_info.max_room_ir_length = seq_len
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // seq_len_frac_of_sec
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "tikz"
    setup.sim_info.start_sources_before_0 = True

    seq_len = setup.sim_info.max_room_ir_length
    sequence = pseq.create_pseq(seq_len)
    sequence_src = sources.Sequence(sequence)

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_mics("image", pos_image)
    setup.add_free_source("src", pos_src, sequence_src)
    setup.add_mics("mic_dynamic", traj_pos)

    sim = setup.create_simulator()
    run_and_save(sim)
    with open(sim.folder_path.joinpath("extra_parameters.json"), "w") as f:
        json.dump({"seq_len" : seq_len, 
                    "initial_delay" : initial_delay,
                   "post_delay" : post_delay,
                   "max_sweep_freq" : sr // 2,
                    "center" : center.tolist(), 
                    "downsampling_factor" : 1,
                    "freq_factors" : freq_factors.tolist(),
                    "speed_factor" : speed_factor,
                    "speed min" : np.min(speed),
                    "speed max" : np.max(speed),
                    "speed mean" : np.mean(speed),
                    } ,f)
    return sim.folder_path




def generate_signals_2d(sr):
    rng = np.random.default_rng(10)
    side_len = 0.3
    #num_eval = 200
    num_mic = 12
    seq_len = sr // 4

    center = np.zeros((1,3))
    
    #eval_region = reg.Disc(radius, (0,0), (0.1, 0.1))
    eval_region = reg.Cuboid((side_len, side_len, 0.01), (0,0,0), (0.005, 0.005, 0.01))

    pos_mic = np.zeros((num_mic, 3))
    pos_mic[:,:2] = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 2))
    pos_eval = eval_region.equally_spaced_points()

    pos_src = np.array([[3,0,0]])

    setup = SimulatorSetup(pathlib.Path(__file__).parent.joinpath("figs"))
    setup.sim_info.samplerate = sr
    

    speed_factor = 0.5
    tot_trajectory_samples = num_mic * seq_len
    freq_factors = np.array([[3,4,0]])
    #trajectory = LissajousTrajectory(side_len/2, speed_factor * freq_factors / sr, center)
    #traj_pos = np.concatenate([trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0)

    trajectory = LissajousTrajectoryConstantSpeed(side_len/2, speed_factor * freq_factors / sr, center, sr, speed_factor, tot_trajectory_samples)
    traj_pos = np.concatenate([trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0)

    speed = np.linalg.norm(traj_pos[1:,:] - traj_pos[:-1,:], axis=-1) * sr
    #speed2 = np.linalg.norm(traj_pos2[1:,:] - traj_pos2[:-1,:], axis=-1) * sr


    initial_delay = seq_len
    post_delay = 0

    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 =  0.1
    setup.sim_info.max_room_ir_length = seq_len
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "tikz"

    seq_len = setup.sim_info.max_room_ir_length
    sequence = pseq.create_pseq(seq_len)
    sequence_src = sources.Sequence(sequence)

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_free_source("src", pos_src, sequence_src)
    setup.add_mics("mic_dynamic", traj_pos)

    sim = setup.create_simulator()
    run_and_save(sim)
    with open(sim.folder_path.joinpath("extra_parameters.json"), "w") as f:
        json.dump({"seq_len" : seq_len, 
                    "initial_delay" : initial_delay,
                   "post_delay" : post_delay,
                   "max_sweep_freq" : sr // 2,
                    "center" : center.tolist(), 
                    "downsampling_factor" : 1,
                    "freq_factors" : freq_factors.tolist(),
                    "speed_factor" : speed_factor,
                    "speed min" : np.min(speed),
                    "speed max" : np.max(speed),
                    "speed mean" : np.mean(speed),
                    } ,f)
    return sim.folder_path


def generate_signals_circular(sr, num_ls, num_eff_mics=None):
    radius = 0.5
    if num_eff_mics is None:
        num_mic = 8
    else:
        num_mic = num_eff_mics
    mic_noise_power = 1e-12 #currently does nothing

    rng = np.random.default_rng(10)
    num_eval = 200

    pos_mic = gp.equiangular_circle(num_mic, radius, z=0, start_angle=rng.uniform(0,1/num_mic))
    pos_eval = gp.equiangular_circle(num_eval, radius, z=0)

    #pos_src_prototype = gp.equiangular_circle(num_ls, (30, 40), z=0)
    #pos_src = pos_src_prototype[:num_ls, :]
    assert num_ls == 1
    pos_src = np.array([[40,0,0]])

    setup = SimulatorSetup(pathlib.Path(__file__).parent.joinpath("figs"))
    setup.sim_info.samplerate = sr
    
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [200, 200, 200]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 = 0
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 80
    setup.sim_info.plot_output = "tikz"
    setup.sim_info.start_sources_before_0 = True

    seq_len = num_ls * setup.sim_info.max_room_ir_length
    seq = pseq.create_pseq(seq_len)
    shifted_seq = pseq.create_shifted_pseq(seq, num_ls, setup.sim_info.max_room_ir_length)
    sequence_src = sources.Sequence(shifted_seq)

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_free_source("src", pos_src, sequence_src)

    initial_delay = 10*seq_len
    post_delay = 0

    if num_eff_mics is None:
        speed_factor = 0.5
        angular_period = dsu.periodic_angular_period(setup.sim_info.c, seq_len, radius, sr, speed_factor)
    else:
        angular_period = num_eff_mics * (seq_len / sr)
    angular_speed = dsu.angular_period_to_angular_speed(angular_period)
    max_freq = dsu.max_frequency_for_angular_speed(angular_speed, setup.sim_info.c, setup.sim_info.samplerate, seq_len, radius)
    
    setup.sim_info.tot_samples = initial_delay + seq_len
    setup.sim_info.export_frequency = setup.sim_info.tot_samples

    tot_trajectory_samples = int(seq_len * ((sr * angular_period) // seq_len)) # use this if using periodic_angular_speed
    trajectory = tr.CircularTrajectory((radius, radius), (0,0,0), 100000, angular_period, sr, start_angle = 0)
    traj_pos = np.concatenate([trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0)
    setup.add_mics("mic_dynamic", traj_pos)

    mic_direction = traj_pos / np.linalg.norm(traj_pos, axis=-1)[:,None]
    setup.add_mics("mic_dynamic_cardioid", traj_pos, directivity_type=tot_trajectory_samples * ["cardioid"], directivity_dir=mic_direction)

    #add_mic_noise(setup, mic_noise_power)
    sim = setup.create_simulator()
    run_and_save(sim)
    with open(sim.folder_path.joinpath("extra_parameters.json"), "w") as f:
        json.dump({"radius" : radius,
                    "seq_len" : seq_len, 
                    "initial_delay" : initial_delay,
                   "post_delay" : post_delay,
                   "max_sweep_freq" : sr // 2,
                   #"max_freq_speed" : max_freq_speed, #this is the desired max freq
                   "max_freq_speed_actual" : max_freq, # this is the max freq after we adjust to make the trajectory periodic
                    "mic noise power" : mic_noise_power, 
                    "angular period" : angular_period, 
                    "angular_speed" : angular_speed, 
                    "center" : trajectory.center, 
                    "downsampling_factor" : 1
                    } ,f)
    return sim.folder_path



def generate_signals_circular_differential_microphone(sr, num_ls, num_eff_mics=None):
    radius = 0.5
    if num_eff_mics is None:
        num_mic = 8
    else:
        num_mic = num_eff_mics
    mic_noise_power = 1e-12 #currently does nothing

    rng = np.random.default_rng(10)
    num_eval = 200

    pos_mic = gp.equiangular_circle(num_mic, radius, z=0, start_angle=rng.uniform(0,1/num_mic))
    pos_eval = gp.equiangular_circle(num_eval, radius, z=0)

    #pos_src_prototype = gp.equiangular_circle(num_ls, (30, 40), z=0)
    #pos_src = pos_src_prototype[:num_ls, :]
    assert num_ls == 1
    pos_src = np.array([[40,0,0]])

    setup = SimulatorSetup(pathlib.Path(__file__).parent.joinpath("figs"))
    setup.sim_info.samplerate = sr
    
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [200, 200, 200]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 = 0
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 80 #40
    setup.sim_info.plot_output = "tikz"
    setup.sim_info.start_sources_before_0 = True

    seq_len = num_ls * setup.sim_info.max_room_ir_length
    seq = pseq.create_pseq(seq_len)
    shifted_seq = pseq.create_shifted_pseq(seq, num_ls, setup.sim_info.max_room_ir_length)
    sequence_src = sources.Sequence(shifted_seq)

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_free_source("src", pos_src, sequence_src)

    initial_delay = 10*seq_len
    post_delay = 0

    if num_eff_mics is None:
        speed_factor = 0.5
        angular_period = dsu.periodic_angular_period(setup.sim_info.c, seq_len, radius, sr, speed_factor)
    else:
        angular_period = num_eff_mics * (seq_len / sr)
    angular_speed = dsu.angular_period_to_angular_speed(angular_period)
    max_freq = dsu.max_frequency_for_angular_speed(angular_speed, setup.sim_info.c, setup.sim_info.samplerate, seq_len, radius)
    
    setup.sim_info.tot_samples = initial_delay + seq_len
    setup.sim_info.export_frequency = setup.sim_info.tot_samples

    tot_trajectory_samples = int(seq_len * ((sr * angular_period) // seq_len)) # use this if using periodic_angular_speed
    trajectory = tr.CircularTrajectory((radius, radius), (0,0,0), 100000, angular_period, sr, start_angle = 0)
    traj_pos = np.concatenate([trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0)
    setup.add_mics("mic_dynamic", traj_pos)

    differential_distance = 0.03
    mic_direction = traj_pos / np.linalg.norm(traj_pos, axis=-1)[:,None]

    pos_diff_mic2 = traj_pos - differential_distance * mic_direction

    setup.add_mics("mic_diff", pos_diff_mic2)

    #add_mic_noise(setup, mic_noise_power)
    sim = setup.create_simulator()
    run_and_save(sim)
    with open(sim.folder_path.joinpath("extra_parameters.json"), "w") as f:
        json.dump({"radius" : radius,
                    "seq_len" : seq_len, 
                    "initial_delay" : initial_delay,
                   "post_delay" : post_delay,
                   "max_sweep_freq" : sr // 2,
                   "differential distance" : differential_distance,
                   #"max_freq_speed" : max_freq_speed, #this is the desired max freq
                   "max_freq_speed_actual" : max_freq, # this is the max freq after we adjust to make the trajectory periodic
                    "mic noise power" : mic_noise_power, 
                    "angular period" : angular_period, 
                    "angular_speed" : angular_speed, 
                    "center" : trajectory.center, 
                    "downsampling_factor" : 1
                    } ,f)
        
    with open(sim.folder_path / "cardioid_direction.npy", "wb") as f:
        np.save(f, mic_direction)
    return sim.folder_path










def setup_circular(num_mic, sr, radius, num_ls):
    rng = np.random.default_rng(10)
    num_eval = 200

    pos_mic = gp.equiangular_circle(num_mic, radius, z=0, start_angle=rng.uniform(0,1/num_mic))
    pos_eval = gp.equiangular_circle(num_eval, radius, z=0)

    pos_src_prototype = np.array([[3,0,0], [2,1,0], [2, -1, 0]])
    pos_src = pos_src_prototype[:num_ls, :]

    setup = SimulatorSetup(pathlib.Path(__file__).parent.joinpath("figs"))
    setup.sim_info.samplerate = sr
    
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 =  0.1
    setup.sim_info.max_room_ir_length = sr // 4
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 40 #40
    setup.sim_info.plot_output = "tikz"
    setup.sim_info.start_sources_before_0 = True

    seq_len = num_ls * setup.sim_info.max_room_ir_length
    #final_samplerate = 2 * max_freq
    #seq_factor = sr // final_samplerate
    #final_seq_len = seq_len // seq_factor
    #print(f"Remainder between high samplerate and final samplerate: {sr % final_samplerate}")
    #print(f"Remainder between long sequence and short sequence: {seq_len % seq_factor}")
    #sequence = pseq.create_pseq(seq_len)
    seq = pseq.create_pseq(seq_len)
    shifted_seq = pseq.create_shifted_pseq(seq, num_ls, setup.sim_info.max_room_ir_length)
    #sequence = resample(pseq.create_pseq(seq_len // seq_factor), seq_factor)[0,:]
    sequence_src = sources.Sequence(shifted_seq)

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_free_source("src", pos_src, sequence_src)
    return setup, seq_len


def add_mic_noise(setup, mic_noise_power):
    for mic_array in setup.arrays.mics():
        if mic_array.name != "eval":
            name = f"{mic_array.name}_noise"
            setup.add_free_source(name, mic_array.pos, sources.WhiteNoiseSource(mic_array.num, mic_noise_power, rng=np.random.default_rng()))

            # == Make sure the noise is only to applied to the correct mics
            for ma in setup.arrays.mics():
                if ma == mic_array:
                    path_type = "isolated"
                else:
                    path_type = "none"
                setup.arrays.path_type[name][ma.name] = path_type
    

def run_and_save(sim):
    for mic_array in sim.arrays.mics():
        sim.diag.add_diagnostic(mic_array.name, dg.RecordSignal(mic_array.name, sim.sim_info, num_channels=mic_array.num, export_func="npz"))

    #sim.diag.add_diagnostic("mic", dg.RecordSignal("mic", sim.sim_info, num_channels=sim.arrays["mic"].num, export_func="npz"))
    #sim.diag.add_diagnostic("eval", dg.RecordSignal("eval", sim.sim_info, num_channels=sim.arrays["eval"].num, export_func="npz"))
    #sim.diag.add_diagnostic("mic_dynamic", dg.RecordSignal("mic_dynamic", sim.sim_info, num_channels=sim.arrays["mic_dynamic"].num, export_func="npz"))
    sim.diag.add_diagnostic("audio_mic", dg.RecordSignal("mic", sim.sim_info, num_channels=1, channel_idx = 0, export_func="wav"))
    sim.diag.add_diagnostic("audio_mic_dynamic", dg.RecordSignal("mic_dynamic", sim.sim_info, num_channels=1, channel_idx=0, export_func="wav"))
    sim.diag.add_diagnostic("src", dg.RecordSignal("src", sim.sim_info, num_channels=sim.arrays["src"].num, export_func="npz"))

    #if "mic_dynamic_cardioid" in sim.arrays:
    #    sim.diag.add_diagnostic("mic_dynamic_cardioid", dg.RecordSignal("mic_dynamic_cardioid", sim.sim_info, num_channels=sim.arrays["mic_dynamic_cardioid"].num, #export_func="npz"))

    if "image" in sim.arrays:
        sim.diag.add_diagnostic("image", dg.RecordSignal("image", sim.sim_info, num_channels=sim.arrays["image"].num, export_func="npz"))
    #sim.diag.add_diagnostic("mic_dynamic", dg.RecordState("mic_dynamic", sim.sim_info, num_channels=1, export_func="npz"))
    #sim.diag.add_diagnostic("mic_dynamic", dg.RecordSignal("mic_dynamic", export_func="npz"))
    
    sim.run_simulation()
    sim.arrays.save_to_file(sim.folder_path)
    sim.sim_info.save_to_file(sim.folder_path)
    #pos_all = np.array(sim.arrays["mic_dynamic"].pos_all)
    #time_all = np.array(sim.arrays["mic_dynamic"].time_all)



def load_npz(signal_paths):
    sig = {}
    for sig_name, sig_path in signal_paths.items():
        loaded_data = np.load(sig_path)
        dict_data = {key: data for key, data in loaded_data.items()}
        for key, data in dict_data.items():
            assert key not in sig
            sig[key] = data
    return sig

def get_signal_paths(fig_folder):
    signal_paths = {}
    for f in fig_folder.iterdir():
        if f.suffix == ".npz":
            sig_name_components = f.stem.split("_")
            sig_name = "_".join(sig_name_components[:-1])
            signal_paths[sig_name] = f
    return signal_paths


def load_session(fig_folder):
    """
    """
    with open(fig_folder.joinpath("extra_parameters.json")) as f:
        extra_params = json.load(f)
    #samplerate = int(2 * extra_params["max frequency"] * bandwidth_factor)
    signal_paths = get_signal_paths(fig_folder)
    sig_orig = load_npz(signal_paths)
    sig = {key : val for key, val in sig_orig.items()}

    sim_info, arrays = sls.load_from_path(fig_folder)
    seq_len = arrays["src"].source.tot_samples
    pos_dyn = arrays["mic_dynamic"].pos[:,None,:]
    sig["mic_dynamic"] = load_dynamic_sig(sig_orig["mic_dynamic"], seq_len)
    
    extra_params["pseq_start_idx"] = sim_info.sim_buffer + extra_params["initial_delay"]

    #assert extra_params["downsampling"] == (sim_info.samplerate / samplerate)
    if "mic_dynamic_cardioid" in arrays: # cardioid response is already saved
        sig["mic_dynamic_cardioid"] = load_dynamic_sig(sig_orig["mic_dynamic_cardioid"], seq_len)
        pos_dyn_cardioid = arrays["mic_dynamic_cardioid"].pos[:,None,:]
        cardioid_direction = arrays["mic_dynamic_cardioid"].directivity_dir
        return sig, sim_info, arrays, pos_dyn, seq_len, extra_params, pos_dyn_cardioid, cardioid_direction
    
    if "mic_diff" in arrays: # generate cardioid from two omnis as differential array
        cardioid_direction = np.load(fig_folder.joinpath("cardioid_direction.npy"))

        diff_filt_len = 2 * seq_len
        cardioid_signal = fd.differential_cardioid_microphone(sig_orig["mic_dynamic"], sig_orig["mic_diff"], extra_params["differential distance"], diff_filt_len, sim_info.c, sim_info.samplerate, filter_below=50)

        assert cardioid_signal.shape[-1] >= 2*diff_filt_len + 2*seq_len
        cardioid_signal = cardioid_signal[:,diff_filt_len+seq_len:diff_filt_len+2*seq_len]

        sig["mic_dynamic_cardioid"] = load_dynamic_sig(cardioid_signal, seq_len)
        pos_dyn_cardioid = pos_dyn

        # samples_delay = sim_info.samplerate * extra_params["differential distance"] / sim_info.c
        # filter_order = sim_info.samplerate // 10
        # delay_filter, int_delay = fd.fractional_delay_filter(samples_delay, filter_order)
        # delayed_omni = spsig.fftconvolve(delay_filter[None,:], sig_orig["mic_diff"], axes=-1)
        # delayed_omni = delayed_omni[...,int_delay:]
        # delayed_omni = delayed_omni[...,:sig_orig["mic_dynamic"].shape[-1]]

        # fft_len = 2*sig_orig["mic_diff"].shape[-1]
        # orig_omni_freq = ft.rfft(sig_orig["mic_dynamic"], n=fft_len)
        # delayed_omni_freq = ft.rfft(delayed_omni, n=fft_len)

        # wave_num = ft.get_real_wavenum(fft_len, sim_info.samplerate, sim_info.c)
        # #angular_freq = 2 * np.pi * freqs
        # #wave_num = angular_freq / sim_info.c

        # # UNCOMMENT FOR TIME DOMAIN SOLUTION, MUST ALREADY BE DELAYEED
        # common_factor = np.conj(1 / (1 - np.exp(-1j * 2 * wave_num * extra_params["differential distance"])))
        # cardioid_freq = common_factor[:,None] * (orig_omni_freq - delayed_omni_freq)
        # cardioid_freq[0,:] = orig_omni_freq[0,:] # original common_factor for frequency 0 is inf.
        # cardioid = np.real_if_close(ft.irfft(cardioid_freq)[:,:sig_orig["mic_diff"].shape[-1]])

        # sig["mic_dynamic_cardioid"] = load_dynamic_sig(cardioid, seq_len)
        # pos_dyn_cardioid = pos_dyn

        return sig, sim_info, arrays, pos_dyn, seq_len, extra_params, pos_dyn_cardioid, cardioid_direction
        #zeroth frtequency is nan. Fix by setting to zero or just set to be the same for both omnis. 




        #sim.arrays.paths["src"]["cardioid_2"] = delayed_cardioid

    return sig_orig, sim_info, arrays, pos_dyn, seq_len, extra_params

def load_dynamic_sig(sig_raw, seq_len):
    """
    takes the many statoinary microphones and puts it together into one seemingly 
    moving microphone. 
    """
    tot_len = sig_raw.shape[-1]
    full_seqs = tot_len // seq_len
    sig_raw = sig_raw[:,(full_seqs-1)*seq_len:full_seqs*seq_len]

    #sig_raw = sig_raw[:,2*seq_len:3*seq_len]
    sig_len = sig_raw.shape[0]

    assert sig_len % seq_len == 0
    num_periods = sig_len // seq_len
    sig_out = np.zeros((1, sig_len))
    for i in range(sig_len):
        sig_out[0,i] = sig_raw[i,i % seq_len]
    return sig_out