import numpy as np

import aspcol.kernelinterpolation as ki
import aspcol.soundfieldestimation as sfe

import aspcore.fouriertransform as ft
import aspcore.matrices as aspmat

import matplotlib.pyplot as plt


def test_time_domain_estimation_is_equivalent_to_multifreq_domain_estimation():
    pass

def test_time_domain_estimation_is_equivalent_to_individual_frequency_domain_estimation():
    pass


def test_time_domain_both_regularization_formulations_are_equivalent_for_diffuse_kernel():
    r"""This is a basic test which is expected to be true even if the other one fails. Only a sanity check.
    
    The optimization problem is \lVert h_m - M_r u \rVert_Q^2 + \lambda \lVert R u \rVert_H^2

    Then the following should be equivalent

    \Gamma_{r^{-1}}(r, r') = 
    
    """
    rng = np.random.default_rng()
    num_mic = 5
    num_eval = 4
    ir_len = 100

    samplerate = 1000
    c = 343
    reg_param = 1

    pos_mic = rng.uniform(-1, 1, (num_mic, 3))
    pos_eval = rng.uniform(-1, 1, (num_eval, 3))
    ir_data = rng.uniform(-1, 1, (num_mic, ir_len))

    est_std = sfe.krr_stationary_mics(ir_data, pos_mic, pos_eval, samplerate, c, reg_param, ki.time_domain_diffuse_kernel, [], verbose=False, max_cond=None, data_weighting = None)
    est_reg = sfe.krr_stationary_mics_regularized(ir_data, pos_mic, pos_eval, samplerate, c, reg_param, ki.time_domain_diffuse_kernel, [], ki.time_domain_diffuse_kernel, [], verbose=False, max_cond=None, data_weighting = None)
    assert np.allclose(est_std, est_reg)


def test_time_domain_both_regularization_formulations_are_equivalent_for_directional_vonmises_kernel():
    r"""Testing whether both regularization approaches are equivalent when the kernel is the closed form directional kernel
    
    The optimization problem is \lVert h_m - M_r u \rVert_Q^2 + \lambda \lVert R u \rVert_H^2

    Then the following should be equivalent

    \Gamma_{r^{-1}}(r, r') = 
    
    """
    rng = np.random.default_rng()
    num_mic = 5
    num_eval = 4
    ir_len = 100

    samplerate = 1000
    c = 343
    reg_param = 1
    beta = 1
    direction = np.array([1, 0, 0])

    pos_mic = rng.uniform(-1, 1, (num_mic, 3))
    pos_eval = rng.uniform(-1, 1, (num_eval, 3))
    ir_data = rng.uniform(-1, 1, (num_mic, ir_len))


    #est_changedip = sfe.krr_stationary_mics_direction_regularized_changedip(ir_data, pos_mic, pos_eval, samplerate, c, reg_param, direction, beta)

    est_std = sfe.krr_stationary_mics(ir_data, pos_mic, pos_eval, samplerate, c, reg_param, ki.time_domain_directional_kernel_vonmises, [-direction, beta], verbose=False, max_cond=None, data_weighting = None)
    est_reg = sfe.krr_stationary_mics_regularized(ir_data, pos_mic, pos_eval, samplerate, c, reg_param, ki.time_domain_directional_kernel_vonmises, [direction, beta], ki.time_domain_directional_kernel_vonmises, [direction, 3*beta], verbose=False, max_cond=None, data_weighting = None)

    pass
    assert np.allclose(est_std, est_reg)