import pytest
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hypnp
import hypothesis as hyp

import numpy as np
#import soundfieldcontrol.szc.sinr_opt as sinropt
#import soundfieldcontrol.szc.sinr_opt_time as sinrt

import soundfieldcontrol as sfc

import aspcol.unittests.matrix_strategies as mst


SEED = 5

NUM_ZONES = st.shared(st.integers(min_value=2, max_value=4))
NUM_LS = st.shared(st.integers(min_value=1, max_value=3))
FILT_LEN = st.shared(st.integers(min_value=1, max_value=3))

@st.composite
def SPATIAL_COV_TD(draw):
    num_zones = draw(NUM_ZONES)
    num_ls = draw(NUM_LS)
    filt_len = draw(FILT_LEN)
    dim = num_ls * filt_len
    cov_mat = np.stack([np.stack([mst.psd_real(draw, dim, dim) for _ in range(num_zones)], axis=0) for _ in range(num_zones)], axis=0)
    return cov_mat

@st.composite
def SINR_TARGETS(draw):
    num_zones = draw(NUM_ZONES)
    targets = draw(hypnp.arrays(dtype=float, shape=num_zones, elements=st.floats(min_value=1e-5, max_value = 100, allow_infinity=False, allow_nan=False)))
    return targets

@st.composite
def AUDIO_COV(draw):
    """
    should really be toeplitz psd matrix
    """
    num_zones = draw(NUM_ZONES)
    num_ls = draw(NUM_LS)
    filt_len = draw(FILT_LEN)
    dim = num_ls * filt_len

    regularization = draw(st.floats(min_value = 0, max_value=1))
    cov_mat = np.stack([mst.psd_real(draw, dim) + regularization * np.eye(dim) for _ in range(num_zones)], axis=0)
    return cov_mat

#def RANDOM_BEAMFORMER(draw):

# SPATIAL_COV_TD = hypnp.arrays(dtype=dtype_tuple[0], 
#                                 shape=st.tuples(st.integers(min_value=1, max_value=1), outer_dim_l, inner_dim),
#                                 elements=st.complex_numbers(allow_infinity=False, allow_nan=False))
#outer_dim_l = st.integers(min_value=1, max_value=5)
#outer_dim_r = st.integers(min_value=1, max_value=5)
#dtype_st = st.shared(st.sampled_from((float, complex)))


# dtype_tuple = st.shared(st.sampled_from((
#                         (float, st.floats(allow_infinity=False, allow_nan=False)), 
#                         (complex, st.complex_numbers(allow_infinity=False, allow_nan=False))
#                             )))

# mat_left_real = hypnp.arrays(dtype=dtype_tuple[0], 
#                                 shape=st.tuples(st.integers(min_value=1, max_value=1), outer_dim_l, inner_dim),
#                                 elements=st.complex_numbers(allow_infinity=False, allow_nan=False))
# mat_left_complex = hypnp.arrays(dtype=complex, 
#                                 shape=st.tuples(st.integers(min_value=1, max_value=1), outer_dim_l, inner_dim),
#                                 elements=st.complex_numbers(allow_infinity=False, allow_nan=False))
# matl = st.sampled_from(mat_left_real, mat_left_complex)

# mat_right_real = hypnp.arrays(dtype=dtype_st, shape=st.tuples(st.integers(min_value=1, max_value=1), inner_dim, outer_dim_r))



def get_random_spatial_cov(num_zones, num_ls):
    rng = np.random.default_rng()
    rank = rng.integers(1, num_ls)
    R = np.zeros((num_zones, num_ls, num_ls), dtype=complex)
    for k in range(num_zones):
        for r in range(rank):
            v = rng.normal(size=(num_ls, 1)) + 1j*rng.normal(size=(num_ls, 1))
            R[k,:,:] += v @ v.conj().T

    assert np.allclose(R, np.moveaxis(R.conj(), 1,2))
    return R

def get_random_spatial_cov_full_rank(num_freq, num_ls):
    rng = np.random.default_rng()
    rank = num_ls + 4
    R = np.zeros((num_freq, num_ls, num_ls), dtype=complex)
    for k in range(num_freq):
        for r in range(rank):
            v = rng.normal(size=(num_ls, 1)) + 1j*rng.normal(size=(num_ls, 1))
            R[k,:,:] += v @ v.conj().T

    assert np.allclose(R, np.moveaxis(R.conj(), 1,2))
    return R


def get_random_beamformer(num_zones, num_ls):
    rng = np.random.default_rng()
    w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, num_ls))
    return w

def get_random_beamformer_normalized(num_zones, num_ls):
    rng = np.random.default_rng()
    w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, num_ls)) + 1j*rng.normal(scale=rng.uniform(high=5),size=(num_zones, num_ls))
    w /= np.sqrt(np.sum(np.abs(w)**2, axis=-1, keepdims=True))
    return w

def get_random_noise_pow(num_zones):
    rng = np.random.default_rng()
    return rng.uniform(low=1e-3, high=10, size=num_zones)

def get_random_sinr_targets(num_zones):
    rng = np.random.default_rng()
    exponent = rng.uniform(-3, 1, size=num_zones)
    return 10**(exponent)
    
def get_random_pow_vec(num_zones):
    rng = np.random.default_rng()
    exponent = rng.uniform(-3, 1, size=num_zones)
    return 10**(exponent)


def get_random_spatial_cov_td(num_zones, bf_len):
    rng = np.random.default_rng()
    #rank = rng.integers(1, num_ls)
    R = np.zeros((num_zones, num_zones, bf_len, bf_len), dtype=float)
    for k in range(num_zones):
        for i in range(num_zones):
            for r in range(bf_len+5):
                v = rng.normal(size=(bf_len, 1))
                R[k,i,:,:] += v @ v.T

    assert np.allclose(R, np.moveaxis(R, 2,3))
    return R

def get_random_audio_cov(num_zones, bf_len):
    rng = np.random.default_rng()
    reg_param = rng.uniform(low=1e-5, high=1)
    R = np.zeros((num_zones, bf_len, bf_len), dtype=float)
    for k in range(num_zones):
        for r in range(bf_len):
            v = rng.normal(size=(bf_len, 1))
            R[k,:,:] += v @ v.T
    R += reg_param * np.eye(bf_len)[None,:,:]

    assert np.allclose(R, np.moveaxis(R, 1,2))
    return R


def get_random_beamformer_normalized_td(num_zones, bf_len):
    rng = np.random.default_rng()
    w = rng.normal(scale=rng.uniform(high=10),size=(num_zones, bf_len))
    w /= np.sqrt(np.sum(np.abs(w)**2, axis=-1, keepdims=True))
    return w

@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
def test_sinr_uplink_specific_and_link_gain_formulation_equal(num_zones, num_ls):
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = np.ones_like(get_random_noise_pow(num_zones))
    p = get_random_pow_vec(num_zones)
    #w = sinropt.apply_power_vec(w_norm, p)
    #sinr_targets = get_random_sinr_targets(num_zones)

    sinr_val = sfc.sinr_uplink(p, w, R, noise_pow)
    #sinr_val1 = sinropt._sinr_uplink_comparison(w_norm, R, p, noise_pow)
    sinr_val2 = sfc._sinr_uplink_comparison(p, w, R, noise_pow)
    
    assert np.allclose(sinr_val, sinr_val2)



@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
def test_sinr_downlink_specific_and_link_gain_formulation_equal(num_zones, num_ls):
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = np.ones_like(get_random_noise_pow(num_zones))
    p = get_random_pow_vec(num_zones)
    #w = sinropt.apply_power_vec(w_norm, p)
    #sinr_targets = get_random_sinr_targets(num_zones)

    sinr_val = sfc.sinr_downlink(p, w, R, noise_pow)
    sinr_val2 = sfc._sinr_downlink_comparison(sfc.apply_power_vec(w, p), R, noise_pow)
    
    assert np.allclose(sinr_val, sinr_val2)

@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
    st.floats(min_value=1, max_value=1000),
)
def test_power_alloc_minmax_equal_to_max_pow(num_zones, num_ls, max_pow):
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = get_random_noise_pow(num_zones)
    sinr_targets = get_random_sinr_targets(num_zones)

    p_dl, _ = sfc.power_alloc_minmax_downlink(w, R, noise_pow, sinr_targets, max_pow)
    assert np.allclose(np.sum(p_dl), max_pow)

    p_ul, _ = sfc.power_alloc_minmax_uplink(w, R, noise_pow, sinr_targets, max_pow)
    assert np.allclose(np.sum(p_ul), max_pow)



@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
def test_power_alloc_qos_zero_sinr_margin_downlink(num_zones, num_ls):
    num_test_attempts = 100
    feasible = False
    for i in range(num_test_attempts):
        w = get_random_beamformer_normalized(num_zones, num_ls)
        R = get_random_spatial_cov(num_zones, num_ls)
        noise_pow = get_random_noise_pow(num_zones)
        rng = np.random.default_rng(SEED+12)
        sinr_targets = rng.uniform(low=0.05, high=0.3, size=num_zones)
        if sfc._power_alloc_qos_is_feasible(sfc.link_gain_downlink(w, R), sinr_targets):
            feasible = True
            break
    assert feasible
    
    p_dl = sfc.power_alloc_qos_downlink(w, R, noise_pow, sinr_targets)
    margin_dl = sfc.sinr_margin_downlink(p_dl, w, R, noise_pow, sinr_targets)
    assert np.allclose(margin_dl, 0)

@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
)
def test_power_alloc_qos_zero_sinr_margin_uplink(num_zones, num_ls):
    """
    Can be flaky, because it does not always get a feasible problem to
        begin with. 
    """
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    noise_pow = get_random_noise_pow(num_zones)
    rng = np.random.default_rng(SEED+12)
    sinr_targets = rng.uniform(low=0.05, high=0.05, size=num_zones)

    #R, noise_pow = sinropt.normalize_system(R, noise_pow)

    p = sfc.power_alloc_qos_uplink(w, R, noise_pow, sinr_targets)
    margin = sfc.sinr_margin_uplink(p, w,  R, noise_pow, sinr_targets)
    #sinr = sinropt._sinr_uplink_comparison(w, R, p, noise_pow)
    #sinr2 = sinropt.sinr_uplink(p, w, R, noise_pow)
    assert np.allclose(margin, 0)

    # p_dl = sinropt.power_alloc_qos_downlink(w, R, noise_pow, sinr_targets)
    # margin_dl = sinropt.sinr_margin_downlink(sinropt.apply_power_vec(w, p_dl), R, noise_pow, sinr_targets)
    
    # assert np.allclose(margin_dl, 0)

    


@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=3, max_value=5),
    st.floats(min_value=1, max_value=10000),
)
def test_equal_capacity_uplink_downlink_duality(num_zones, num_ls, max_pow):
    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    ones = np.ones_like(get_random_noise_pow(num_zones))
    #noise_pow = get_random_noise_pow(num_zones)
    sinr_targets = get_random_sinr_targets(num_zones)

    q, c_ul = sfc.power_alloc_minmax_uplink(w, R, ones, sinr_targets, max_pow)
    p, c_dl = sfc.power_alloc_minmax_downlink(w, R, ones, sinr_targets, max_pow)
    assert np.allclose(c_ul, c_dl)


@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=3, max_value=5),
)
def test_equal_feasibility_spectral_radius_uplink_downlink(num_zones, num_ls):

    w = get_random_beamformer_normalized(num_zones, num_ls)
    R = get_random_spatial_cov(num_zones, num_ls)
    sinr_targets = get_random_sinr_targets(num_zones)

    gain_ul = sfc.link_gain_uplink(w, R)
    spec_rad_ul = sfc._power_alloc_qos_feasibility_spectral_radius(gain_ul, sinr_targets)
    
    gain_dl = sfc.link_gain_downlink(w, R)
    spec_rad_dl = sfc._power_alloc_qos_feasibility_spectral_radius(gain_dl, sinr_targets)

    assert np.allclose(spec_rad_ul, spec_rad_dl)







@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=3, max_value=5),
    #st.floats(min_value=100, max_value=10000),
)
def test_uplink_capacity_equals_feasiblity_spectral_radius(num_zones, num_ls):
    num_feasible = 0
    num_non_feasible = 0
    for i in range(1000):
        w = get_random_beamformer_normalized(num_zones, num_ls)
        R = get_random_spatial_cov(num_zones, num_ls)
        noise_pow = np.ones_like(get_random_noise_pow(num_zones))
        #noise_pow = get_random_noise_pow(num_zones)
        sinr_targets = get_random_sinr_targets(num_zones)

        gain_ul = sfc.link_gain_uplink(w, R)
        spec_rad_ul = sfc._power_alloc_qos_feasibility_spectral_radius(gain_ul, sinr_targets)
        
        q, c_ul = sfc.power_alloc_minmax_uplink(w, R, noise_pow, sinr_targets, 1e6)
        
        print(c_ul > 1)
        if c_ul >= 1:
            assert spec_rad_ul < 1
            num_feasible += 1
        else:
            num_non_feasible += 1
    assert num_feasible > 0 
    assert num_non_feasible > 0
        #assert (c_ul >= 1) == (spec_rad_ul < 1)



#unity_capacity_equals_feasibility


@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=3, max_value=5),
)
def test_same_qos_power_alloc_feasibility_spectral_radius(num_zones, num_ls, max_pow):
    assert False



@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=3, max_value=5),
    st.floats(min_value=1, max_value=10000),
)
def test_same_objective_implies_same_beamformer(num_zones, num_ls, max_pow):
    assert False


@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=3, max_value=5),
    st.floats(min_value=1, max_value=10000),
)
def test_strong_duality_holds_freq_domain_sdr_vs_schubert(num_zones, num_ls, max_pow):
    assert False



@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=5, max_value=10),
)
def test_strong_duality_holds_time_domain_sdr_vs_schubert(num_zones, bf_len):
    R = get_random_spatial_cov_td(num_zones, bf_len)
    noise_pow = np.ones(num_zones)
    sinr_targets = get_random_sinr_targets(num_zones)
    w, q = sfc.solve_qos_uplink(R, noise_pow, sinr_targets, 1e3)

    audio_cov = np.stack([np.eye(bf_len) for _ in range(num_zones)], axis=0)
    
    W2 = sinrt.sinr_constrained_weighted_pow_min_downlink(R, noise_pow, sinr_targets, audio_cov)
    w2 = sfc.select_solution_eigenvalue(W2)
    w2, p = sfc.extract_power_vec(w2)
    
    assert np.allclose(np.sum(q), np.sum(p))


@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=3, max_value=10),
)
def test_strong_duality_holds_weighted_time_domain_sdr_vs_schubert(num_zones, bf_len):
    R = get_random_spatial_cov_td(num_zones, bf_len)
    audio_cov = get_random_audio_cov(num_zones, bf_len)
    noise_pow = np.ones(num_zones)
    sinr_targets = get_random_sinr_targets(num_zones)
    w, q = sfc.solve_power_weighted_qos_uplink(R, noise_pow, sinr_targets, 1e3, audio_cov)

    ul_objective = np.sum(q)
    
    W2 = sinrt.sinr_constrained_weighted_pow_min_downlink(R, noise_pow, sinr_targets, audio_cov)
    w2 = sfc.select_solution_eigenvalue(W2)
    w2, p = sfc.extract_power_vec(w2)

    dl_objective = np.sum([p[k] * w2[k,:,None].T @ audio_cov[k,:,:] @ w2[k,:,None] for k in range(num_zones)])
    
    assert np.allclose(ul_objective, dl_objective)



@hyp.settings(deadline=None)
@given(
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5),
    st.floats(min_value=0.5, max_value=5),
)
def test_mwf_and_gevd_with_full_rank_are_equivalent(num_zones, num_ls, mu):
    Rb = get_random_spatial_cov_full_rank(num_zones, num_ls)
    Rd = get_random_spatial_cov_full_rank(num_zones, num_ls)

    mwf = sfc.szc_transform_mwf(Rb, Rd, mu)
    gevd = sfc.szc_transform_mwf_gevd(Rb, Rd, num_ls, mu)
    assert np.allclose(mwf, gevd)

