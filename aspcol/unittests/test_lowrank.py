
import numpy as np
import aspcore.filterclasses as fc

import hypothesis as hyp
import hypothesis.strategies as st

import aspcol.lowrank as lr
import time


@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=1, max_value=3),
            num_out = st.integers(min_value=1, max_value=3), 
            rank = st.integers(min_value=1, max_value=3), 
            ir_len1 = st.integers(min_value=3, max_value=7),
            ir_len2 = st.integers(min_value=3, max_value=7))
def test_low_rank_2d_filter_same_result_as_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2):
    rng = np.random.default_rng()
    ir1 = rng.normal(size=(num_in, num_out, rank, ir_len1))
    ir2 = rng.normal(size=(num_in, num_out, rank, ir_len2))
    lr_ir = (ir1, ir2)
    ir = lr.reconstruct_ir(lr_ir)
    tot_len = ir_len1 * ir_len2

    lrfilt = lr.LowRankFilter2D(ir1, ir2)
    filt = fc.create_filter(ir)

    num_samples = 30
    in_sig = rng.normal(size = (num_in, num_samples))

    out_sig = filt.process(in_sig)
    #out_sig = filt.process(in_sig)
    out_lr = lrfilt.process(in_sig)
    #out_lr = lrfilt.process(in_sig)
    assert np.allclose(out_sig, out_lr)
    #assert np.allclose(out_sig[:,tot_len:], out_lr[:,tot_len:])


@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=1, max_value=3),
            num_out = st.integers(min_value=1, max_value=3), 
            rank = st.integers(min_value=1, max_value=3), 
            ir_len1 = st.integers(min_value=3, max_value=7),
            ir_len2 = st.integers(min_value=3, max_value=7),
            ir_len3 = st.integers(min_value=3, max_value=7),)
def test_low_rank_3d_filter_same_result_as_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2, ir_len3):
    rng = np.random.default_rng()
    ir1 = rng.normal(size=(num_in, num_out, rank, ir_len1))
    ir2 = rng.normal(size=(num_in, num_out, rank, ir_len2))
    ir3 = rng.normal(size=(num_in, num_out, rank, ir_len3))
    lr_ir = (ir1, ir2, ir3)
    ir = lr.reconstruct_ir(lr_ir)
    tot_len = ir_len1 * ir_len2 * ir_len3

    lrfilt = lr.LowRankFilter3D(ir1, ir2, ir3)
    filt = fc.create_filter(ir)

    num_samples = 30
    in_sig = rng.normal(size = (num_in, num_samples))

    out_sig = filt.process(in_sig)
    #out_sig = filt.process(in_sig)
    out_lr = lrfilt.process(in_sig)
    #out_lr = lrfilt.process(in_sig)
    assert np.allclose(out_sig, out_lr)
    #assert np.allclose(out_sig[:,tot_len:], out_lr[:,tot_len:])



@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=5, max_value=5),
            num_out = st.integers(min_value=5, max_value=5), 
            rank = st.integers(min_value=10, max_value=10), 
            ir_len1 = st.integers(min_value=128, max_value=128),
            ir_len2 = st.integers(min_value=128, max_value=128))
def test_speed_comparison_low_rank_2d_filter_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2):
    """
    Clearly not a great way to write a unit test, but is just here as a sanity check to 
    flag if the low rank is slower for a case it should be faster for. First run gave 
    0.9s for regular filter, 0.2s for LowRankFilter2D. 
    """
    rng = np.random.default_rng()
    ir1 = rng.normal(size=(num_in, num_out, rank, ir_len1))
    ir2 = rng.normal(size=(num_in, num_out, rank, ir_len2))
    lr_ir = (ir1, ir2)
    ir = lr.reconstruct_ir(lr_ir)

    lrfilt = lr.LowRankFilter2D(ir1, ir2)
    filt = fc.create_filter(ir)

    num_samples = 1000
    in_sig = rng.normal(size = (num_in, num_samples))

    t1 = time.process_time()
    out_sig = filt.process(in_sig)
    t2 = time.process_time()
    s1 = t2-t1
    t3 = time.process_time()
    out_lr = lrfilt.process(in_sig)
    t4 = time.process_time()
    s2 = t4-t3
    
    #assert np.allclose(s1, s2)
    assert s1 > s2


@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=5, max_value=5),
            num_out = st.integers(min_value=5, max_value=5), 
            rank = st.integers(min_value=30, max_value=30), 
            ir_len1 = st.integers(min_value=32, max_value=32),
            ir_len2 = st.integers(min_value=32, max_value=32),
            ir_len3 = st.integers(min_value=32, max_value=32),)
def test_speed_comparison_low_rank_3d_filter_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2, ir_len3):
    """
    Clearly not a great way to write a unit test, but is just here as a sanity check to 
    flag if the low rank is slower for a case it should be faster for. 

    Change assert to np.allclose(s1, s2) to see the difference in speed yourself.
    """
    rng = np.random.default_rng()
    ir1 = rng.normal(size=(num_in, num_out, rank, ir_len1))
    ir2 = rng.normal(size=(num_in, num_out, rank, ir_len2))
    ir3 = rng.normal(size=(num_in, num_out, rank, ir_len3))
    lr_ir = (ir1, ir2, ir3)
    ir = lr.reconstruct_ir(lr_ir)

    lrfilt = lr.LowRankFilter3D(ir1, ir2, ir3)
    filt = fc.create_filter(ir)

    num_samples = 1000
    in_sig = rng.normal(size = (num_in, num_samples))

    t1 = time.process_time()
    out_sig = filt.process(in_sig)
    t2 = time.process_time()
    s1 = t2-t1
    t3 = time.process_time()
    out_lr = lrfilt.process(in_sig)
    t4 = time.process_time()
    s2 = t4-t3
    #assert np.allclose(s1, s2)
    assert s1 > s2