
import numpy as np
import aspcore.filterclasses as fc

import hypothesis as hyp
import hypothesis.strategies as st

import aspcol.lowrank as lr
import time


def compare_two_processors(processor1, processor2, num_in, num_out, num_samples, block_size, rng=None):
    """
    helper function to process a random signal with two processors in 
    a block-wise way
    """
    if rng is None:
        rng = np.random.default_rng()
    in_sig = rng.normal(size = (num_in, num_samples))
    out_sig1 = np.zeros((num_out, num_samples))
    out_sig2 = np.zeros((num_out, num_samples))

    num_blocks = num_samples // block_size + 1
    for b in range(num_blocks):
        bs = min(block_size, num_samples - b*block_size)
        start, end = b*block_size, b*block_size+bs
        out_sig1[:,start:end] = processor1.process(in_sig[:,start:end])
        out_sig2[:,start:end] = processor2.process(in_sig[:,start:end])
    return out_sig1, out_sig2
    


@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=1, max_value=3),
            num_out = st.integers(min_value=1, max_value=3), 
            rank = st.integers(min_value=1, max_value=3), 
            ir_len1 = st.integers(min_value=3, max_value=7),
            ir_len2 = st.integers(min_value=3, max_value=7),
            block_size = st.integers(min_value=1, max_value=10),
            num_samples = st.integers(min_value=5, max_value=35))
def test_low_rank_2d_filter_same_result_as_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2, block_size, num_samples):
    rng = np.random.default_rng()
    ir1 = rng.normal(size=(num_in, num_out, rank, ir_len1))
    ir2 = rng.normal(size=(num_in, num_out, rank, ir_len2))
    lr_ir = (ir1, ir2)
    ir = lr.reconstruct_ir(lr_ir)

    lrfilt = lr.create_filter(lr_ir)
    filt = fc.create_filter(ir)

    out_sig, out_lr = compare_two_processors(filt, lrfilt, num_in, num_out, num_samples, block_size, rng)
    assert np.allclose(out_sig, out_lr)


@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=1, max_value=3),
            num_out = st.integers(min_value=1, max_value=3), 
            rank = st.integers(min_value=1, max_value=3), 
            ir_len1 = st.integers(min_value=3, max_value=7),
            ir_len2 = st.integers(min_value=3, max_value=7),
            block_size = st.integers(min_value=1, max_value=10),
            num_samples = st.integers(min_value=5, max_value=35))
def test_low_rank_2d_filter_same_result_as_decomposed_and_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2, block_size, num_samples):
    rng = np.random.default_rng()
    ir_orig = rng.normal(size=(num_in, num_out, ir_len1*ir_len2))
    ir_decomp = lr.decompose_ir(ir_orig, (ir_len1, ir_len2), rank)
    ir = lr.reconstruct_ir(ir_decomp)
    tot_len = ir_len1 * ir_len2

    lrfilt = lr.create_filter(ir_decomp)
    filt = fc.create_filter(ir)
    out_sig, out_lr = compare_two_processors(filt, lrfilt, num_in, num_out, num_samples, block_size, rng)
    assert np.allclose(out_sig, out_lr)

@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=1, max_value=3),
            num_out = st.integers(min_value=1, max_value=3), 
            rank = st.integers(min_value=1, max_value=3), 
            ir_len1 = st.integers(min_value=3, max_value=7),
            ir_len2 = st.integers(min_value=3, max_value=7),
            block_size = st.integers(min_value=1, max_value=10),
            num_samples = st.integers(min_value=5, max_value=35))
def test_low_rank_2d_filter_nosum_same_result_as_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2, block_size, num_samples):
    rng = np.random.default_rng()
    ir1 = rng.normal(size=(num_in, num_out, rank, ir_len1))
    ir2 = rng.normal(size=(num_in, num_out, rank, ir_len2))
    lr_ir = (ir1, ir2)
    ir = lr.reconstruct_ir(lr_ir)
    filt_lr = lr.create_filter(ir = (ir1, ir2))
    filt = fc.create_filter(ir, sum_over_input=False)

    in_sig = rng.normal(size = (num_in, num_samples))
    out_sig = np.zeros((num_in, num_out, num_samples))
    out_lr = np.zeros((num_in, num_out, num_samples))

    num_blocks = num_samples // block_size + 1
    for b in range(num_blocks):
        bs = min(block_size, num_samples - b*block_size)
        start, end = b*block_size, b*block_size+bs
        out_sig[...,start:end] = filt.process(in_sig[:,start:end])
        out_lr[...,start:end] = filt_lr.process_nosum(in_sig[:,start:end])
    assert np.allclose(out_sig, out_lr)
    #assert np.allclose(out_sig[:,tot_len:], out_lr[:,tot_len:])


@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=1, max_value=3),
            num_out = st.integers(min_value=1, max_value=3), 
            rank = st.integers(min_value=1, max_value=3), 
            ir_len1 = st.integers(min_value=3, max_value=7),
            ir_len2 = st.integers(min_value=3, max_value=7),
            ir_len3 = st.integers(min_value=3, max_value=7),
            block_size = st.integers(min_value=1, max_value=10),
            num_samples = st.integers(min_value=5, max_value=35)
        )
def test_low_rank_3d_filter_same_result_as_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2, ir_len3, block_size, num_samples):
    rng = np.random.default_rng()
    ir1 = rng.normal(size=(num_in, num_out, rank, ir_len1))
    ir2 = rng.normal(size=(num_in, num_out, rank, ir_len2))
    ir3 = rng.normal(size=(num_in, num_out, rank, ir_len3))
    lr_ir = (ir1, ir2, ir3)
    ir = lr.reconstruct_ir(lr_ir)
    tot_len = ir_len1 * ir_len2 * ir_len3

    lrfilt = lr.LowRankFilter3D(ir1, ir2, ir3)
    filt = fc.create_filter(ir)

    out_sig, out_lr = compare_two_processors(filt, lrfilt, num_in, num_out, num_samples, block_size, rng)
    assert np.allclose(out_sig, out_lr)
    #assert np.allclose(out_sig[:,tot_len:], out_lr[:,tot_len:])


@hyp.settings(deadline=None)
@hyp.given(num_in = st.integers(min_value=1, max_value=3),
            num_out = st.integers(min_value=1, max_value=3), 
            rank = st.integers(min_value=1, max_value=3), 
            ir_len1 = st.integers(min_value=3, max_value=7),
            ir_len2 = st.integers(min_value=3, max_value=7),
            ir_len3 = st.integers(min_value=3, max_value=7),
            block_size = st.integers(min_value=1, max_value=10),
            num_samples = st.integers(min_value=5, max_value=35))
def test_low_rank_3d_filter_nosum_same_result_as_reconstructed_filter(num_in, num_out, rank, ir_len1, ir_len2, ir_len3, block_size, num_samples):
    rng = np.random.default_rng()
    ir1 = rng.normal(size=(num_in, num_out, rank, ir_len1))
    ir2 = rng.normal(size=(num_in, num_out, rank, ir_len2))
    ir3 = rng.normal(size=(num_in, num_out, rank, ir_len3))
    lr_ir = (ir1, ir2, ir3)
    ir = lr.reconstruct_ir(lr_ir)
    filt_lr = lr.create_filter(ir = lr_ir)
    filt = fc.create_filter(ir, sum_over_input=False)

    in_sig = rng.normal(size = (num_in, num_samples))
    out_sig = np.zeros((num_in, num_out, num_samples))
    out_lr = np.zeros((num_in, num_out, num_samples))

    num_blocks = num_samples // block_size + 1
    for b in range(num_blocks):
        bs = min(block_size, num_samples - b*block_size)
        start, end = b*block_size, b*block_size+bs
        out_sig[...,start:end] = filt.process(in_sig[:,start:end])
        out_lr[...,start:end] = filt_lr.process_nosum(in_sig[:,start:end])
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

    lrfilt = lr.create_filter(lr_ir)
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