"""Helper functions for naive monte carlo sampling of a given function
"""
import numpy as np



def integrate_parallell(
    func, point_generator, tot_num_samples, total_volume, num_per_iter=5, verbose=False
):
    """Uses mcIntegrate but paralellized"""
    import dill
    dill.settings["recurse"] = True
    import pathos.multiprocessing as mp
    from pathos.helpers import freeze_support


    ncpu = mp.cpu_count()
    # ncpu = 2
    integration_samples = int(np.ceil(tot_num_samples / ncpu))
    intArgs = [
        [func for _ in range(ncpu)],
        [point_generator for _ in range(ncpu)],
        [integration_samples for _ in range(ncpu)],
        [total_volume for _ in range(ncpu)],
        [num_per_iter for _ in range(ncpu)],
        [verbose for _ in range(ncpu)],
    ]

    with mp.ProcessingPool(ncpu) as p:
        integral = p.map(integrate, *intArgs)

    integral = np.mean(np.stack(integral, axis=-1), axis=-1)
    return integral

def integrate_fast(
    func, point_generator, tot_num_samples, total_volume, *args, num_per_iter=50,
):
    """identical to integrate, but is meant to be simpler."""
    num_blocks = int(np.ceil(tot_num_samples / num_per_iter))
    test_val = func(point_generator(1), *args)
    out_dims = np.squeeze(test_val, axis=-1).shape
    integral_val = np.zeros(out_dims, dtype=test_val.dtype)

    for i in range(num_blocks):
        f_vals = func(point_generator(num_per_iter), *args)
        integral_val = (integral_val * i + np.mean(f_vals, axis=-1)) / (i + 1)
    integral_val *= total_volume
    return integral_val



def integrate(
    func, point_generator, tot_num_samples, total_volume, num_per_iter=50, verbose=False, *args
):
    """pointGenerator should return np array, [numPoints, numSpatialDimensions]
    func should return np array [funcDims, numPoints],
    where funcDims can be any number of dimensions (in a multidimensional array sense)"""
    print("Starting MC Integration")
    samples_per_iter = num_per_iter
    num_blocks = int(np.ceil(tot_num_samples / samples_per_iter))
    out_dims = np.squeeze(func(point_generator(1), *args), axis=-1).shape
    integral_val = np.zeros(out_dims)

    for i in range(num_blocks):
        points = point_generator(samples_per_iter)
        f_vals = func(points, *args)

        new_int_val = (integral_val * i + np.mean(f_vals, axis=-1)) / (i + 1)
        print("Block ", i)
        if verbose:
            diagnostics(new_int_val, integral_val, i)

        integral_val = new_int_val
    integral_val *= total_volume
    print("Finished!!")
    return integral_val


def diagnostics(new_val, old_val, block_idx):
    # print("Block ", blockIdx)
    change = new_val - old_val
    rel_change = change / old_val

    mean_square_change = np.mean(np.square(new_val - old_val))
    mean_abs_change = np.mean(np.abs(new_val - old_val))
    print("Mean Square Change: " + "{:2.12E}".format(mean_square_change))
    print("Mean Abs Change: " + "{:2.12E}".format(mean_abs_change))

    max_abs_change = np.max(np.abs(change))
    max_abs_idx = np.argmax(np.abs(change))
    max_abs_change_val = np.abs(new_val.flatten()[max_abs_idx])
    max_abx_change_rel = np.abs(rel_change.flatten()[max_abs_idx])
    print(
        "Max Abs change: "
        + "{:2.12E}".format(max_abs_change)
        + "     Its Value: "
        + "{:2.12E}".format(max_abs_change_val)
        + "     rel change: "
        + "{:2.12E}".format(max_abx_change_rel)
    )

    max_rel_change = np.max(np.abs(rel_change))
    max_rel_idx = np.argmax(np.abs(rel_change))
    max_rel_change_val = np.abs(new_val.flatten()[max_rel_idx])
    max_rel_change_abs = np.abs(change.flatten()[max_rel_idx])
    print(
        "Max rel change: "
        + "{:2.12E}".format(max_rel_change)
        + "     Its Value: "
        + "{:2.12E}".format(max_rel_change_val)
        + "     abs change "
        + "{:2.12E}".format(max_rel_change_abs)
    )

    avg_rel_change = np.mean(np.abs(rel_change))
    avg_abs_change = np.mean(np.abs(change))
    print(
        "Avg rel change: "
        + "{:2.12E}".format(avg_rel_change)
        + "     Avg abs change: "
        + "{:2.12E}".format(avg_abs_change)
    )

    max_val = np.max(np.abs(new_val))
    print("Max Value: " + "{:2.12E}".format(max_val))
    print()
    return max_rel_change
