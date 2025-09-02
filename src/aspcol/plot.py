"""Module for plotting helper functions

References
----------
"""

import matplotlib.pyplot as plt
import numpy as np
import json

import aspcore.fouriertransform as ft
import aspcore.utilities as utils

# import tikzplotlib
# try:
#     import tikzplotlib
# except ImportError:
#     tikzplotlib = None

# def _tikzplotlib_fix_ncols(obj):
#     """Workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib

#     Parameters
#     ----------
#     obj : Figure object
#         Figure object to fix the _ncol attribute in.
#     """
#     if hasattr(obj, "_ncols"):
#         obj._ncol = obj._ncols
#     for child in obj.get_children():
#         _tikzplotlib_fix_ncols(child)


# def save_plot(print_method, folder, name=""):
#     """Save plot to file in a number of formats.

#     Parameters
#     ----------
#     print_method : str
#         Method for saving the plot. Options are 'show', 'tikz', 'pdf', 'svg', 'none'.
#         If 'show', the plot is shown in a window.
#         If 'tikz', the plot is saved as a tikz file and a pdf file. Requires tikzplotlib installed. 
#         If 'pdf', the plot is saved as a pdf file.
#         If 'svg', the plot is saved as a svg file.
#         If 'none', the plot is not saved.
#     folder : Path
#         Folder to save the plot in.
#     name : str, optional
#         Name of the file. The default is "".
#     """
#     if print_method == "show":
#         plt.show()
#     elif print_method == "tikz":
#         if folder is not None:
#             nested_folder = folder.joinpath(name)
#             try:
#                 nested_folder.mkdir()
#             except FileExistsError:
#                 pass

#             fig = plt.gcf()
#             _tikzplotlib_fix_ncols(fig)
#             tikzplotlib.save(
#                 str(nested_folder.joinpath(f"{name}.tex")),
#                 externalize_tables=True,
#                 float_format=".8g",
#             )
#             plt.savefig(
#                 str(nested_folder.joinpath(name + ".pdf")),
#                 dpi=300,
#                 facecolor="w",
#                 edgecolor="w",
#                 orientation="portrait",
#                 format="pdf",
#                 transparent=True,
#                 bbox_inches=None,
#                 pad_inches=0.2,
#             )
#     elif print_method == "pdf":
#         if folder is not None:
#             plt.savefig(
#                 str(folder.joinpath(name + ".pdf")),
#                 dpi=300,
#                 facecolor="w",
#                 edgecolor="w",
#                 orientation="portrait",
#                 format="pdf",
#                 transparent=True,
#                 bbox_inches="tight",
#                 pad_inches=0.2,
#             )
#     elif print_method == "svg":
#         if folder is not None:
#             plt.savefig(
#                 str(folder.joinpath(name + ".svg")),
#                 dpi=300,
#                 format="svg",
#                 transparent=True,
#                 bbox_inches="tight",
#                 pad_inches=0.2,
#             )
#     elif print_method == "none":
#         pass
#     else:
#         raise ValueError
#     plt.close("all")


# def set_basic_plot_look(ax):
#     """Sets basic look for a plot.
    
#     Parameters
#     ----------
#     ax : Axes
#         Axes object to set the look of.
#     """
#     ax.grid(True)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)




def soundfield_estimation_comparison(
        pos_est, 
        p_est, 
        p_true, 
        freqs, 
        fig_folder,
        shape="",
        center=None,
        num_ls=1, 
        output_method="pdf", 
        pos_mic = None,
        images=None, 
        image_true=None,
        pos_image=None,
        remove_freqs_below=0,
        remove_freqs_above=np.inf,
        num_examples = 4,
        points_for_errors = None,
    ):
    """Outputs a series of plots comparing a set of sound field estimates to the true sound field.
    
    The function accepts sound field estimates in the frequency domain. 

    Parameters
    ----------
    pos_est : np.ndarray
        Positions of the estimated sound field. Shape (num_positions, spatial_dim)
    p_est : np.ndarray
        Estimated sound field. Shape (num_freqs, num_ls, num_positions) or (num_freqs, num_positions)
    p_true : np.ndarray
        True sound field. Shape (num_freqs, num_ls, num_positions) or (num_freqs, num_positions)
    freqs : np.ndarray of shape (num_freqs,)
        Frequencies used in the simulation.
    fig_folder : pathlib Path
        Folder to save the figures in.
    shape : str, optional
        Shape of the evaluation point array. If any of the valid options "rectangle", "circle" is used, then a few more plots will be created. 
        The default is "".
    center : np.ndarray, optional
        Center of the circle if shape is "circle". The default is None.
    num_ls : int, optional
        Number of loudspeakers. The default is 1.
    output_method : str, optional
        Method for saving the plot. Options are 'show', 'tikz', 'pdf', 'svg', 'none'.
        If 'show', the plot is shown in a window.
        If 'tikz', the plot is saved as a tikz file and a pdf file. Requires tikzplotlib installed.
        If 'pdf', the plot is saved as a pdf file.
        If 'svg', the plot is saved as a svg file.
        If 'none', the plot is not saved. The default is "pdf".
    pos_mic : np.ndarray, optional
        Positions of the microphones. Shape (num_mics, spatial_dim)
        The positions are added to the sound field images if supplied. The default is None.
    images : np.ndarray, optional
        Estimated sound field on a grid array appropriate for creating a sound field image. Shape (num_freqs, num_image)
    image_true : np.ndarray, optional
        True sound field on a grid array appropriate for creating a sound field image. Shape (num_freqs, num_image)
        If supplied along with images, the error between the images and the true value will be shown. 
    pos_image : np.ndarray, optional
        Positions of the image array. Shape (num_image, spatial_dim)
    remove_freqs_below : float, optional
        Remove frequencies below this value in Hz. The default is 0.
    remove_freqs_above : float; optional
        Remove frequencies above this value in Hz. 
    points_for_errors : np.ndarray of shape (num_positions,) with Boolean values, optional
        Points which have True values are included in error calculations. The default is None.
        It also specifies which points to use in setting of colormap limits in the soundfield MSE plots. Useful if a few points
        are known exactly, which usually leads to useless sound field MSE plots. 
    """
    p_all, p_est, p_true = _cleanup_args(p_est, p_true, num_ls)
    fig_folder.mkdir(exist_ok=True, parents=True)

    p_all_orig = p_all
    p_est_orig = p_est
    p_true_orig = p_true
    freqs_orig = freqs

    example_responses(p_all_orig, freqs_orig, fig_folder, output_method=output_method, num_examples=num_examples) # needs to be done before removing 0Hz bin
    error_per_sample(p_est_orig, p_true_orig, fig_folder, output_method=output_method)

    freqs_to_keep = np.logical_and(freqs > remove_freqs_below, freqs < remove_freqs_above)
    p_all = {name : sig[freqs_to_keep,...] for name, sig in p_all.items()}
    p_est = {name : sig[freqs_to_keep,...] for name, sig in p_est.items()}
    p_true = p_true[freqs_to_keep,...]
    freqs = freqs[freqs_to_keep]

    if points_for_errors is not None:
        p_est_masked = {name : sig[..., points_for_errors] for name, sig in p_est.items()}
        p_true_masked = p_true[..., points_for_errors]
    else:
        p_est_masked = p_est
        p_true_masked = p_true

    mse(p_est_masked, p_true_masked, fig_folder, num_ls = num_ls)
    error_per_frequency(p_est_masked, p_true_masked, freqs, fig_folder, output_method=output_method)

    if num_ls > 1:
        for l in range(num_ls):
            #p_all_single = {name : sig[:,l:l+1,:] for name, sig in p_all.items()}
            p_est_single = {name : sig[:,l:l+1,:] for name, sig in p_est_masked.items()}
            p_true_single = p_true_masked[:,l:l+1,:]
            error_per_frequency(p_est_single, p_true_single, freqs, fig_folder, output_method=output_method, plot_name=f"src_{l}")


    if shape == "rectangle":
        rectangle_folder = fig_folder / "rectangle"
        rectangle_folder.mkdir(exist_ok=True)
        compare_soundfields_all_freq(p_all, p_est, p_true, freqs, pos_est, rectangle_folder, pos_mic=pos_mic, output_method=output_method, num_examples=num_examples, points_for_errors=points_for_errors)
        compare_soundfields_all_time(p_all_orig, p_est_orig, p_true_orig, freqs_orig, pos_est, rectangle_folder, pos_mic=pos_mic, output_method=output_method, num_examples=num_examples, points_for_errors=points_for_errors)

        #compare_soundfields(p_all, freqs, arrays, fig_folder)
        #compare_soundfield_error(p_est, p_true, freqs, arrays, fig_folder)
    elif shape == "circle":
        assert center is not None
        if exclude_points_from_error is not None:
            raise NotImplementedError("Excluding points from error is not implemented for circular shapes")
        error_per_angle(pos_est, p_est, p_true, center, fig_folder, output_method=output_method)
        estimates_per_angle(pos_est, p_all, freqs, center, fig_folder, output_method=output_method)

    if images is not None:
        assert pos_image is not None
        if image_true is None:
            image_true_placeholder = images[list(images.keys())[0]]
            im_all, im_est, _ = _cleanup_args(images, image_true_placeholder, num_ls)
            im_all.pop("true")
            im_true = None
        else:
            im_all, im_est, im_true = _cleanup_args(images, image_true, num_ls)
        image_folder = fig_folder / "images"
        image_folder.mkdir(exist_ok=True)

        compare_soundfields_all_freq(im_all, im_est, im_true, freqs, pos_image, image_folder, pos_mic=pos_mic, output_method=output_method, num_examples=num_examples, num_ls=num_ls)
        compare_soundfields_all_time(im_all, im_est, im_true, freqs, pos_image, image_folder, pos_mic=pos_mic, output_method=output_method, num_examples=num_examples, num_ls=num_ls)






def _cleanup_args(p_est, p_true, num_ls = 1):
    """Cleans up the input arguments to a standardized format for the plotting functions
    
    The arguments are assumed to be frequency-domain signals

    Parameters
    ----------
    p_est : np.ndarray or dict of np.ndarrays
        Each ndarray should have the shape (num_freqs, num_ls, num_positions) or (num_freqs, num_positions)
    p_true : np.ndarray
        True sound field. Should be the same shape as p_est, but only one is supplied. 
    num_ls : int, optional
        Number of loudspeakers. The default is 1. Supply this if the sound field values contains data
        from multiple loudspeakers. 

    Returns
    -------
    p_all : dict of np.ndarrays
        Dictionary containing the cleaned up estimates as well as the true sound field
    p_est : dict of np.ndarrays
        Dictionary containing the cleaned up estimates
    p_true : np.ndarray
        The cleaned up true sound field
    """
    if isinstance(p_est, np.ndarray):
        p_est = {"estimate" : p_est}
    for name, est in p_est.items():
        if est.ndim == 3 and est.shape[-1] == 1: # assume we have vectors that have not been squeezed properly
            p_est[name] = np.squeeze(p_est[name], axis=-1)
        if num_ls == 1:
            #if est.ndim == 3:
            #    assert est.shape[1] == 1
            #    p_est[name] = np.squeeze(est, axis=1) # squeeze the loudspeaker dim
            if p_est[name].ndim == 2: # add the singleton loudspeaker axis
                p_est[name] = p_est[name][:,None,:]
        #elif num_ls > 1:
        #    assert est.shape[1] == num_ls
            #raise NotImplementedError("The reshaping must be looked at since switching to aspcol fft function")
            #td_est = ft.irfft(p_est[name])
            #assert td_est.shape[1] % num_ls == 0
            #rir_len = td_est.shape[-1] // num_ls
            #num_eval = td_est.shape[1]
            #td_est = np.reshape(td_est.T, (num_eval, num_ls, rir_len))
            #p_est[name] = ft.rfft(td_est, axis=-1).T
        assert p_est[name].ndim == 3
        assert p_est[name].shape[1] == num_ls

    if p_true.ndim == 3 and p_true.shape[-1] == 1:
        p_true = np.squeeze(p_true, axis=-1)
    if num_ls == 1 and p_true.ndim == 2:
        p_true = p_true[:,None,:]
    for name, est in p_est.items():
        assert p_est[name].shape == p_true.shape
    
    p_all = {name : est for name, est in p_est.items()}
    p_all["true"] = p_true
    return p_all, p_est, p_true

def _get_freq_example_idxs(num_freqs, num_examples):
    if num_freqs <= num_examples:
        idxs = np.arange(num_freqs)
    else:
        idxs = np.linspace(num_freqs/num_examples, num_freqs-num_freqs/num_examples, num_examples).astype(int)
    return idxs

def example_responses(p_all, freqs, fig_folder, num_examples = 3, output_method="pdf"):
    rng = np.random.default_rng(123456)
    num_estimates = p_all[list(p_all.keys())[0]].shape[-1]

    idxs = rng.permutation(np.arange(num_estimates))[:num_examples]
    freq_response(p_all, freqs, fig_folder, idxs, output_method)
    time_response(p_all, fig_folder, idxs, output_method, scaling="linear")
    time_response(p_all, fig_folder, idxs, output_method, scaling="db")

def freq_response(p_all, freqs, fig_folder, index_to_show, output_method="pdf"):
    fig, axes = plt.subplots(len(index_to_show), 1, figsize = (8,4*len(index_to_show)))

    for i, ax in enumerate(axes):
        for name, est in p_all.items():
            selected_est = est[:,0, index_to_show[i]]
            selected_est = 20*np.log10(np.abs(selected_est))
            ax.plot(freqs, selected_est, label = name, alpha=0.8)

        ax.set_title(f"Estimate index {index_to_show[i]}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Absolute amplitude")
        ax.legend(loc="best")
        utils.set_basic_plot_look(ax)
    utils.save_plot(output_method, fig_folder, f"freq_response")

def time_response(p_all, fig_folder, index_to_show, output_method="pdf", scaling="linear"):
    fig, axes = plt.subplots(len(index_to_show), 1, figsize = (8,4*len(index_to_show)))

    for i, ax in enumerate(axes):
        for name, est in p_all.items():
            time_est = np.reshape(ft.irfft(est[:,:,index_to_show[i]]), -1)
            #time_est = np.reshape(np.fft.irfft(est[:,:,index_to_show[i]], axis=0).T, -1).T
            if scaling == "db":
                time_est = 20*np.log10(np.abs(time_est))
            elif scaling != "linear":
                raise ValueError("Invalid scaling option")
            ax.plot(time_est, label = name, alpha=0.8)

        ax.set_title(f"Estimate index {index_to_show[i]}")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="best")
        utils.set_basic_plot_look(ax)
    utils.save_plot(output_method, fig_folder, f"time_response_{scaling}")


def error_per_sample(p_est, p_true, fig_folder, output_method="pdf"):
    true_td = ft.irfft(p_true)

    fig, ax = plt.subplots(1,1, figsize = (8,4))
    for name, est in p_est.items():
        time_est = ft.irfft(est)
        error = np.mean(np.abs(true_td - time_est)**2, axis=(0,1)) / np.mean(np.abs(true_td)**2)
        error = 10*np.log10(error)
        ax.plot(error, label=name)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Mean square error (dB)")
    ax.legend(loc="best")
    utils.set_basic_plot_look(ax)
    utils.save_plot(output_method, fig_folder, "error_per_sample")

def error_per_angle(pos_est, p_est, p_true, center, fig_folder, output_method="pdf"):
    angle = np.arctan2(pos_est[:,1]-center[0,1], pos_est[:,0]-center[0,0])
    index_array = np.argsort(angle)

    sorted_angle = angle[index_array]

    fig, ax = plt.subplots(1,1, figsize = (8,4))
    for name, est in p_est.items():
        
        error_per_pos = np.mean(np.abs(est - p_true)**2, axis=(0,1)) / np.mean(np.abs(p_true)**2, axis=(0,1))
        error_per_pos = 10*np.log10(error_per_pos)
        error_per_pos = error_per_pos[index_array]
        ax.plot(sorted_angle, error_per_pos, label=name)
        
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Square error (dB)")
    ax.legend(loc="best")
    utils.set_basic_plot_look(ax)
    utils.save_plot(output_method, fig_folder, "error_per_angle")

def estimates_per_angle(pos_est, p_all, freqs, arrays, center, fig_folder, output_method="pdf", num_examples=5):
    angle = np.arctan2(pos_est[:,1]-center[0,1], pos_est[:,0]-center[0,0])
    index_array = np.argsort(angle)

    sorted_angle = angle[index_array]

    num_freqs = len(freqs) / 2
    num_sf_plots = num_examples
    freq_idxs = np.linspace(num_freqs/num_sf_plots, num_freqs-num_freqs/num_sf_plots, num_sf_plots)
    freq_idxs = [int(f) for f in freq_idxs]
    fig, axes = plt.subplots(len(freq_idxs), 3, figsize = (16,10))
    fig.tight_layout()
    for f_idx, f in enumerate(freq_idxs):
        for i, (name, est) in enumerate(p_all.items()):
            axes[f_idx,0].plot(sorted_angle, np.abs(est[f,0,:][index_array]), label=name)
            axes[f_idx,1].plot(sorted_angle, np.real(est[f,0,:][index_array]), label=name)
            axes[f_idx,2].plot(sorted_angle, np.imag(est[f,0,:][index_array]), label=name)

        axes[f_idx,0].set_title(f"Absolute pressure at {freqs[f]}")
        axes[f_idx,1].set_title(f"Real pressure at {freqs[f]}")
        axes[f_idx,2].set_title(f"Imag pressure at {freqs[f]}")

        axes[f_idx,0].set_ylabel("Amplitude (abs)")
        axes[f_idx,1].set_ylabel("Amplitude (real)")
        axes[f_idx,1].set_ylabel("Amplitude (real)")
        for j in range(3):
            axes[f_idx,j].set_xlabel("Angle (radians)")
            axes[f_idx,j].legend(loc="best")
            utils.set_basic_plot_look(axes[f_idx, j])

        plt.suptitle(f"Estimated pressures at frequency {freqs[f]} Hz")
    utils.save_plot(output_method, fig_folder, f"estimates_per_angle")

    #fig, ax = plt.subplots(1,2, figsize = (8,4))
    #for name, est in p_all.items():
    #    error_per_pos = error_per_pos[index_array]
    #    ax[0].plot(sorted_angle, est[index_array], label=name)
    
    
    #plt.legend(loc="best")
    #save_plot("pdf", fig_folder, "error_per_angle", False)



def soundfield_image(sig, pos, pos_mic=None, title="", vminmax=None, ax = None, cmap="inferno"):
    sig = np.squeeze(sig)
    if sig.ndim == 1:
        sig = sig[:,None]
    assert sig.shape[0] == pos.shape[0]
    if ax is None:
        fig, ax = plt.subplots()
    
    pos, sig2 = _sort_for_imshow(pos, sig)

    if vminmax is None:
        im = ax.imshow(sig2, interpolation="none", extent=(pos[:,:,0].min(), pos[:,:,0].max(), pos[:,:,1].min(), pos[:,:,1].max()), cmap=cmap)
    else:
        im = ax.imshow(sig2, interpolation="none", extent=(pos[:,:,0].min(), pos[:,:,0].max(), pos[:,:,1].min(), pos[:,:,1].max()), vmin=vminmax[0], vmax=vminmax[1], cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    
    if pos_mic is not None:
        for mp in pos_mic:
            ax.plot(mp[0], mp[1], "x")



def mse(p_est, p_true, fig_folder, num_ls = 1):
    if isinstance(p_est, np.ndarray):
        p_est = {"estimate" : p_est}

    #total MSE
    mse = {}
    for name, p in p_est.items():
        mse[name] = np.mean(np.abs(p - p_true)**2) / np.mean(np.abs(p_true)**2)
    with open(fig_folder.joinpath("mse.json"), "w") as f:
        json.dump(mse, f, indent=4)

    mse_db = {name : 10*np.log10(mse_val) for name, mse_val in mse.items()}
    with open(fig_folder.joinpath("mse_db.json"), "w") as f:
        json.dump(mse_db, f, indent=4)

    if num_ls > 1:
        mse = {}
        for name, p in p_est.items():
            mse[name] = np.mean(np.abs(p - p_true)**2, axis=(0,2)) / np.mean(np.abs(p_true)**2, axis=(0,2))
            mse[name] = mse[name].tolist()
        with open(fig_folder.joinpath("mse_src.json"), "w") as f:
            json.dump(mse, f, indent=4)

        mse_db = {name : (10*np.log10(mse_val)).tolist() for name, mse_val in mse.items()}
        with open(fig_folder.joinpath("mse_db_src.json"), "w") as f:
            json.dump(mse_db, f, indent=4)


def error_per_frequency(p_est, p_true, freqs, fig_folder, output_method="pdf", plot_name=""):
    """Plots the normalized mean square error as a function of frequency
    """
    fig, ax = plt.subplots(1,1, figsize = (8,4))
    for name, est in p_est.items():
        square_error = np.abs(est - p_true)**2
        error_per_freq = np.mean(square_error, axis=(1,2)) / np.mean(np.abs(p_true)**2, axis=(1,2))
        error_per_freq = 10*np.log10(error_per_freq)
        ax.plot(freqs, error_per_freq, label=name, alpha=0.85)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Square error (dB)")
    plt.legend(loc="best")
    utils.set_basic_plot_look(ax)

    plot_name = f"error_per_frequency{plot_name}"
    utils.save_plot(output_method, fig_folder, plot_name)

def compare_soundfields_all_freq(p_all, p_est, p_true, freqs, pos_im, fig_folder, pos_mic=None, output_method="pdf", num_examples=5, plot_name="", num_ls=1, points_for_errors = None):
    if num_ls > 1:
        if p_true is not None:
            compare_soundfields({name : 10*np.log10(np.mean(np.abs(p_true - p)**2, axis=1)) for name, p in p_est.items()}, freqs, pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"{plot_name}_square_error", output_method=output_method, only_positive=True, num_examples=num_examples, points_to_set_colormap=points_for_errors)
            compare_soundfields({name : 10*np.log10(np.mean(np.abs(p_true - p)**2, axis=(0,1)))[None,...] 
                                for name, p in p_est.items()}, np.zeros((1)), pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"{plot_name}_mse", output_method=output_method, only_positive=True, num_examples=1)
    for l in range(num_ls):
        if num_ls > 1:
            p_all_l = {name : p[:,l,:] for name, p in p_all.items()}
            p_est_l = {name : p[:,l,:] for name, p in p_est.items()}
            p_true_l = p_true[:,l,:]
            extra_name = f"_src_{l}"
        else:
            p_all_l = p_all
            p_est_l = p_est
            p_true_l = p_true
            extra_name = ""
        _single_src_soundfield_comparison(p_all_l, p_est_l, p_true_l, freqs, pos_im, fig_folder, pos_mic=pos_mic, output_method=output_method, num_examples=num_examples, plot_name=f"{plot_name}{extra_name}", points_to_set_colormap=points_for_errors)
            #compare_soundfields({name : 10*np.log10(np.abs(p_true[:,l,:] - p[:,l,:])**2) for name, p in p_est.items()}, freqs, pos_im, pos_mic, fig_folder, plot_name=f"{plot_name}_square_error_src_{l}", output_method=output_method, only_positive=True, num_examples=num_examples)

def _single_src_soundfield_comparison(p_all, p_est, p_true, freqs, pos_im, fig_folder, pos_mic=None, output_method="pdf", num_examples=5, plot_name="", points_to_set_colormap=None):
    compare_soundfields({name : np.abs(p) for name, p in p_all.items()}, freqs, pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"{plot_name}_abs", output_method=output_method, only_positive=True, num_examples=num_examples)
    compare_soundfields({name : np.real(p) for name, p in p_all.items()}, freqs, pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"{plot_name}_real", output_method=output_method, num_examples=num_examples)
    compare_soundfields({name : np.imag(p) for name, p in p_all.items()}, freqs, pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"{plot_name}_imag", output_method=output_method, num_examples=num_examples)
    if p_true is not None:
        compare_soundfields({name : 10*np.log10(np.abs(p_true - p)**2) for name, p in p_est.items()}, freqs, pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"{plot_name}_square_error", output_method=output_method, only_positive=True, num_examples=num_examples, points_to_set_colormap=points_to_set_colormap)
        compare_soundfields({name : 10*np.log10(np.mean(np.abs(p_true - p)**2, axis=0, keepdims=True)) 
                            for name, p in p_est.items()}, np.zeros((1)), pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"{plot_name}_mse", output_method=output_method, only_positive=True, num_examples=1, points_to_set_colormap=points_to_set_colormap)
        compare_soundfields({name : 10*np.log10(np.mean(np.abs(p_true - p)**2, axis=0, keepdims=True) / np.mean(np.abs(p_true)**2, axis=0, keepdims=True)) 
                            for name, p in p_est.items()}, np.zeros((1)), pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"{plot_name}_nmse", output_method=output_method, only_positive=True, num_examples=1, points_to_set_colormap=points_to_set_colormap)


    
def compare_soundfields_all_time(p_all, p_est, p_true, freqs, pos_im, fig_folder, pos_mic=None, output_method="pdf", num_examples=5, num_ls = 1, points_for_errors = None):
    if freqs.shape[-1] > 10: # if there's a reasonable amount of frequencies, show time domain responses
        for l in range(num_ls):
            if num_ls > 1:
                p_all_l = {name : p[:,l,:] for name, p in p_all.items()}
                extra_name = f"_src_{l}"
            else:
                p_all_l = p_all
                extra_name = ""
        
            ir_all = {name : ft.irfft(p) for name, p in p_all_l.items()}
            compare_time_domain_soundfields(ir_all, pos_im, fig_folder, pos_mic=pos_mic, plot_name=f"timedomain{extra_name}", output_method=output_method, num_examples=num_examples)
            

def compare_time_domain_soundfields(ir_all, pos_im, fig_folder, pos_mic=None, plot_name="", num_examples = 5, output_method="pdf", only_positive=False):
    num_samples = ir_all[list(ir_all.keys())[0]].shape[-1]
    if ir_all[list(ir_all.keys())[0]].ndim == 3:
        ir_all = {name : np.squeeze(ir, axis=0) for name, ir in ir_all.items()}

    num_td_plots = num_examples
    td_idxs = np.linspace(num_samples/num_td_plots, num_samples-num_samples/num_td_plots, num_td_plots).astype(int)

    mean_energy = np.mean(np.stack([np.mean(np.abs(ir)**2, axis=0) for ir in ir_all.values()], axis=0), axis=0)
    assert mean_energy.ndim == 1
    max_energy_idx = np.argmax(mean_energy)
    td_idxs = np.concatenate((td_idxs, [max_energy_idx]))

    for n in td_idxs:
        fig, axes = plt.subplots(len(ir_all), 1, figsize = (16,10))
        if not isinstance(axes, (list, np.ndarray, tuple)):
            axes = [axes]
        fig.tight_layout()

        v_max = np.max([np.max(ir[...,n]) for ir in ir_all.values()])
        v_min = np.min([np.min(ir[...,n]) for ir in ir_all.values()])

        if only_positive:
            cmap="inferno"
        else:
            max_abs_value = np.max((np.abs(v_max), np.abs(v_min)))
            v_min = -max_abs_value
            v_max = max_abs_value
            cmap="RdBu"

        for i, (name, ir) in enumerate(ir_all.items()):
            soundfield_image(ir[...,n], pos_im, pos_mic, ax=axes[i], title=f"{name}", vminmax=(v_min, v_max), cmap=cmap)

        plt.suptitle(f"Time domain sound field {plot_name} at sample {n}")
        utils.save_plot(output_method, fig_folder, f"soundfield_td_{plot_name}_{n}")


def compare_soundfields(p_all, freqs, pos_im, fig_folder, pos_mic=None, plot_name="", num_examples = 5, output_method="pdf", only_positive=False, points_to_set_colormap=None):
    num_freqs = len(freqs)

    freq_idxs = _get_freq_example_idxs(num_freqs, num_examples)
    #num_sf_plots = num_examples
    #freq_idxs = np.linspace(num_freqs/num_sf_plots, num_freqs-num_freqs/num_sf_plots, num_sf_plots)
    #freq_idxs = [int(f) for f in freq_idxs]

    for f in freq_idxs:
        fig, axes = plt.subplots(len(p_all), 1, figsize = (16,10))
        if not isinstance(axes, (list, np.ndarray, tuple)):
            axes = [axes]
        fig.tight_layout()

        if points_to_set_colormap is not None:
            v_max = np.max([np.max(sf[f,...,points_to_set_colormap]) for sf in p_all.values()])
            v_min = np.min([np.min(sf[f,...,points_to_set_colormap]) for sf in p_all.values()])
        else:
            v_max = np.max([np.max(sf[f,...]) for sf in p_all.values()])
            v_min = np.min([np.min(sf[f,...]) for sf in p_all.values()])


        if only_positive:
            cmap="inferno"
        else:
            max_abs_value = np.max((np.abs(v_max), np.abs(v_min)))
            v_min = -max_abs_value
            v_max = max_abs_value
            cmap="RdBu"
            

        for i, (name, sf) in enumerate(p_all.items()):
            soundfield_image(sf[f,...], pos_im, pos_mic, ax=axes[i], title=f"{name}", vminmax=(v_min, v_max), cmap=cmap)


        plt.suptitle(f"Sound field {plot_name} at frequency {freqs[f]} Hz")
        utils.save_plot(output_method, fig_folder, f"soundfield_{plot_name}_{freqs[f]}")





def _sort_for_imshow(pos, sig, pos_decimals=5):
    """
        Sorts the position and signal values to display correctly when
        imshow is used to plot the sound field image

        Parameters
        ---------
        pos : ndarray of shape (num_pos, spatial_dim)
            must represent a rectangular grid, but can be in any order.
        sig : ndarray of shape (num_pos, signal_dim)
            signal value for each position of the sound field
        pos_decimals : int
            selects how many decimals the position values are rounded to when 
            calculating all the unique position values
        
        Returns
        -------
        pos_sorted : ndarray of shape (num_rows, num_cols, spatial_dim)
        sig_sorted : ndarray of shape (num_rows, num_cols, signal_dim)
    """
    if pos.shape[1] == 3:
        assert np.allclose(pos[:,2], np.ones_like(pos[:,2])*pos[0,2])

    num_rows, num_cols = _get_num_pixels(pos, pos_decimals)
    unique_x = np.unique(pos[:,0].round(pos_decimals))
    unique_y = np.unique(pos[:,1].round(pos_decimals))

    sort_indices = np.zeros((num_rows, num_cols), dtype=int)
    for i, y in enumerate(unique_y):
        row_indices = np.where(np.abs(pos[:,1] - y) < 10**(-pos_decimals))[0]
        row_permutation = np.argsort(pos[row_indices,0])
        sort_indices[i,:] = row_indices[row_permutation]

    pos = pos[sort_indices,:]

    #sig = np.moveaxis(np.atleast_3d(sig),1,2)
    #dims = sig.shape[:2]
    signal_dim = sig.shape[-1]
    sig_sorted = np.zeros((num_rows, num_cols, signal_dim), dtype=sig.dtype)
    sig_sorted = np.flip(sig[sort_indices,:], axis=0)
    #for i in range(dims[0]):
     #   for j in range(dims[1]):
     #       single_sig = sig[i,j,:]
     #       sig_sorted[i,j,:,:] = np.flip(single_sig[sort_indices],axis=0)
    # sig_sorted = np.squeeze(sig_sorted)
    
    #sig = [np.flip(s[sortIndices],axis=0) for s in sig]
    
    return pos, sig_sorted


def _get_num_pixels(pos, pos_decimals=5):
    pos_cols = np.unique(pos[:,0].round(pos_decimals))
    pos_rows = np.unique(pos[:,1].round(pos_decimals))
    num_rows = len(pos_rows)
    num_cols = len(pos_cols)
    return num_rows, num_cols