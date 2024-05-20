
import numpy as np
import matplotlib.pyplot as plt




def image_scatter_freq_response(p_all, freqs, pos, fig_folder=None, plot_name="", same_scale=True, dot_size=500):
    """

    Parameters
    ----------
    ir_all_freq : dict of np.ndarray of shape (num_freq, num_pos) or (num_pos,)
        or single np.ndarray of shape (num_freq, num_pos) or (num_pos,)
        represents the complex sounds pressure for each frequency at each position
    freqs : np.ndarray of shape (num_freq,)
        frequency values in Hz
    pos : ndarray of shape (num_pos, 3)
        positions of each microphone
    fig_folder : pathlib.Path, optional
        folder where to save the figures
    plot_name : str, optional
        Provide to add suffix to the saved plot name. The default is "".

    ir_all_freq is a dict, where each value is a ndarray of shape (num_freq, num_pos)
    freqs is a 1-d np.ndarray with all frequencies
    pos is a ndarray of shape (num_pos, 3)
    """
    if isinstance(p_all, np.ndarray):
        p_all = {"": p_all}

    for ir_name, ir_val in p_all.items():
        if ir_val.ndim == 1:
            p_all[ir_name] = ir_val[None, :]
    
    num_pos = pos.shape[0]
    if np.mean(np.abs(pos[0,2] - pos[:,2])) > 1e-3:
        print("Warning: The z-coordinates of the positions are not equal. This is currently ignored, and the values are shown on the xy-plane without further note.")
    num_freqs = freqs.shape[-1]
    assert all([ir_val.shape == (num_freqs, num_pos) for ir_val in p_all.values()])

    num_example_freqs = np.min((4, num_freqs))
    idx_interval = num_freqs // num_example_freqs
    freq_idxs = np.arange(num_freqs, step=idx_interval)
    #freq_idxs = np.arange(num_freqs)[idx_selection]

    for fi in freq_idxs:
        fig, axes = plt.subplots(len(p_all), 3, figsize=(14, len(p_all)*4), squeeze=False)

        p_real_fi = {est_name : np.real(p_val[fi,:]) for est_name, p_val in p_all.items()}
        p_imag_fi = {est_name : np.imag(p_val[fi,:]) for est_name, p_val in p_all.items()}
        p_abs_fi = {est_name : np.abs(p_val[fi,:]) for est_name, p_val in p_all.items()}
        real_max = np.max([np.max(np.abs(p)) for p in p_real_fi.values()])
        imag_max = np.max([np.max(np.abs(p)) for p in p_imag_fi.values()])
        abs_max = np.max([np.max(p) for p in p_abs_fi.values()])
        abs_min = np.min([np.min(p) for p in p_abs_fi.values()])

        for ax_row, (est_name, ir_val) in zip(axes, p_all.items()):
            #mse_val += 1e-6
            #mse_val = 10 * np.log10(mse_val)

            clr = ax_row[0].scatter(pos[:,0], pos[:,1], c=np.real(ir_val[fi,:]), marker="s", s=dot_size, cmap="RdBu", vmin=-real_max, vmax=real_max)
            cbar = fig.colorbar(clr, ax=ax_row[0])
            cbar.set_label('Real pressure')

            clr = ax_row[1].scatter(pos[:,0], pos[:,1], c=np.imag(ir_val[fi,:]), marker="s", s=dot_size, cmap="RdBu", vmin=-imag_max, vmax=imag_max)
            cbar = fig.colorbar(clr, ax=ax_row[1])
            cbar.set_label('Imag pressure')

            clr = ax_row[2].scatter(pos[:,0], pos[:,1], c=np.abs(ir_val[fi,:]), marker="s", s=dot_size, vmin=abs_min, vmax=abs_max, cmap="inferno")
            cbar = fig.colorbar(clr, ax=ax_row[2])
            cbar.set_label('Abs pressure')

            ax_row[0].set_title(f"Real: {est_name}")
            ax_row[1].set_title(f"Imag: {est_name}")
            ax_row[2].set_title(f"Abs: {est_name}")

            for ax in ax_row:
                #ax.legend(loc="lower left")
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                
                ax.set_aspect("equal")
                #aspplot.set_basic_plot_look(ax)

        #plt.show()   
        #aspplot.output_plot("pdf", fig_folder, f"image_scatter_freq_{freqs[fi]}Hz{plot_name}")