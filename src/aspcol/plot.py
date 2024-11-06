"""Module for plotting helper functions

References
----------
"""

import matplotlib.pyplot as plt

try:
    import tikzplotlib
except ImportError:
    tikzplotlib = None

def _tikzplotlib_fix_ncols(obj):
    """Workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib

    Parameters
    ----------
    obj : Figure object
        Figure object to fix the _ncol attribute in.
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        _tikzplotlib_fix_ncols(child)


def save_plot(print_method, folder, name=""):
    """Save plot to file in a number of formats.

    Parameters
    ----------
    print_method : str
        Method for saving the plot. Options are 'show', 'tikz', 'pdf', 'svg', 'none'.
        If 'show', the plot is shown in a window.
        If 'tikz', the plot is saved as a tikz file and a pdf file. Requires tikzplotlib installed. 
        If 'pdf', the plot is saved as a pdf file.
        If 'svg', the plot is saved as a svg file.
        If 'none', the plot is not saved.
    folder : Path
        Folder to save the plot in.
    name : str, optional
        Name of the file. The default is "".
    """
    if print_method == "show":
        plt.show()
    elif print_method == "tikz":
        if folder is not None:
            nested_folder = folder.joinpath(name)
            try:
                nested_folder.mkdir()
            except FileExistsError:
                pass

            fig = plt.gcf()
            _tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                str(nested_folder.joinpath(f"{name}.tex")),
                externalize_tables=True,
                float_format=".8g",
            )
            plt.savefig(
                str(nested_folder.joinpath(name + ".pdf")),
                dpi=300,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                format="pdf",
                transparent=True,
                bbox_inches=None,
                pad_inches=0.2,
            )
    elif print_method == "pdf":
        if folder is not None:
            plt.savefig(
                str(folder.joinpath(name + ".pdf")),
                dpi=300,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                format="pdf",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.2,
            )
    elif print_method == "svg":
        if folder is not None:
            plt.savefig(
                str(folder.joinpath(name + ".svg")),
                dpi=300,
                format="svg",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.2,
            )
    elif print_method == "none":
        pass
    else:
        raise ValueError
    plt.close("all")


def set_basic_plot_look(ax):
    """Sets basic look for a plot.
    
    Parameters
    ----------
    ax : Axes
        Axes object to set the look of.
    """
    ax.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)