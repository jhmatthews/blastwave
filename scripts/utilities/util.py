from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm
import numpy as np
from scipy.signal import convolve
from scipy.signal.windows import boxcar
from cycler import cycler
import os
import astropy.constants as const

default = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

g_DataDir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..', 'data'))
g_FigureDir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..', 'figures'))
g_SimDir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..', 'simulations'))

def save_paper_figure(savename, fig=None, figure_dir=g_FigureDir, transparent=False, **savefig_kwargs):
    """wrapper to save a paper figure in the main figures directory"""
    if fig == None:
        fig = plt.gcf()

    full_savename = "{}/{}".format(figure_dir, savename)
    fig.savefig(full_savename, **savefig_kwargs)
    if transparent:
        full_savename_tr= "{}/transp_{}".format(figure_dir, savename)
        if ".pdf" in full_savename_tr:
            full_savename_tr = full_savename_tr.replace(".pdf", ".png")
        fig.savefig(full_savename_tr, dpi=300, transparent=True, **savefig_kwargs)


def gradient_fill(x, y, fill_color=None, ax=None, mappable=None, alpha=None, zorder=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    kwargs2 = kwargs.copy()
    kwargs2["label"] = None
    kwargs2["lw"] = 0
    line, = ax.plot(x, y, **kwargs2)
    if fill_color is None:
        fill_color = line.get_color()

    if zorder is None:
        zorder = line.get_zorder()
    if alpha is None:
        alpha = line.get_alpha()
        alpha = 1.0 if alpha is None else alpha

    if mappable is None:
        Nz = 10000
        z = np.empty((Nz, 1, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        rgb = fill_color[:3]
        z[:, :, :3] = rgb

        z[:, :, -1] = np.linspace(0, 0.85, Nz)[:, None] ** 1.5

    else:
        xx, yy = np.meshgrid(x, y)
        z = mappable.to_rgba(xx[::-1])
        z[:, :, -1] = alpha

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, 0, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, 0], xy, [xmax, 0], [xmin, 0]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)
    line, = ax.plot(x, y, color=fill_color, **kwargs, zorder=zorder)
    return line, im

SEABORN = dict(
    deep=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    deep6=["#4C72B0", "#55A868", "#C44E52",
           "#8172B3", "#CCB974", "#64B5CD"],
    muted=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
           "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"],
    muted6=["#4878D0", "#6ACC64", "#D65F5F",
            "#956CB4", "#D5BB67", "#82C6E2"],
    pastel=["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
            "#DEBB9B", "#FAB0E4", "#CFCFCF", "#FFFEA3", "#B9F2F0"],
    pastel6=["#A1C9F4", "#8DE5A1", "#FF9F9B",
             "#D0BBFF", "#FFFEA3", "#B9F2F0"],
    bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
            "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"],
    bright6=["#023EFF", "#1AC938", "#E8000B",
             "#8B2BE2", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71",
          "#592F0D", "#A23582", "#3C3C3C", "#B8850A", "#006374"],
    dark6=["#001C7F", "#12711C", "#8C0800",
           "#591E71", "#B8850A", "#006374"],
    colorblind=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"],
    colorblind6=["#0173B2", "#029E73", "#D55E00",
                 "#CC78BC", "#ECE133", "#56B4E9"]
)


def set_ui_cycler(name="canada"):
    if name == "canada" or name == None:
        colors = ["#2e86de", "#ff9f43", "#10ac84",
                  "#ee5253", "#341f97", "#feca57", "#ff9ff3"]
    elif name == "british":
        colors = ["#0097e6", "#e1b12c", "#8c7ae6", "#c23616",
                  "#273c75", "#353b48", "#44bd32", "#fbc531"]
    elif name in SEABORN.keys():
        colors = SEABORN[name]
    my_cycler = cycler('color', colors)
    plt.rc('axes', prop_cycle=my_cycler)


def set_plot_defaults(tex="True"):

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["text.usetex"] = tex
    if tex == "True":
        # plt.rcParams['font.serif'] = ['Times']
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


    plt.rcParams['font.size'] = 18
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rcParams["lines.linewidth"] = 2.2
    # TICKS
    plt.rcParams['xtick.top'] = 'True'
    plt.rcParams['xtick.bottom'] = 'True'
    plt.rcParams['xtick.minor.visible'] = 'True'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.left'] = 'True'
    plt.rcParams['ytick.right'] = 'True'
    plt.rcParams['ytick.minor.visible'] = 'True'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['ytick.minor.size'] = 3


def get_aspect(ax):
    from operator import sub
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)

    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    return (disp_ratio, data_ratio)


def grav_radius(mass):
    rg = const.GM_sun.cgs.value * mass / const.c.cgs.value / const.c.cgs.value
    return rg


def smooth(data, width=5):
    """
    boxcar smooth 1d data
    """
    if (width) > 1:
        q = convolve(data, boxcar(width) / float(width), mode="same")
        return q
    else:
        return data


def set_cmap_cycler(cmap_name="viridis", N=None):
    '''
    set the cycler to use a colormap
    '''
    if cmap_name == "default" or N is None:
        my_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    else:
        _, colors = get_mappable(N, cmap_name=cmap_name)
        # if type(style) == str:
        my_cycler = (cycler(color=colors))

    plt.rc('axes', prop_cycle=my_cycler)


class colour_func:
    """
    A class for managing a colormap and its normalization based on given minimum and maximum values.

    Attributes:
    ----------
    vmin : float
        The minimum value for normalization.
    vmax : float
        The maximum value for normalization.
    cmap_name : str
        The name of the colormap to be used.
    my_cmap : Colormap
        The colormap instance from matplotlib.

    Methods:
    -------
    __init__(self, vmin, vmax, cmap_name):
        Initializes the colour_func with normalization and colormap.
    """

    def __init__(self, vmin, vmax, cmap_name):
        my_cmap = matplotlib.colormaps.get_cmap(cmap_name)


def get_mappable(N, vmin=0, vmax=1, cmap_name="Spectral", return_func=False):
    """
    Generates a ScalarMappable object and an array of colors based on the given colormap.

    Parameters:
    ----------
    N : int
        The number of colors to generate.
    vmin : float, optional
        The minimum value for normalization. Default is 0.
    vmax : float, optional
        The maximum value for normalization. Default is 1.
    cmap_name : str, optional
        The name of the colormap to be used. Default is "Spectral".
    return_func : bool, optional
        Whether to return a colour_func instance for color mapping. Default is False.

    Returns:
    -------
    tuple
        A tuple containing:
            - mappable (ScalarMappable): The ScalarMappable object for the colormap.
            - colors (ndarray): An array of RGBA colors.
            - to_rgba (function), optional: A function for mapping values to RGBA colors (only if return_func is True).
    """
    my_cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colors = my_cmap(np.linspace(0, 1, num=N))

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_name)
    if return_func:
        #fcol = colour_func(norm, cmap_name)
        return (mappable, colors, mappable.to_rgba)
    else:
        return (mappable, colors)




