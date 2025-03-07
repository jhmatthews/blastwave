import numpy as np
from matplotlib import pyplot as plt
import astropy.io.ascii as io 
from scipy import stats
from utilities import util 
from utilities import blast_wave_util 
from utilities.blast_wave_util import MPROT, C

def make_figure(transparent=False):
    """Make the figure for the Lorentz factor distribution.

    Args:
        transparent (bool, optional): Whether to make the figure transparent. Defaults to False.
    """
    data = io.read("{}/grb_data.dat".format(util.g_DataDir))
    data2 = io.read("{}/xrb_data.dat".format(util.g_DataDir))
    bins = np.logspace(0.01,3.5,50)

    plt.figure(figsize=(10,4))

    francesco = [2.6, 1.6, 3.4, 1.85]

    select1 = data["Lower_Limit"] == "n"
    plt.hist(data["gamma"], density=False, color=util.default[0], alpha=0.6, bins=bins, edgecolor="k", label=r"$\Gamma_0$, GRBs, lower limits (Table A2)", hatch="\\\\")
    plt.hist(data["gamma"][select1], density=False, color=util.default[0], alpha=1.0, edgecolor="k", bins=bins, label=r"$\Gamma_0$, GRBs, heterogenous methodologies (Table A1)", fill=True)

    full_data = np.concatenate([data2["Gamma"], francesco])

    plt.hist(full_data, density=False, color=util.default[6], alpha=0.6, bins=bins, edgecolor='k', label=r"$\Gamma$, XRBs, estimates from $\beta_{\rm app}$ (Fender \& Motta, sub.)", hatch='//')
    plt.hist(francesco, density=False, color=util.default[6], alpha=1, bins=bins, edgecolor="k", label=r"$\Gamma_0$, XRBs, kinematic modelling (Carotenuto$+$ 2022,2024)")
    
    bin_centres = np.sqrt(bins[1:] * bins[:-1])

    plt.scatter(bin_centres[4],6.3,marker=">", color=util.default[6])
    plt.scatter(bin_centres[10],1.3,marker=">", color=util.default[6])
    plt.scatter(bin_centres[7],1.3,marker=">", color=util.default[6])
    plt.gca().set_xscale("log")
    plt.xlabel(r"$\log \Gamma$ or $\log \Gamma_0$", fontsize=20)
    plt.ylabel("Number of sources", fontsize=16)
    plt.legend()
    plt.xlim(1,1000)
    plt.ylim(0,9)
    util.add_shaded_bands(ax=plt.gca())
    plt.tight_layout(pad=0.05)
    util.save_paper_figure("lorentz.pdf", transparent=transparent)

if __name__ == "__main__":
    util.set_plot_defaults()
    make_figure()