import numpy as np
from matplotlib import pyplot as plt
import jetsimpy
from constants import * 
import scipy.integrate as integrate
import astropy.io.ascii as io 
from scipy import stats
from utilities import util 
util.set_plot_defaults()


def make_figure(transparent=False):
    data = io.read("{}/grb_data.dat".format(util.g_DataDir))
    data2 = io.read("{}/xrb_data.dat".format(util.g_DataDir))
    bins = np.logspace(0,3.5,50)
    plt.figure(figsize=(10,4))

    francesco = [2.6, 1.6, 3.4, 1.85]

    select = data["Lower_Limit"] == "n"
    plt.hist(data["gamma"][select], density=False, color=util.default[0], alpha=0.6, bins=bins, edgecolor=util.default[0], label=r"$\Gamma_0$, GRBs, heterogenous methodologies (Table 1)")
    #plt.hist(np.log10(data["gamma"][~select]), density=True, color=util.default[0], alpha=0.6, bins=bins)
    kde = stats.gaussian_kde(np.log10(data["gamma"][select]))
    xx = np.linspace(0, 3.5, 300)
    #plt.plot(xx, kde(xx), color=util.default[0])
    #plt.fill_between(xx, y1=0, y2= kde(xx), color=util.default[0], alpha=0.5, label="GRBs")
    xx = np.logspace(1.5, 3, 100)
    plt.fill_between(xx, y1=0, y2=9, color=util.default[0], alpha=0.2, label=None)

    plt.hist(data2["Gamma"], density=False, color=util.default[6], alpha=0.5, bins=bins, edgecolor=util.default[6], label=r"$\Gamma$, XRBs, lower limits from $\beta_{\rm app}$ (Fender \& Motta, in prep)")
    plt.hist(francesco, density=False, color=util.default[6], alpha=1, bins=bins, edgecolor="k", label=r"$\Gamma_0$, XRBs, kinematic modelling (Carotenuto 2022,2024)")
    plt.gca().set_xscale("log")
    kde = stats.gaussian_kde(np.log10(data2["Gamma"]))
    xx = np.linspace(0, 3.5, 300)
    #plt.fill_between(xx, y1=0, y2= kde(xx), color=util.default[6], alpha=0.5, label="XRBs")

    xx = np.logspace(np.log10(1.5), np.log10(5), 100)
    plt.fill_between(xx, y1=0, y2=9, color=util.default[6], alpha=0.2, label=None, zorder=0)

    plt.xlabel("$\log \Gamma$ or $\log \Gamma_0$", fontsize=20)
    plt.ylabel("Number of sources", fontsize=16)
    plt.legend()
    plt.xlim(1,3000)
    plt.ylim(0,9)
    plt.tight_layout(pad=0.05)
    util.save_paper_figure("lorentz.pdf", transparent=transparent)

if __name__ == "__main__":
    make_figure()