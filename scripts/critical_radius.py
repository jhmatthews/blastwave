import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from utilities import util
from utilities import blast_wave_util
from utilities.blast_wave_util import MPROT, C


def make_figure(transparent=False):
    gammas = np.logspace(0,3,1000)
    E0 = 1e45
    n = 1.0
    rho = n * MPROT
    omega = 1.0
    blast = blast_wave_util.blast_wave(gammas, n, E0, omega)
    blast.set_radii()


    # jm_util.set_cmap_cycler("RdYlBu_r", N=4)
    util.set_ui_cycler("canada")

    l = (3.0 * blast.E0 / C / C / blast.rho) ** (1./3.)

    plt.rcParams["lines.linewidth"] = 3
    plt.figure(figsize=(9,4))
    plt.subplot(121)
    #print (blast.__dict__)

    colors = ["C0", "C3", "C3", "C2", "C1"]
    ls = ["-", "--", "-", "-", "-"]
    for i, radius in enumerate(["R_RS", "Rdiss0p5", "Rdiss0p1", "R_slow"]):
        #print (blast.labels[radius])
        plt.plot(gammas, blast.__dict__[radius]/l, label=r"${}/l$".format(blast.labels[radius]), ls=ls[i], c=colors[i])


    # plt.fill_between([2,5],y1=6e-3,y2=20,alpha=0.3, color="#e377c2")
    plt.fill_between([2,5],y1=0.001,y2=10000,alpha=0.3, color=util.default[6])
    plt.fill_between([100,1000],y1=0.001,y2=10000,alpha=0.2, color=util.default[0])
    plt.loglog()
    plt.xlabel(r"$\Gamma_0$")
    plt.legend()
    plt.xlim(1,1000)
    plt.ylim(6e-3,20)

    plt.subplot(122)
    plt.fill_between([2,5],y1=0.001,y2=10000,alpha=0.3, color=util.default[6])
    plt.fill_between([100,1000],y1=0.001,y2=10000,alpha=0.2, color=util.default[0])
    plt.plot(gammas, blast.R_RS/blast.R_slow, label=r"$R_{\rm RS}/R_{\rm slow}$", c="C0", ls="-")
    plt.plot(gammas, blast.R_RS/blast.Rdiss0p1, label=r"$R_{\rm RS}/R_E~(f_E=0.1)$", c="C3", ls="-")
    plt.plot(gammas, blast.R_RS/blast.Rdiss0p5, label=r"$R_{\rm RS}/R_E~(f_E=0.5)$", c="C3", ls="--")
    plt.xlabel(r"$\Gamma_0$")
    plt.loglog()
    plt.legend()
    plt.xlim(1,1000)
    plt.ylim(6e-3,20)
    plt.tight_layout(pad=0.05)
    util.save_paper_figure("critical_radii.pdf", transparent=transparent)

if __name__ == "__main__":
    util.set_plot_defaults()
    make_figure(transparent=False)