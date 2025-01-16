import numpy as np
from matplotlib import pyplot as plt
from utilities import util
from utilities import blast_wave_util

def make_figure(transparent=False):
    # Get fiducial parameters for the blast wave
    params = blast_wave_util.get_fiducial_parameters()


    gammas = np.logspace(0,3,1000)
    E0 = params["E0"]
    n = params["n0"]
    #rho = n * MPROT

    util.set_cmap_cycler("viridis", N=5)


    plt.rcParams["lines.linewidth"] = 3
    plt.figure(figsize=(5,4))

    phis = [8,4,2,1,0.5]
    for i, phi in enumerate(phis):
        omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(phi)))
        blast = blast_wave_util.blast_wave(gammas, n, E0, omega)
        blast.set_radii()
        plt.plot(gammas, blast.__dict__["R_spread"]/blast.__dict__["R_RS"], label=r"$\phi = {}^\circ$".format( phi), ls="-", c=f"C{i}")

    plt.gca().axhline(1, c="k", ls="--")
    plt.fill_between([2,5],y1=0.001,y2=10000,alpha=0.3, color=util.default[6])
    plt.fill_between([100,1000],y1=0.001,y2=10000,alpha=0.2, color=util.default[0])
    plt.loglog()
    plt.xlabel("$\Gamma_0$", fontsize=20)
    plt.ylabel(r"$R_\phi / R_{\rm RS}$", fontsize=20)
    plt.legend(ncol=1, fontsize=12)
    plt.xlim(1,1000)
    plt.ylim(6e-3,20)
    plt.tight_layout(pad=0.05)
    util.save_paper_figure("spreading_radius.pdf", transparent=transparent)

if __name__ == "__main__":
    util.set_plot_defaults()
    make_figure(transparent=False)