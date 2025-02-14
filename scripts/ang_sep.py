import numpy as np
from matplotlib import pyplot as plt
from utilities import util
from utilities.blast_wave_util import MPROT, C

def R_RS(Gamma0, n, E0, Omega=np.pi/10):
    rho = n * MPROT 
    M0 = E0 / C / C / (Gamma0 - 1.0)
    R = (3.0 * M0 / Gamma0 / Omega / rho) ** (1./3.)
    return (R)

def ang_sep_RS(theta, Gamma0, n, E0, Omega=np.pi/10, D=3.0):
    Rsintheta = R_RS(Gamma0, n, E0, Omega) * np.sin(np.radians(theta))
    return (Rsintheta / D / (3.086e21))

def make_figure(transparent=False):
    phi = 1.0
    omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(phi)))
    gammas = np.logspace(0.02,2,200)
    thetas = np.linspace(0,90,100)

    gg, tt = np.meshgrid(gammas, thetas)

    alpha = np.degrees(ang_sep_RS(tt, gg, 0.002, 1e45, Omega=omega)) * 3600.0

    #print (R_RS(3, 0.002, 1e45, omega))

    #print (R_RS(2.5, 0.002, 1e45, omega) * np.sin(np.radians(60)) /  3.0 / (3.086e21))
    #print (R_RS(3, 0.002, 1e45, omega) * np.sin(np.radians(60)))

    #plt.pcolormesh(gg, tt, alpha)
    levels = np.array([0.1,0.3,1,3,10,30,100])

    levels = 10.0 ** np.arange(-2,2,0.25)
    #levels = np.array([0.1,0.3,1,3,10,30,100])

    fig = plt.figure(figsize=(5,4))
    CS1 = plt.contourf(gg, tt, alpha, levels = levels, cmap="Blues")
    cbar = plt.colorbar()

    #plt.contour(gg, tt, alpha, levels = levels, colors="k", linewidths=0.5)
    cbar.set_label(r"$\alpha_{\rm RS}$~(arcseconds)", fontsize=16, labelpad=-2)
    plt.xlabel(r"$\Gamma_0$", fontsize=20)
    plt.ylabel(r"$\theta~(^\circ)$", fontsize=20)
    plt.contour(gg, tt, alpha, levels = (5.4,), colors=util.default[3])
    #plt.clabel(CS1, CS1.levels, inline=True, fmt=fmt, fontsize=10)
    #plt.colorbar()
    #plt.legend()
    plt.text(5.5,60,"MeerKAT",rotation=82, color=util.default[3], fontsize=18)
    plt.semilogx()
    plt.tight_layout(pad=0.05)
    util.save_paper_figure("ang_sep_RS.pdf", fig = fig, transparent=transparent)

if __name__ == "__main__":
    util.set_plot_defaults()
    make_figure(transparent=False)