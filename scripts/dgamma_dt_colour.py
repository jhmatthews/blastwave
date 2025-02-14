import numpy as np
import matplotlib.pyplot as plt
from utilities import util 

def make_figure(transparent=False):
    plt.rcParams["lines.linewidth"] = 4
    util.set_cmap_cycler("Dark2", N=8)
    util.set_ui_cycler("canada")

    gmm = np.logspace(0,3,1000)  
    beta = np.sqrt(1.0 - (1.0/gmm/gmm))

    dRdt1 = beta / (1.0 - beta)
    dRdt2 = 2.0 * gmm * gmm 


    costheta = np.cos(np.radians(60))
    dRdt4 = beta / (1.0 - beta * costheta)

    costheta = np.cos(np.radians(30.0))
    dRdt3 = beta / (1.0 - beta * costheta)

    costheta = np.cos(np.radians(3.0))
    dRdt3deg = beta / (1.0 - beta * costheta)

    costheta = np.cos(np.radians(90.0))
    dRdt5 = beta / (1.0 - beta * costheta)

    plt.figure(figsize=(6.5,5))
    #
    # plt.title("Approaching jets")
    #util.set_cmap_cycler("viridis_r", N=7)
    plt.plot(gmm, dRdt2, label=r"$2 \Gamma^2$", ls=(0, (3, 1, 1, 1)), c="k", zorder=5)

    # theta_mesh = np.degrees(np.arccos(costheta_mesh))
    #plt.pcolormesh(gmm_mesh, dRdt_mesh, theta_mesh, alpha=0.5)
    thetas = (0.25,1,3,10,30,60,90)
    mappable, colors, mappable.to_rgba = util.get_mappable(len(thetas)+1, vmin=0, vmax=1, cmap_name = "viridis", return_func = True)

    plt.plot(gmm, dRdt1, label=r"On-axis full ($\theta=0^\circ$)", c=colors[0])
    for i,theta in enumerate(thetas):
        theta_rad = np.radians(theta)
        costheta = np.cos(theta_rad)
        dRdt = beta / (1.0 - beta * costheta) 
        plt.plot(gmm, dRdt, label=r"$\theta={}^\circ$".format(theta), c=colors[i+1])

    ylims = (0.1,1e6)

    plt.fill_between([2,5],y1=ylims[0],y2=ylims[1],alpha=0.3, color="#e377c2")
    plt.fill_between([100,1000],y1=ylims[0],y2=ylims[1],alpha=0.2, color="C0")

    plt.ylabel(r"$c^{-1} dR/dt$", fontsize=20)
    plt.xlabel("$\Gamma$", fontsize=20)
    plt.loglog()
    plt.legend(frameon=True, loc="upper left", ncol=2, fontsize=14, handletextpad=0.6,
            labelspacing=0.2, columnspacing=1.0, handlelength=1.5)
    plt.xlim(1,1000)
    plt.ylim(ylims[0], ylims[1])
    plt.tight_layout(pad=0.05)
    util.save_paper_figure("dgamma_dt.pdf", transparent=transparent)

if __name__ == "__main__":
    util.set_plot_defaults()
    make_figure(transparent=False)  
