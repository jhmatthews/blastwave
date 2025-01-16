import numpy as np
from matplotlib import pyplot as plt
from utilities import util
from utilities import blast_wave_util
from utilities import  francesco 
import cmasher as cmr
from astropy.table import Table
from blast_wave_util import UNIT_RHO, UNIT_V, UNIT_VOLUME, C, UNIT_LENGTH, PARSEC

util.set_ui_cycler("british")
util.set_plot_defaults()

def make_figure(load = True, log=False):
    """
    Create and save a schematic plot illustrating the evolution of blast wave distance over time for different initial
    Lorentz factors and energies.

    Parameters:
    -----------
    load : bool, optional
        If True, loads precomputed data from files. If False, computes data from scratch. Default is True.
    log : bool, optional
        If True, plots the data on a logarithmic scale. Default is False.

    The function performs the following tasks:
    1. Initializes parameters for the blast wave including Lorentz factors, initial energy, density, and other
       constants.
    2. Sets up the plot with two subplots for different scenarios: XRB (Gamma ~ 2) and GRB (Gamma ~ 100).
    3. Depending on the `load` parameter, it either loads precomputed data from files or computes new data using
       blast wave utilities.
    4. Plots the blast wave distance normalized by a characteristic length scale over time.
    5. Adds visual elements like gamma color coding, critical times, and relevant labels and titles to the plot.
    6. Saves the resulting figure to a PDF file, either as a logarithmic or linear plot depending on the `log` parameter.

    Notes:
    ------
    - This function assumes that `blast_wave_util`, `Table`, `francesco`, `jm_util`, `plt`, `np`, `MPROT`, `PARSEC`, and `C`
      are predefined and imported appropriately.
    - Ensure the necessary data files are available in the current directory if `load` is set to True.
    - The function produces two different scenarios for visualizing the blast wave evolution: one for an XRB and one for
      a GRB.

    Example usage:
    --------------
    To create and save the schematic plot with default settings:
    >>> make_figure()
    
    To create and save the schematic plot with logarithmic scaling and recompute data:
    >>> make_figure(load=False, log=True)
    """

    # Get fiducial parameters for the blast wave
    params = blast_wave_util.get_fiducial_parameters()

    gammas = np.logspace(0,3,1000)
    
    n0 = params["n0"] 

    dkpc = 3.1
    d_cm = dkpc * 1000.0 * PARSEC


    phi = params["phi"]
    omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(params["phi"])))

    print ("PHI", phi)
    pc_to_cm = 3.086e18

    gammas = [params["gamma0"],100]
    angles = [60.0, 1e-10]
    E0s = [np.log10(params["E0"]),51.0]
    cmap = "Blues"
    cmaps = [cmap, cmap]
    tmaxes = [4000,100000]
    tmaxes = [100,100]
    titles = [r"XRB, $\Gamma_0 \approx 2$", r"GRB, $\Gamma_0 \approx 100$"]
    titles = [r"XRB", r"GRB"]

    fig, ax = plt.subplots(figsize=(8,4), ncols = 2)
    for i, Gamma0 in enumerate(gammas):
        print ("Making plot for gamma0 {}".format(Gamma0))
        E0 = E0s[i]
        theta_i = angles[i]
        theta_rad = np.radians(angles[i])

        data = dict()

        if load:
            data = Table.read("schematic_data_{:d}.dat".format(i), format="ascii.fixed_width_two_line")
        else:
            times, ang_sep_true_app, distance = blast_wave_util.get_distance_over_time(Gamma0, E0, theta_i, dkpc=dkpc, n0=n0, phi=phi, tmax=tmaxes[i], log_times=False, ntimes=10000)
            data["times"] = times - 55200
            data["distance"] = distance
            p = [Gamma0, E0, 1.0, 1.0, 1.0, np.radians(phi), n0, 'app']

            data["gammas"] = np.zeros_like(distance)
            for j, d in enumerate(distance):
                g, sol = francesco.Gamma_t(d/pc_to_cm, p)
                #print (g)
                if sol == 1:
                    data["gammas"][j] = g[0]
            #betagamma = np.sqrt(1-1./gammas_over_time**2) * gammas_over_time  
            blast_wave_util.save_dict_to_astropy_table(data, "schematic_data_{:d}.dat".format(i), format="ascii.fixed_width_two_line")

        blast = blast_wave_util.blast_wave(Gamma0, n0, 10.0**E0, omega)
        l = (3.0 * blast.E0 / C / C / blast.rho / omega) ** (1./3.)
        #print (blast.E0, E0)

        ax[i].plot(data["times"], data["distance"]/l, label="$\Gamma_0 = {}, $\theta={}$".format(Gamma0, theta_i), c="C{:d}".format(i), lw=4)


        blast.set_radii()
        tcrit = blast_wave_util.get_times_critical(blast, data["times"], data["distance"])


        betagamma = np.sqrt(1-1./data["gammas"]**2) * data["gammas"]

        # create gamma colour coding 
        tmax = tcrit["R_slow"] * 1.2
        if log: tmax = tmaxes[i]
        select = data["times"]<tmax
        polygon = ax[i].fill_between(data["times"][select], 0, data["distance"][select]/l, lw=0, color='none')
        xlim =  ax[i].get_xlim()
        ylim = ax[i].get_ylim()
        xlim = (xlim[0],tmax)
        verts = np.vstack([p.vertices for p in polygon.get_paths()])
        if log:
            gradient = ax[i].imshow(np.log10(betagamma[select][::2]).reshape(1, -1), cmap=cmr.fusion_r, aspect='auto', interpolation="gaussian",
                        extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()], vmax=1, vmin=-1, alpha=0.8)
        else:
            gradient = ax[i].imshow(np.log10(betagamma[select][::2]).reshape(1, -1), cmap=cmr.fusion_r, aspect='auto', interpolation="gaussian",
                        extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()], vmax=1, vmin=-1, alpha=0.8)
        gradient.set_clip_path(polygon.get_paths()[0], transform=ax[i].transData)
        print ("HERE")

        colours = [util.default[3], "k"]
        t0 = 0
        for j,key in enumerate(["R_RS","R_slow"]):
            xx = data["times"]
            select = (xx <= tcrit[key]) * (xx >= t0)
            xx = xx[select]

            ax[i].vlines([tcrit[key]], 0,  blast.__dict__[key]/l, color=colours[j], zorder=5, ls="--")
            ax[i].scatter(tcrit[key], blast.__dict__[key]/l, c=colours[j], zorder=5, s=80)
            #if j == 0:
             #   ax[i].fill_between(xx, y1=0, y2=data["distance"][select]/l, color=colours[j], alpha=0.4)
            t0 = tcrit[key]

        ax[i].set_xlim(0,np.min((tmaxes[i], tmax)))
        #if log:
            #if i == 0:
            #    ax[i].set_xlim(0,1000.0)
            #else:
            #    ax[i].set_xlim(0,90000.0)
        #plt.ylim(0,20)
        ax[i].set_ylim(0,2)
        ax[i].set_xlabel(r"$t~({\rm days})$", fontsize=20)
        if i == 0: 
            ax[i].set_ylabel(r"$R/l$", fontsize=20)

        #ax[i].set_title(titles[i])
        if log:
            if i == 1:
                ax[i].plot(data["times"], data["times"]**0.25, c="k", ls="--")
            ax[i].plot(data["times"], data["times"]**0.4, c="k", ls=":")
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_ylim(0.01,10)
            if i == 0:
                ax[i].set_xlim(1,3e3)
            else:
                ax[i].set_xlim(0,100000.0)
            #ax[i].set_xlim(0.01,np.min((tmaxes[i], tmax))*10)
        #ax[i].text(0.80,4,"Sedov-Taylor",rotation=90,ha="center", va="center")

    fig.tight_layout(pad=0.05)
    if log:
        fig.savefig("log_schematic.png", dpi=300)
        fig.savefig("log_schematic_transp.png", dpi=300, transparent=True)
    else:
        fig.savefig("schematic.png", dpi=300)
        fig.savefig("schematic_transp.png", dpi=300, transparent=True)

if __name__ == "__main__":
    make_figure(load = False, log=True)
    #make_figure(load = True, log=False)