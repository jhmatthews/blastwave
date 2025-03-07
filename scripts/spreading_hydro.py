
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii as io 
import warnings
from utilities import util
from utilities import blast_wave_util
from blast_wave_util import C, UNIT_LENGTH
warnings.filterwarnings("ignore")

def make_figure(transparent=False):
    """
    Generates and saves a figure illustrating the forward shock radius over time for both 1D and 2D data of a blast wave.

    Steps:
    1. Retrieve fiducial parameters.
    2. Calculate solid angle omega.
    3. Initialize the blast wave model.
    4. Read 1D and 2D data.
    5. Calculate characteristic length scale.
    6. Set radii for the blast wave.
    7. Find index for ISM mass closest to blob mass divided by gamma0.
    8. Plot forward shock radius for 1D and 2D data.
    9. Highlight reverse shock crossing point.
    10. Set axis labels, limits, and add legend.
    11. Adjust layout and save figure as "spreading_hydro.pdf".
    """
    util.set_ui_cycler("dark")
    
    # Get fiducial parameters for the blast wave
    params = blast_wave_util.get_fiducial_parameters()

    # Calculate the solid angle omega
    omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(params["phi"])))

    # Initial radius and number density
    R0 = params["R0"]
    n0 = params["n0"]

    # Read 1D and 2D data
    data = io.read('{}/1d/data_out_fiducial.dat'.format(util.g_SimDir))

    data2 = dict()
    for phi in [1,4,8]:
        data2["phi{}".format(phi)] = io.read('{}/2d/data_2d_conical_fiducial_phi{}.dat'.format(util.g_SimDir, phi))

    # Initialize the blast wave model
    blast = blast_wave_util.blast_wave(params["gamma0"], params["n0"], params["E0"], omega)

    # Calculate the characteristic length scale
    l = (3.0 * blast.E0 / C / C / blast.rho / omega) ** (1./3.)

    # Set the radii for the blast wave
    blast.set_radii(approx=True)

    # Find the index where the ISM mass is closest to the blob mass divided by gamma0
    i_cross = np.argmin(np.fabs(data["ism_mass"] - (data["blob_mass"][0]/params["gamma0"])))

    # Define colors for the plot and create figure
    c0 = util.SEABORN["dark"][0]
    c1 = "#e1b12c"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))

    # Plot the forward shock radius for 2D data
    util.set_cmap_cycler("viridis", N=5)

    colors = ["C3"]
    for i,phi in enumerate([1]):
        ax.plot(data2["phi{}".format(phi)]["time"], ((data2["phi{}".format(phi)]["r_forward"]*(C * 86400.0))-(R0 * UNIT_LENGTH))/l, 
                label=r"2D, $\phi = {}^\circ$".format(phi), lw=3, c= colors[i])

    # Plot the forward shock radius for 1D data
    ax.plot(data["time"], ((data["r_forward"]*(C * 86400.0))-(R0 * UNIT_LENGTH))/l, label="1D (No Spreading)", lw=3, c= "k")

    # Highlight the point of reverse shock crossing
    ax.scatter(data["time"][i_cross], ((data["r_forward"][i_cross]*(C * 86400.0))-(20.0 * UNIT_LENGTH))/l, c="k", label = "Point of RS crossing", zorder=3)

    # Set axis labels and limits and add a legend
    ax.set_xlabel(r"$t_{\rm lab}$~(days)", fontsize=20)
    ax.set_ylabel(r"$R / l$, Forward shock", fontsize=20)
    ax.set_xlim(0,400)
    ax.set_ylim(0,2.2)
    ax.legend(frameon=False, fontsize=14, loc="upper left")

    # Adjust layout and save the figure
    fig.tight_layout(pad=0.05)
    util.save_paper_figure("spreading_hydro.pdf", transparent=transparent)

if __name__ == "__main__":
    util.set_plot_defaults()
    make_figure(transparent=False)