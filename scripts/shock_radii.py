
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii as io 
import warnings
from utilities import util
from utilities import blast_wave_util
import blast_wave_util 
from blast_wave_util import UNIT_RHO, UNIT_VOLUME, C, UNIT_LENGTH, MPROT
warnings.filterwarnings("ignore")


def make_figure(transparent=False):
    util.set_ui_cycler("dark")
    # Get fiducial parameters for the blast wave
    params = blast_wave_util.get_fiducial_parameters()

    # Calculate the solid angle omega
    omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(params["phi"])))

    # Initial radius and number density
    R0 = params["R0"]
    n0 = params["n0"]

    data = io.read('{}/1d/data_out_fiducial.dat'.format(util.g_SimDir))

    Miso = data["blob_mass"][0] * UNIT_RHO * UNIT_VOLUME
    M0 = Miso * omega / 4.0 / np.pi
    Eiso = np.log10((params["gamma0"] - 1) * Miso * C * C)
    E0 = np.log10((params["gamma0"] - 1) * M0 * C * C)
    #print (10.0**E0, params["E0"])

    volume = R0 * R0 * R0 * omega * np.pi / 3.0 * UNIT_VOLUME
    M0_expected = params["gamma0"] * 2.2126212826013743 * MPROT * UNIT_VOLUME
    Miso_expected = M0_expected * 4.0 * np.pi / omega
    #print (M0, M0_expected, Miso, Miso_expected * 2.5)

    # Initialize the blast wave model
    blast = blast_wave_util.blast_wave(params["gamma0"], params["n0"], params["E0"], omega)

    # Calculate the characteristic length scale
    l = (3.0 * blast.E0 / C / C / blast.rho / omega) ** (1./3.)

    blast.set_radii(approx=True)
    i_cross = np.argmin(np.fabs(data["ism_mass"] - (data["blob_mass"][0]/params["gamma0"])))


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4))
    c0 = util.SEABORN["dark"][0]
    c1 = "#e1b12c"
    ax.plot(data["time"], ((data["r_forward"]*(C * 86400.0))-(R0 * UNIT_LENGTH))/l, label="Forward Shock", lw=3, c= c0)
    ax.plot(data["time"][:i_cross], ((data["r_reverse"][:i_cross]*(C * 86400.0))-(20.0 * UNIT_LENGTH))/l, label="Ejecta (Reverse shock proxy)", lw=3, c=c1)
    ax.scatter(data["time"][i_cross], ((data["r_reverse"][i_cross]*(C * 86400.0))-(20.0 * UNIT_LENGTH))/l, c=c1, label = "Point of RS crossing", zorder=3)
    ax.plot(data["time"][i_cross:], ((data["r_reverse"][i_cross:]*(C * 86400.0))-(20.0 * UNIT_LENGTH))/l, lw=3, c=c1, ls="--")

    ax.set_xlabel(r"$t$~(days)", fontsize=20)
    ax.set_ylabel("$R / l$", fontsize=20)
    ax.set_xlim(0,400)
    ax.set_ylim(0,1.5)
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    fig.tight_layout(pad=0.05)
    util.save_paper_figure("shock_radii.pdf", fig = fig, transparent=transparent)

if __name__ == "__main__":
    util.set_plot_defaults()
    make_figure(transparent=False)