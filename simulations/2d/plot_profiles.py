import numpy as np
import scipy
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from pluto_jm import pyPLUTO as pp
from astropy.table import Table
from constants import C
from astropy.io import ascii as io 
from tqdm import tqdm
from numba import jit 
import warnings
warnings.filterwarnings("ignore")

###returns internal energy in each cell
@jit
def taub_energy(pressure, rho, gamma):
    P = pressure
    c = 1
    first_term = (3*P)-(2*rho*(c**2))
    sqrt_term = (3*P - 2*rho*(c**2))**2 + (12*P*rho*(c**2))
    u = 0.5*(first_term + np.sqrt(sqrt_term))
    return (((gamma**2)*u) + ( ((gamma**2)-1.) * P) )

# define functions
call_K = lambda rho, gm : gm * (gm - 1.) * rho
call_gm = lambda vsq : 1. / np.sqrt(1.-vsq)
call_epsilon = lambda prs, rho: ((3. * prs - 2. * rho) + np.sqrt((3. * prs - 2. * rho)**2. + 12. * rho * prs))/(2.)


from sim_util import * 

def get_shock_radii_and_properties(D, data, i):
    """
    Calculates and updates shock radii and various physical properties for a pload object 
    at a given timestamp.

    This function computes key physical quantities related to shock waves, such as internal energy,
    kinetic energy, mass, and luminosities, and stores these values in the `data` dictionary at the 
    specified time index `i`. It also returns the computed internal energy, kinetic energy, and 
    conserved energy.

    Args:
        D (object): A pload object containing data arrays for the simulation, including pressure 
                    (`prs`), density (`rho`), Lorentz factor (`lorentz`), and spatial coordinates (`x1`).
        data (dict): A dictionary to store the computed shock radii and physical properties, indexed by time.
        i (int): The current time step index at which to store the calculated properties.

    Returns:
        tuple: A tuple containing the following arrays:
            - internal_energy (numpy.ndarray): The internal energy of the system.
            - kinetic_energy (numpy.ndarray): The kinetic energy of the system.
            - conserved_energy (numpy.ndarray): The conserved energy of the system.
    """
    internal_energy = taub_energy(D.prs, D.rho, D.lorentz)
    kinetic_energy = call_K(D.rho, D.lorentz)
    total_energy = kinetic_energy+internal_energy
    enthalpy = D.rho + D.prs + internal_energy
    conserved_energy = (D.lorentz * D.lorentz * enthalpy) - D.prs
    emiss = (D.prs ** 1.75) 
    xx,yy = np.meshgrid(D.x1, D.x2)
    xx = xx.T

    if np.sum(D.select_ism) > 0:
        data["r_forward"][i] = np.max(xx[D.select_ism]) * length_conversion
        data["ism_mass"][i] = np.sum(D.lorentz[D.select_ism] * D.rho[D.select_ism] * D.volume[D.select_ism])
        data["ism_energy"][i] = np.sum(total_energy[D.select_ism] * D.volume[D.select_ism])
        data["ism_rest"][i] = np.sum(D.rho[D.select_ism] * D.volume[D.select_ism])
        data["total_mass"][i] = np.sum(D.rho[D.select_ism_all] * D.volume[D.select_ism_all] * D.lorentz[D.select_ism_all])
        select_swept = (D.x1 <= (data["r_forward"][i]/length_conversion))
        data["swept_up"][i] = np.sum(D.rho[select_swept] * D.volume[select_swept] * D.lorentz[select_swept])

    data["r_reverse"][i] = np.max(xx[D.select_blob]) * length_conversion

    data["blob_mass"][i] = np.sum(D.lorentz[D.select_blob] * D.rho[D.select_blob] * D.volume[D.select_blob])
    data["blob_mass2"][i] = np.sum(D.rho[D.select_blob] * D.volume[D.select_blob])
    data["critical_mass"][i] = np.sum(D.rho[D.select_blob] * D.volume[D.select_blob] / np.max(D.lorentz))
    data["maxgamma"][i] = np.max(D.lorentz)
    data["total_mass"][i] = np.sum(D.lorentz * D.rho * D.volume)

    data["total_energy"][i] = np.sum(total_energy * D.volume)
    data["blob_energy"][i] = np.sum(total_energy[D.select_blob] * D.volume[D.select_blob])
    data["blob_kinetic"][i] = np.sum(kinetic_energy[D.select_blob] * D.volume[D.select_blob])
    data["total_lum"][i] = np.sum(emiss * D.volume)
    data["reverse_lum"][i] = np.sum(emiss * D.volume * D.tr1)
    data["reverse_lum2"][i] = np.sum(emiss[D.select_blob] * D.volume[D.select_blob])
    data["reverse_lum2"][i] = np.sum(emiss[D.select_blob2] * D.volume[D.select_blob2])

    return (internal_energy, kinetic_energy, conserved_energy)

def init_data_table(nrange):
    """
    Initializes a data table to store various physical quantities over a range of time steps.

    This function creates and returns an astropy `Table` object initialized with zeroed arrays 
    for various physical quantities related to a simulation of a blast wave. The time array 
    is calculated based on the provided range of steps (`nrange`) and is converted to days.


    Args:
        nrange (array-like): An array or list of integers representing time steps in the simulation.

    Returns:
        data: astropy.table.Table
            A table containing initialized arrays for time and various physical quantities 
            (e.g., mass, energy, shock positions, luminosities). The time is stored in days, 
            and all other quantities are initialized to zero.
    """
    ndo = len(nrange)
    data = Table() 
    data["time"] = nrange * UNIT_TIME * 10.0 / (86400.0)
    for key in ["blob_mass", "ism_mass", "r_forward", "r_reverse", "total_mass", 
                "E_blob", "E_ism", "blob_energy", "ism_energy", "total_energy", 
                "ism_rest","total_lum","reverse_lum","reverse_lum2","reverse_lum3", 
                "blob_mass2", "critical_mass", "maxgamma", "blob_kinetic", "swept_up"]:
        data[key] = np.zeros(ndo)
    return (data)

def read_pluto_sim(nn, wdir, datatype="float", printout=False):
    """
    Read a pluto simulation from wdir output folder and 
    calculate things like volume and Lorentz factors (and add to class),
    also defines some Boolean arrays for blob and ISM selection.

    Args:
        nn (int): timestep to read
        wdir (str): output directory of simulation to match pload kwarg
        datatype (str, optional): data type of simulation to match pload kwarg. Defaults to "float".

    Returns:
        D: Pluto pload object containing sim variables
    """
    D = pp.pload(int(nn),w_dir=wdir, datatype=datatype, printout=printout) 
    ddx = np.diff(D.x1)[0]
    D.xx, D.yy = np.meshgrid(D.x1, D.x2)
    D.xx = D.xx.T 
    D.yy = D.yy.T
    D.volume = 4.0 * np.pi * D.xx * D.xx * ddx
    D.lorentz = call_gm(D.vx1 * D.vx1)
    D.select_ism = (D.prs > 1e-7) * (D.tr1 < 1e-2)
    D.select_ism_all = (D.vx1 > 1e-5) 
    D.select_blob = (D.tr1 > 1e-2)
    D.select_blob2 = (D.tr1 > 1e-3)
    return (D)


def make_summary_plot(data, D, i, wdir, gamma0=2.5):
    """
    Generates and saves a summary plot of various profiles and shock wave properties.

    This function creates a multi-panel plot that visualizes several key characteristics of 
    a blast wave simulation at a given time step. The plots include:
    
    1. Pressure, Density, and Tracer profiles normalized by the shock position.
    2. Lorentz factor profile normalized by the shock position.
    3. Shock positions plotted on both logarithmic and linear scales.
    4. Swept-up mass over time, including blob mass and interstellar medium (ISM) mass.
    5. Energetic properties over time, including blob energy, ISM energy, ISM rest energy, 
       and total energy.

   Args:
        data : dict
            A dictionary containing time series data for various physical quantities related to 
            the shock wave, such as positions, masses, and energies.
        D : object
            An object that contains data arrays for pressure (prs), density (rho), tracer (tr1), 
            and Lorentz factor (lorentz), along with spatial information (x1).
        i : int
            The current index in the time series, indicating the time step to plot.
        gamma0: float
            initial Lorentz factor

    Returns:
        None
            The function saves the generated plot as a PNG file in the specified directory.

    """
    shock_position = data["r_forward"][i] 
    r_norm = D.x1 * length_conversion / shock_position
    plt.figure(figsize=(6,10))
    plt.subplot(323)
    plt.title("Profiles in blast wave")
    plt.plot(r_norm, D.prs, label="Pressure")
    plt.plot(r_norm, D.rho, label="Density")
    plt.plot(r_norm, D.tr1, label="Tracer")
    plt.legend()
    plt.xlim(0,1.1)
    plt.ylim(1e-5,20)
    plt.xlabel("$R/R_{f}$")
    plt.semilogy()

    plt.subplot(324)
    plt.title("Lorentz factor")
    plt.plot(r_norm, D.lorentz, label="Gamma")
    plt.legend()
    plt.xlim(0,1.1)
    plt.ylim(1,3.1)
    plt.xlabel("$R/R_{f}$")

    tt = np.logspace(-4,4,1000)
    plt.subplot(321)
    plt.title("Shock positions, log")
    plt.plot(data["time"][:i+1], data["r_reverse"][:i+1])
    plt.scatter(data["time"][i], data["r_reverse"][i])
    plt.plot(data["time"][:i+1], data["r_forward"][:i+1])
    plt.scatter(data["time"][i], data["r_forward"][i])
    plt.plot(tt, tt,ls="--", c="k")
    plt.plot(tt, 100.0 * tt**0.4,ls=":", c="k")
    plt.xlim(10,2000)
    plt.ylim(10,2000)
    plt.loglog()

    plt.subplot(322)
    plt.title("Shock positions, linear")
    plt.plot(data["time"][:i+1], data["r_reverse"][:i+1])
    plt.scatter(data["time"][i], data["r_reverse"][i])
    plt.plot(data["time"][:i+1], data["r_forward"][:i+1])
    plt.scatter(data["time"][i], data["r_forward"][i])

    plt.plot(tt, tt,ls="--", c="k")
    plt.plot(tt, 100.0 * tt**0.4,ls=":", c="k")
    plt.xlim(0,2000)
    plt.ylim(0,2000)

    plt.subplot(325)
    plt.title("Swept up mass")
    plt.gca().axhline(data["blob_mass"][0], ls="--", c="k")
    plt.gca().axhline(data["blob_mass"][0]/gamma0, ls="--", c="k")

    for key in ["blob_mass", "ism_mass"]:
        plt.plot(data["time"][:i+1], data[key][:i+1], label=key)
        plt.scatter(data["time"][i], data[key][i])
    plt.loglog()

    plt.subplot(326)
    plt.title("Energetics")
    for key in ["blob_energy", "ism_energy", "ism_rest", "total_energy"]:
        plt.plot(data["time"][:i+1], data[key][:i+1], label=key)
        plt.scatter(data["time"][i], data[key][i])
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/profile{:03d}.png".format(wdir, int(i)))
    plt.close("all")

def plot_and_save_profiles(wdir, nrange, load=False, gamma0=2.5, plot=True):
    """
    Generates and saves profile plots for a range of time steps in a simulation.

    This function processes simulation data over a specified range of time steps, calculates 
    shock radii and other properties if needed, and generates summary plots for each time step.
    The plots are then saved to the specified working directory.

    Args:
        wdir (str): The working directory containing the simulation data files.
        nrange (array-like): The range of time steps to process and plot. If `load` is False, this 
                             determines the range for which data will be generated and stored.
        load (bool, optional): If True, loads pre-existing data from a file instead of generating it 
                               anew. Defaults to False.

    Returns:
        None: The function saves the generated plots to the specified working directory but does not 
              return any value.
    """

    nlinf = pp.nlast_info(w_dir=wdir, datatype="float")
    nlast = nlinf["nlast"]
    nrange = np.arange(0,nlast+1,1)
    ndo = len(nrange)
    if load:
        data = io.read('data_{}.dat'.format(wdir[:-1]))
    else:
        data = init_data_table(nrange)

    for i,nn in enumerate(tqdm(nrange)):
        D = read_pluto_sim(nn, wdir, datatype="float")

        if load == False:
            _ = get_shock_radii_and_properties(D, data, i)

        if plot:
            make_summary_plot(data, D, i, wdir, gamma0=gamma0)

    if load == False:
        io.write(data, 'data_{}.dat'.format(wdir[:-1]), overwrite=True) 


gmm = [2,3,6,9]
rho = [10,20]
rho = [10,100]
load = False
nrange = np.arange(0,400,1)
# for g in gmm:
#     for r in rho:
#         wdir = 'g{:d}_rho{:d}/'.format(int(g), int(r))
#         print ("Plotting and saving profiles for {}".format(wdir))
#         plot_and_save_profiles(wdir, nrange, load=False, gamma0=g)

print ("Plotting and saving profiles for 2D conical sims")
#plot_and_save_profiles("test_katie_analogue/", nrange, load=False, gamma0=2.5, plot=False)  
#plot_and_save_profiles("2d_conical_chi100_phi1/", nrange, load=False, gamma0=2.5, plot=False)      
#plot_and_save_profiles("2d_conical_chi100_phi4/", nrange, load=False, gamma0=2.5, plot=False)   
#plot_and_save_profiles("2d_conical_chi100_phi90/", nrange, load=False, gamma0=2.5, plot=False)  
# plot_and_save_profiles("2d_conical_fiducial_phi90/", nrange, load=False, gamma0=2.5, plot=False)  
# plot_and_save_profiles("2d_conical_fiducial_phi1/", nrange, load=False, gamma0=2.5, plot=False)  
# plot_and_save_profiles("2d_conical_fiducial_phi4/", nrange, load=False, gamma0=2.5, plot=False)  
plot_and_save_profiles("2d_conical_fiducial_phi8/", nrange, load=False, gamma0=2.5, plot=False)  
#plot_and_save_profiles("out_gmm1.2/", nrange, load=False, gamma0=1.2, plot=False)   
#plot_and_save_profiles("katie_gmm5/", nrange, load=False, gamma0=5, plot=True)        
            