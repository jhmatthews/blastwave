import numpy as np
from matplotlib import pyplot as plt
import os, sys 
from scipy.interpolate import interp1d
from astropy.table import Table
import matplotlib.colors as mcolors
# Add the directory containing the util module to the Python path
UtilDir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(UtilDir)
import francesco, util 
#util
#util.set_mod_defaults()

UNIT_V = C = 2.998e+10
UNIT_RHO = 1.672661e-24
UNIT_LENGTH = 1.000e+15
UNIT_VOLUME = UNIT_LENGTH ** 3
length_conversion = UNIT_LENGTH / (C * 86400.0) # convert to light days
UNIT_TIME = 3.336e+04
pc_to_cm = 3.086e18
PARSEC = pc_to_cm
MPROT = 1.6726219e-24

def get_fiducial_parameters(fname = "{}/fiducial.txt".format(util.g_SimDir)):
    params = {}
    with open(fname, 'r') as f:
        for line in f:
            #print (line)
            key, value = line.split()
            params[key] = float(value)
    return params

def get_distance_over_time(Gamma0, E0, theta_i, 
                           phi=1.5, delta = 1.0, zeta=1.0, dkpc = 3.1, R_cavity_pc=1.0, 
                           R0 = 0.0, n0=0.002, t_ej = 0.0, tmax=1000, log_times=False, res=0.01, 
                           ntimes=1000, mass_swept_func=None, app_or_rec='app'):
    p=[Gamma0,E0,theta_i,phi,R_cavity_pc,delta,zeta,dkpc,R0,n0,t_ej]
    d_cm = dkpc * 1000.0 * PARSEC
    #times = np.arange(p[10]+55200,p[10]+55200+tmax,0.01)
    if log_times:
        times = np.logspace(np.log10(p[10]+55200), np.log10(p[10]+55200+tmax),ntimes)
    else:
        times = np.arange(p[10]+55200,p[10]+55200+tmax,res)

    #print (times, E0)
    # calculate the apparent angular distance over time 
    ang_sep_true_app=francesco.total_motion(p,times,app_or_rec, mass_swept_func=mass_swept_func)
    theta_rad = np.radians(theta_i)
    #factor = (np.sin(theta_rad) / d_cm) * (3600.0 * 180.0 / np.pi)
    # converts angular separation in arcsec to radians
    if theta_rad == 0:
        distance = ang_sep_true_app
    else:
        ang_sep_radians = ang_sep_true_app / (3600.0 * 180.0 / np.pi) 
        # converts angular distance in radians to physical distance in cm
        distance = ang_sep_radians * d_cm / np.sin(theta_rad)

    return (times, ang_sep_true_app, distance)

def save_dict_to_astropy_table(data_dict, filename, format="ascii.fixed_width_two_line"):
    """
    Create an Astropy table from a dictionary and save it to a file.

    Parameters:
        data_dict (dict): Dictionary where each key is a column name and each value is an array of column data.
        filename (str): The name of the file to save the table to.
    """
    # Create the Astropy table from the dictionary
    table = Table(data_dict)
    
    # Save the table to a file
    table.write(filename, format=format, overwrite=True)

def beta(gmm):
    return (np.sqrt(1.0 - (1.0/gmm/gmm)))

def doppler(gmm, theta):
    b = beta(gmm)
    doppler = 1.0 / (gmm * (1.0 - b * np.cos(theta)))
    return doppler

def delta_geometric(gmm, theta):
    b = beta(gmm)
    doppler = 1.0 / ((1.0 - b * np.cos(theta)))
    return doppler


def get_times_critical(blast, times, distances):
    tcrit = dict()
    #print (blast.Gamma0)
    interp_func = interp1d(distances, times, fill_value="extrapolate")
    for key in  blast.labels.keys():
        rcrit = blast.__dict__[key]
        #if blast.Gamma0 == 4.4887065280015115:
        #    print (np.fabs(distances - rcrit), distances, rcrit, key)
        #print (np.fabs(distances - rcrit), rcrit, key)
        #iarg = np.nanargmin(np.fabs(distances - rcrit))
        #tcrit[key] = times[iarg]
        tcrit[key] = interp_func(rcrit)

    return (tcrit)

def R_slow(Gamma0, n, E0, Gammacrit=1.118033988749895, sigma=0.7, Omega=np.pi/10, approx=False):
    rho = n * MPROT 
    # Convert to numpy arrays for element-wise operations
    Gamma0 = np.asarray(Gamma0)
    
    numerator = 3.0 * (Gamma0 - Gammacrit) * E0
    if approx:
        gsq = Gammacrit*Gammacrit
    else:
        gsq = gsh2(Gammacrit)
    denominator = sigma * Omega * rho * C * C * (Gamma0 - 1.0) * (gsq - 1.0)
    
    
    # Create a mask for Gamma0 values greater than Gammacrit
    mask = Gamma0 > Gammacrit
    
    # Calculate the result using numpy's vectorized operations
    result = np.zeros_like(Gamma0, dtype=np.float64)
    result[mask] = (numerator[mask] / denominator[mask]) ** (1.0 / 3.0)
    
    return result if result.shape else result.item()  # Return scalar if input was scalar


def R_RS(Gamma0, n, E0, Omega=np.pi/10):
    rho = n * MPROT 
    M0 = E0 / C / C / (Gamma0 - 1.0)
    R = (3.0 * M0 / Gamma0 / Omega / rho) ** (1./3.)
    return (R)

def R_sw(Gamma0, n, E0, Omega=np.pi/10,):
    rho = n * MPROT * np.ones_like(Gamma0)
    R = (3.0 * E0 / C / C / Omega / rho) ** (1./3.)
    return (R)

def l_Sedov(E0, n, phi):
    """
    Calculate the Sedov-Taylor radius for a given energy, particle number density, 
    Lorentz factor, and angle. Only works for small angles

    Args:
        E0 (float): The energy of the explosion in erg.
        n (float): The number density of particles in cm^-3.
        phi (float):  half-opening angle, degrees.

    Returns:
        float: The calculated radius in cm.
    """
    Omega = np.pi * (np.radians(phi)**2) # convert to solid angle
    rho = n * MPROT 
    R = (3.0 * E0 / C / C / Omega / rho) ** (1./3.)
    return (R)

def Rdiss(Gamma0, n, E0, f = 0.1, sigma=0.7, Omega=np.pi/10, approx=True):
    rho = n * MPROT
    numerator = 3.0 * (1.0-f) * E0
    denominator_term1 = sigma * Omega * rho * C * C 
    if approx:
        gsh2_minus_one = (f * f * ((Gamma0-1.0)**2)) + (2.0 * f * (Gamma0-1.0))
    else:
        gmm = f * (Gamma0 - 1.0) + 1
        gsh2_minus_one = gsh2(gmm) - 1.0
        
    x = (numerator / (denominator_term1 * gsh2_minus_one)) **(1./3.)
    return (x)

def adiab(Gamma):
    """
    Calculate the adiabatic index for a given specific heat ratio (Gamma).

    Args:
        Gamma (float): The specific heat ratio (Cp/Cv) of the gas.

    Returns:
        float: The adiabatic index.
    """
    return ( ((4.0 * Gamma) + 1) / (3.0 * Gamma))

def gsh2(Gamma):
    g = adiab(Gamma)
    numerator = (Gamma + 1.0) * (g * (Gamma - 1) + 1) ** 2
    denom = (g * (2.0 - g) * (Gamma - 1)) + 2
    return (numerator/denom)

def r_spread(Gamma0, n, E0, Omega=np.pi/10, approx=True, sigma=0.7):
    phi = np.arccos(1.0 - (Omega / 2.0 / np.pi))
    Gammacrit = 1.0 / phi
    #print (Gammacrit)
    R = R_slow(Gamma0, n, E0, Gammacrit=Gammacrit, sigma=sigma, Omega=Omega, approx=approx)
    return (R)

class blast_wave:
    """
    A class to represent a blast wave and calculate various radii associated with it.

    Attributes
    ----------
    E0 : float
        Initial energy of the blast wave in erg/s.
    Gamma0 : float
        Initial Lorentz factor of the blast wave.
    n : float
        Number density of the ambient medium in particles/cm3.
    Omega : float
        Solid angle of the blast wave.
    rho : float
        Density of the ambient medium  in g/cm3.
    labels : dict
        Dictionary containing labels for various radii.

    Methods
    -------
    set_radii(approx=True, sigma=0.7):
        Calculates and sets various radii associated with the blast wave.
    """
    def __init__(self, Gamma0, n, E0, Omega):
        self.E0 = E0 
        self.Gamma0 = Gamma0
        self.n = n 
        self.Omega = Omega
        self.rho = n * MPROT

        self.labels = dict() 
        self.labels["R_RS"] = r"R_{\rm RS}"
        self.labels["Rdiss0p5"] = r"R_{E}~(f_E=0.5)"
        self.labels["Rdiss0p1"] = r"R_{E}~(f_E=0.1)"
        self.labels["R_sw"] =r"R_{\rm sw}"
        self.labels["R_slow"] = r"R_{\rm slow}"
        self.labels["R_spread"] = r"R_{\phi}"

    def set_radii(self, approx=True, sigma=0.7):
        self.Rdiss0p1 = Rdiss(self.Gamma0, self.n, self.E0, f = 0.1, sigma=sigma, Omega=self.Omega, approx=approx)
        self.Rdiss0p5 = Rdiss(self.Gamma0, self.n, self.E0, f = 0.5, sigma=sigma, Omega=self.Omega, approx=approx)
        self.R_sw = R_sw(self.Gamma0, self.n, self.E0, Omega=self.Omega)
        self.R_RS = R_RS(self.Gamma0, self.n, self.E0, Omega=self.Omega)
        self.R_slow = R_slow(self.Gamma0, self.n, self.E0, Gammacrit=1.1, sigma=sigma, Omega=self.Omega, approx=approx)
        self.R_spread = r_spread(self.Gamma0, self.n, self.E0, Omega=self.Omega, approx=approx)

def doppler_time(time_array,theta,gmm):
    """
    Adjusts a given array of time values for Doppler shift based on the provided angle and Lorentz factor.

    Parameters:
        time_array (list or array-like): Array of time values in the rest frame.
        theta (float): Angle in radians between the direction of motion and the line of sight.
        gmm (float): Lorentz factor (gamma), which is a measure of relativistic effects.

    Returns:
        list: Array of time values adjusted for Doppler shift in the observer's frame.
    """
    delta_array = delta_geometric(gmm, theta) #used to use angle_to_boost() which is the relativistic correction
    cumulative_time_restframe = 0
    cumulative_time_obsframe = 0
    time_array_obsframe = []
    for index, t in enumerate(time_array):
        timestep_restframe = t-cumulative_time_restframe
        timestep_obsframe = timestep_restframe/delta_array[index]
        #print(timestep_obsframe)
        cumulative_time_obsframe = cumulative_time_obsframe + timestep_obsframe
        cumulative_time_restframe = cumulative_time_restframe+timestep_restframe
        time_array_obsframe.append(cumulative_time_obsframe)
    return time_array_obsframe