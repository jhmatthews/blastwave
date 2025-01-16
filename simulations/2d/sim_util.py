# simulation parameters
from constants import MPROT
import numpy as np 
import sys 
sys.path.append("../../utilities")
import blast_wave_util 
from blast_wave_util import UNIT_RHO, UNIT_V, UNIT_VOLUME, C, UNIT_LENGTH, PARSEC, UNIT_TIME, length_conversion


params = blast_wave_util.get_fiducial_parameters()
gamma0 = params["gamma0"]
phi = params["phi"]
omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(phi)))
R0 = params["R0"]
n0 = params["n0"]
rho_ism = n0 * blast_wave_util.UNIT_RHO
#n0 = rho_ism / 1.672e-24
theta_i = 90.0
R0cubed = (R0  * blast_wave_util.UNIT_LENGTH)**3

# Set up model directory
wdir = "katie_analogue/"