import sys 
from constants import *
import numpy as np 
sys.path.append("../utilities")
import blast_wave_util 
params = blast_wave_util .get_fiducial_parameters()

def print_helper(key, value):
    print ("{}: {}".format(key, value))
E0 = params["E0"]
gamma0 = params["gamma0"]
n0 = params["n0"]
phi = params["phi"]

omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(phi)))
R0 = params["R0"] * blast_wave_util.UNIT_LENGTH
rho_ism = n0 * MPROT    
volume = omega * np.pi * R0 * R0 * R0 / 3.0

Eiso = 4.0 * np.pi * E0 / omega
Miso = Eiso / ((gamma0 - 1) * C * C)

#n0 = 2.223 * gamma0
#M0 = volume * n0 * MPROT

#E0 = (gamma0 - 1) * M0 * C * C 

M0 = E0 / ((gamma0 - 1) * C * C)
volume = omega * np.pi * R0 * R0 * R0 / 3.0
rho_blob = M0 / volume
n_blob = rho_blob / MPROT 

print_helper("M0", M0)
print_helper("Miso", Miso)
print_helper("E0", E0)
print_helper("Eiso", Eiso)
print_helper("gamma0", gamma0)
print_helper("n0", n0)
print_helper("phi", phi)
print_helper("omega", omega)
print_helper("R0", R0)
# print_helper("rho_ism", rho_ism)
print_helper("rho_blob", rho_blob)
print_helper("n_blob", n_blob / gamma0)
print_helper("MPROT", MPROT)
#print ("M0 is {} ".format(M0))
#print ("E0 is {} omega is {} n0  is {}".format(E0, omega, n0))

