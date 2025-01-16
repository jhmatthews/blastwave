import numpy as np
import matplotlib.pyplot as plt

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

Gamma = np.logspace(-10,3,10000)
gsh = np.sqrt(gsh2(Gamma))
beta = np.sqrt(1 - 1/Gamma/Gamma)
betash = np.sqrt(1 - 1/gsh/gsh)
#plt.plot(Gamma, np.sqrt(gsh2(Gamma))/Gamma)
plt.plot(beta * Gamma, betash * gsh / (beta * Gamma))
plt.semilogx()
plt.show()