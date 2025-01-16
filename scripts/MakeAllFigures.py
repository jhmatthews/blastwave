from utilities import util
import sys
import importlib
import numpy as np
import warnings
import matplotlib.pyplot as plt
np.seterr(divide='ignore')
warnings.filterwarnings("ignore")

print("Making all figures for Matthews et al. Blast Wave paper.")
tex = "True"
if len(sys.argv) > 1:
    if sys.argv[1] == "--notex":
        tex = "False"

print("Using tex?", tex)
util.set_plot_defaults(tex=tex)
print ("----------------------------")

list_of_modules = ["critical_radius", "critical_times", "lorentz", "dgamma_dt_colour", "spreading_hydro", 
                   "spreading_radius", "ang_sep", "shock_radii"]

for mod_name in list_of_modules:
    print("{}: importing and running...".format(mod_name), end="")
    module = importlib.import_module(mod_name)
    module.make_figure(transparent=False)
    plt.clf()
    print("Done.")
