import numpy as np
from matplotlib import pyplot as plt
from utilities import util
from utilities import blast_wave_util
from utilities.blast_wave_util import MPROT, C, PARSEC
from astropy.table import Table
from tqdm import tqdm

        
def get_critical_times(gammas = np.logspace(0.1,3,100), angles = [60.0, 3.0], energies = [1e45,1e45]):
    """get critical timescales for a set of Lorentz factors, viewing angles, and energies

    Args:
        gammas (array-like, optional): array of gammas. Defaults to np.logspace(0.1,3,100).
        angles (list, optional): list of angles. Defaults to [60.0, 3.0].
        energies (list, optional): list of energies. Defaults to [1e45,1e51].

    Returns:
        _type_: _description_
    """

    params = blast_wave_util.get_fiducial_parameters()
    n0 = params["n0"]
    rho = n0 * MPROT
    phi = params["phi"]
    E0 = params["E0"]

    omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(phi)))
    dkpc = 3.1
    d_cm = dkpc * 1000.0 * PARSEC
    #phi = phi = np.degrees(np.sqrt(omega / np.pi))
    ##print (phi)

    tcrit_all  = []
    for i in angles:
        tcrit_all.append(dict())

    for j, theta_i in enumerate(angles):
        theta_rad = np.radians(theta_i)
        factor = (np.sin(theta_rad) / d_cm) * (3600.0 * 180.0 / np.pi)


        blast = blast_wave_util.blast_wave(2.5, n0, E0, omega)  
        blast.set_radii()

        for key in  blast.labels.keys():
            tcrit_all[j][key] = np.zeros_like(gammas)

        tcrit_all[j]["gamma"] = gammas

        if theta_i < 0.0:
            app_or_rec = 'rec'
            theta_i = -theta_i
        else:
            app_or_rec = 'app'

        for i, Gamma0 in enumerate(tqdm(gammas)):
            #print (i)
            logE0 = np.log10(energies[j])
            E0 = energies[j]
            times, ang_sep_true_app, distances = blast_wave_util.get_distance_over_time(Gamma0, logE0, theta_i, dkpc=dkpc, n0=n0, phi=phi, app_or_rec=app_or_rec)

            blast = blast_wave_util.blast_wave(Gamma0, n0, E0, omega)
            l = (3.0 * blast.E0 / C / C / blast.rho) ** (1./3.)
            blast.set_radii()

            #print (times, distances, blast.R_RS)
            tcrit = blast_wave_util.get_times_critical(blast, times - 55200, distances)
            for key in tcrit.keys():
                tcrit_all[j][key][i] = tcrit[key]

    return (tcrit_all)

def make_figure(load = True, transparent=False):
    """Make the figure for the critical timescales

    Args:
        load (bool, optional): Whether to load the data. Defaults to True.
        transparent (bool, optional): Whether to make the figure transparent. Defaults to False.
    """
    util.set_ui_cycler("british")
    gammas = np.logspace(0.01,3,100)
    if load:
        tcrit_xrb = Table.read("{}/xrb_tcrits.dat".format(util.g_DataDir), format="ascii.fixed_width_two_line")
        tcrit_grb = Table.read("{}/grb_tcrits.dat".format(util.g_DataDir), format="ascii.fixed_width_two_line")
        tcrit_reced = Table.read("{}/xrb_receding_tcrits.dat".format(util.g_DataDir), format="ascii.fixed_width_two_line")
    else:
        #tcrit_all = get_critical_times(gammas = gammas, angles = [60.0, 0.01], energies = [1e45,1e45])
        tcrit_all = get_critical_times(gammas = gammas, angles = [60.0, 0.01, -60.0], energies = [1e44,1e44,1e44])
        blast_wave_util.save_dict_to_astropy_table(tcrit_all[0], "{}/xrb_tcrits.dat".format(util.g_DataDir), format="ascii.fixed_width_two_line")
        blast_wave_util.save_dict_to_astropy_table(tcrit_all[1], "{}/grb_tcrits.dat".format(util.g_DataDir), format="ascii.fixed_width_two_line")
        blast_wave_util.save_dict_to_astropy_table(tcrit_all[2], "{}/xrb_receding_tcrits.dat".format(util.g_DataDir), format="ascii.fixed_width_two_line")
        tcrit_xrb = tcrit_all[0]
        tcrit_grb = tcrit_all[1]
        tcrit_reced = tcrit_all[2]

    rr = np.zeros_like(gammas)
    for i,g in enumerate(gammas):
        blast = blast_wave_util.blast_wave(gammas[i], 0.002, 1e51, 0.05)
        l = (3.0 * blast.E0 / C / C / blast.rho) ** (1./3.)
        blast.set_radii()
        rr[i] = blast.R_RS

    tt = rr / (16.0 * gammas * gammas * C) / (86400.0)
    #print (tt)
    plt.figure(figsize=(5,4))#
    #plt.plot(gammas, tcrit_all[0]["R_RS"]/tcrit_all[0]["R_sw"], label=r"$t_{\rm RS}/ t(l_\Omega)$, Off-axis")
    ##plt.plot(gammas, tcrit_all[1]["R_RS"]/tcrit_all[1]["R_sw"], label=r"$t_{\rm RS}/ t(l_\Omega)$, On-axis")
    #plt.plot(gammas, gammas**(-(2.0/3.0)) * 100)
    #plt.plot(gammas, (1.0-np.exp(gammas))*100)
    plt.plot(tcrit_xrb["gamma"], tcrit_xrb ["R_RS"], label=r"``XRB'': $10^{44}~{\rm erg}, 60^\circ$, Approaching", lw=3, c="C3")
    plt.plot(tcrit_reced["gamma"], tcrit_reced ["R_RS"], label=r"``XRB'': $10^{44}~{\rm erg}, 60^\circ$, Receding", lw=3, c="C3", ls="--")
    factor = 1e7 ** (1./3.)
    plt.plot(tcrit_xrb["gamma"], factor*tcrit_xrb ["R_RS"], label=r"``Far off-axis GRB'': $10^{51}~{\rm erg}, 60^\circ$", lw=3, c="C4")
    beta = np.sqrt(1.0 - (1.0/tcrit_xrb["gamma"]/tcrit_xrb["gamma"]))
    factor1 = 1.0 - beta * np.cos(np.radians(60))
    factor2 = 1.0 + beta * np.cos(np.radians(60))

    #plt.plot(tcrit_xrb["gamma"], tcrit_reced ["R_RS"] / tcrit_xrb ["R_RS"])
    #plt.plot(tcrit_xrb["gamma"], factor2 / factor1, ls =":")

    #print (tcrit_reced ["R_RS"]/tcrit_xrb ["R_RS"])
    #print (blast_wave_util.doppler(tcrit_reced["gamma"], np.radians(60)))
    select = (np.isinf(tcrit_grb ["R_RS"]) == False) * (np.isnan(tcrit_grb ["R_RS"]) == False) * (tcrit_grb ["R_RS"] > 0)
    #plt.plot(tcrit_grb["gamma"][select], 100.0*tcrit_grb ["R_RS"][select], label=r"``Off-axis GRB'': $10^{51}~{\rm erg}, 3^\circ$", lw=3, c="C4")

    #plt.plot(gammas, rr / (2.0 * gammas * gammas * C) / (86400.0) / (1e7 ** 0.3333), c="C6", lw=3, label=r"``On-axis XRB'': $10^{44}~{\rm erg}, 0^\circ$")
    plt.plot(gammas, rr / (2.0 * gammas * gammas * C) / (86400.0), c="C0", lw=3, label=r"``On-axis GRB'': $10^{51}~{\rm erg}, 0^\circ$")
    plt.xlabel(r"$\Gamma_0$", fontsize=20)
    plt.ylabel(r"$t_{\rm RS}$ (days)", fontsize=20)

    # plt.fill_between([1.5,8],y1=0.001,y2=10000,alpha=0.3, color=util.default[6])
    # plt.fill_between([100,1000],y1=0.001,y2=10000,alpha=0.2, color=util.default[0])

    plt.grid(ls=":")
    plt.ylim(0.001,9999)
    plt.xlim(1,1000)
    util.add_shaded_bands(ax=plt.gca())
    plt.legend(fontsize=10)
    plt.loglog()
    plt.tight_layout(pad=0.05)
    util.save_paper_figure("times.pdf", transparent=transparent)

if __name__ == "__main__":
    util.set_plot_defaults()
    make_figure(load = False, transparent=False)