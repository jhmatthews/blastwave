import numpy as np
import scipy
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from numba import jit  
#Computing the Lorentz factor (squared) of the swept up material, with an interpolated adiabatic index
@jit
def Gamma_sh_square(Gamma):
    adiab_index = (4*Gamma+1)/(3*Gamma)
    return ((Gamma+1)*(adiab_index*(Gamma-1)+1)**2)/(adiab_index*(2-adiab_index)*(Gamma-1) +2)

#Simple v/c of the jet from the Lorentz factor
#@jit
def beta(Gamma):
    #if loop prevents beta from being < 0.
    if Gamma >1:
        return np.sqrt(1-1./Gamma**2)
    else:
        return(0.)
    
# two parts of the mass_swept_cavity function
@jit
def R_bubble2(phi_jet,R_pc,n_ism):
    return((phi_jet)**2 * (n_ism)  * R_pc**3)

@jit
def R_bubble(phi_jet,R_pc,delta,n_ism):
    return((phi_jet)**2 * (n_ism/delta)  * R_pc**3)
@jit
def R_ism(phi_jet,R_pc,delta,zeta,R_cavity_pc,n_ism):
    return((phi_jet)**2 * (n_ism/delta)  *((zeta*R_cavity_pc)**3 + delta*(R_pc**3 - (zeta*R_cavity_pc)**3)))

#The growing mass swept up by the jet, with the scaling factor absorbing the large numbers.
#Before reaching the cavity radius R_cavity_pc, the jet moves in a an "empty" bubble with a density n_ism/delta.
#After reaching R_cavity_pc (the cavity wall), we sum the mass swept up so far with the mass swept up in a normal n_ism ISM, after a jump of order delta in the ISM density
def mass_swept_cavity_old(R_pc, R_cavity_pc, delta,zeta0,phi_jet,n_ism,flag, R0=20.0):
    c=3e10
    m_p = 1.672e-24
    pc_to_cm = 3.086e18
    R0cubed = (phi_jet)**2 * n_ism * ((R0 * 1.000e+15 / pc_to_cm)**3)
    #n_ism = 1e-3
    scaling_factor = (m_p*(c**2)*(np.pi/3)*pc_to_cm**3)
    if flag=='app':
        zeta=1.
    elif flag=='rec':
        zeta=zeta0
    R_term = R_bubble2(phi_jet,R_pc,n_ism)-R0cubed
    #print (R0cubed, R_term)
    return scaling_factor*R_term


def mass_swept_cavity(R_pc, R_cavity_pc, delta,zeta0,phi_jet,n_ism,flag):
    c=3e10
    m_p = 1.672e-24
    pc_to_cm = 3.086e18
    #n_ism = 1e-3
    scaling_factor = (m_p*(c**2)*(np.pi/3)*pc_to_cm**3)
    if flag=='app':
        zeta=1.
    elif flag=='rec':
        zeta=zeta0
    R_term = np.where(R_pc < zeta*R_cavity_pc, R_bubble(phi_jet,R_pc,delta,n_ism), R_ism(phi_jet,R_pc,delta,zeta,R_cavity_pc,n_ism))
    return scaling_factor*R_term


def mass_swept_cavity2(R_pc,zeta0,phi_jet,n_ism,flag, R0=20.0):
    c=3e10
    m_p = 1.672e-24
    pc_to_cm = 3.086e18
    #n_ism = 1e-3
    R0cubed = (R0 * 1.000e+15 / pc_to_cm)**3
    scaling_factor = m_p*(c**2)*(np.pi/3)*(pc_to_cm**3)
    if flag=='app':
        zeta=1.
    elif flag=='rec':
        zeta=zeta0
    R_term = R_bubble2(phi_jet,R_pc,n_ism)
    return scaling_factor*R_term

#def mass_swept_cavity_sim(interp_func, R_pc,zeta0,phi_jet,n_ism,flag):



#In the model we assume energy conservation (adiabatic expansion). This function is Equation 2 from Steiner & McClintock. 2012.
#M_0 is obtained from E_0/(Gamma0-1)c**2. This function is used through fsolve to obtain the Lorentz factor Gamma that satisfies energy conservation, returning 0.
#E0_45 is E0 in units of 10^45 erg, R_pc and R_cavity_pc are in units of parsecs.
def energy_cavity(Gamma, params, mass_swept_func = None):
    R_pc, Gamma0, E0_45, R_cavity_pc, delta,zeta,phi_jet,n_ism,flag= params
    if mass_swept_func == None:
        m_sw = mass_swept_cavity(R_pc, R_cavity_pc, delta,zeta,phi_jet,n_ism,flag)
    else:
        m_sw = mass_swept_func(R_pc, R_cavity_pc, delta,zeta,phi_jet,n_ism,flag)

    return -10**(E0_45) + ((Gamma-1)/(Gamma0-1))*10**(E0_45) + (0.73-0.38*beta(Gamma)) * (Gamma_sh_square(Gamma) - 1)*m_sw


#Function for the numerical solution of the "energy conservation constraint".
#It is used to obtain the Lorentz factor at every time step of the integration of the differential equation governing the jet motion.
#The first guess for Gamma is Gamma0, with no values below 1.001 (I first saw that it was helping for stability, but probably it is not really necessary)
#@jit
def Gamma_t(R_pc, parameters, mass_swept_func=None):
    Gamma0, E0_45, R_cavity_pc, delta,zeta,phi_jet,n_ism,flag = parameters
    pars = [R_pc, Gamma0, E0_45, R_cavity_pc, delta,zeta,phi_jet,n_ism,flag]
    Gamma_new = fsolve(energy_cavity, Gamma0, args=(pars, mass_swept_func),full_output=1)
    #if fsolve fails we set Gamma_new = Gamma0, and keep track of when a fail occurs (0=fail, 1=sucessful)
    if Gamma_new[2] == 1:
        return Gamma_new[0],1
    else:
        return Gamma0,0

#Computing the derivative of R(t), to be integrated with scipy.odeint. At every time step the Lorentz factor is obtained using the distance information (R_pc).
#We only test an approaching jet component, as we know that this is the case for MAXI J1348-630.
#Theta_i is the inclination angle.
def jet_motion_cavity(y, t, Gamma0, E0_45, theta_jet, R_cavity_pc, delta,zeta,phi_jet,n_ism,flag, mass_swept_func=None):
    #Conversion to pc, as y is in cm
    pc_to_cm = 3.086e18
    c=3e10
    R_pc = y/pc_to_cm
    pars_app = [Gamma0, E0_45, R_cavity_pc, delta,zeta,phi_jet,n_ism,flag]
    Gamma,sol = Gamma_t(R_pc, pars_app, mass_swept_func=mass_swept_func)
    #Equation 4 of Steiner & McClintock. 2012
    if sol==1 and flag=='app':
        #print (R_pc, pars_app)
        dR_dt = (beta(Gamma)*c/(1-beta(Gamma)*np.cos(theta_jet)))
    elif sol==1 and flag=='rec':
        dR_dt = (beta(Gamma)*c/(1+beta(Gamma)*np.cos(theta_jet)))
    #if fsolve fails in Gamma_t function, we set derivative here to 0 to induce a warning in 
    #the ODE solver in total_motion function
    else:
        dR_dt=0.
    #We return cm per day
    return dR_dt*86400

#solve the differential equation in jet_motion_cavity function, output angular distance as function of time
#@jit
def total_motion(p,time_array0,flag,mass_swept_func=None):
    Gamma0, E0_45, theta_i,phi_j, R_cavity_pc, delta,zeta,D,R0,n_ism,tej = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10]
    d_cm=D * 1000 * 3.086e18
    theta_rad=theta_i*np.pi/180.
    phi_rad=phi_j*np.pi/180.
    #for time_array in absolute MJD, convert to time after ejection
    time_array=time_array0-(tej+55200)
    #add a 0 to beginning of time_array if needed for differential equation solver to work properly
    if time_array[0]!=0.:
        time_array=np.insert(time_array,0,0.)
        inde=1
    else:
        inde=0
    
    time_array_diff = np.diff(time_array)
    
    #Ensuring that values in the time array increase monotonically after the t_ej subtraction:
    #the proposal cannot bring a walker in a point of the parameter space where t_ej is > 58551
    if time_array_diff[0] > 0:
        
        #if fsolve fails, with dR_dt=0, odeint fails with ODSint warning. We turn this warning to an error
        #and catch in a try and accept statment, so we can set logprob=-inf at this point in parameter space.
        try:
            # hack to allow this to work for on axis case
            if theta_rad == 0:
                factor = 1.0
            else:
                factor = (np.sin(theta_rad) /d_cm)* (3600 * 180 / np.pi)
            alpha_app = odeint(jet_motion_cavity, t=time_array, y0=R0, 
                               args=(Gamma0, E0_45, theta_rad, R_cavity_pc, delta,zeta,phi_rad,n_ism,flag,mass_swept_func)
                              )* factor
        except scipy.integrate.odepack.ODEintWarning:
            alpha_app=np.zeros(len(time_array))
            print ("warning!")

        return(alpha_app.flatten()[inde:])
    
    else:
        alpha_app=np.zeros(len(time_array))
        return(alpha_app.flatten()[inde:])
    

if __name__ == "__main__":
    import jm_util
    jm_util.set_mod_defaults()
    dkpc = 5.3
    R_cavity_pc = 1.0
    delta = 1.0
    zeta = 1.0
    phi = 1.5
    theta_i = 50.0
    E0 = np.log10(3.6e44)
    n0 = 1.5e-4
    R0 = 1e7
    t_ej = 0.0
    Gamma0 = 3.0
    p=[Gamma0,E0,theta_i,phi,R_cavity_pc,delta,zeta,dkpc,R0,n0,t_ej]
    times = np.arange(p[10]+55200,p[10]+55200+1000,1)
    ang_sep_true_app=total_motion(p,times,'app')
    plt.plot(times-55200, ang_sep_true_app, label="Carotenuto code")

    twang, ywang = np.genfromtxt("wang_data.dat", unpack=True)
    # sintheta = np.sin(np.radians(50.0))
    # ang_radians = np.radians(ywang / 3600.0)
    # Rwang =  ang_radians / sintheta * 5300.0 * PARSEC 
    # print (Rwang)
    plt.plot(twang, ywang, label="Wang+ 2003", ls="--")
    plt.legend()
    plt.xlabel("$t$~(Days)")
    plt.ylabel("Ang Sep. (arcsec)")
    #plt.xlim(0,600)
    plt.savefig("wang_comparison.png", dpi=100, tight_layout=True)