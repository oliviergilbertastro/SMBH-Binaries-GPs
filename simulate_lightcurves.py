# 1
import numpy as np
from celerite.terms import Matern32Term, RealTerm, JitterTerm
from mind_the_gaps.models.celerite_models import Lorentzian
from mind_the_gaps.simulator import Simulator
from mind_the_gaps.lightcurves import GappyLightcurve
import matplotlib.pyplot as plt
from mind_the_gaps.gpmodelling import GPModelling
from mind_the_gaps.stats import aicc
from scipy.stats import norm, ks_1samp
from random import gauss
import corner
plt.rcParams['figure.figsize'] = [16, 8]


np.random.seed(10)

def simulate_lc(model="DRW", savename=None, P_qpo=25,mean=100,P_drw=100,Q=50, sigma_noise=1, timerange=3000, length=500, time_sigma=0.2, exposure_time=1):
    """
    model : string (either DRW or DRW+QPO), chooses the model
    savename : string, name for saving the numpy array in the save directory
    P_qpo : float/int, period of the QPO in seconds (or in days if everything else is in days too)
    P_drw : float/int, period of the DRW in seconds (or in days if everything else is in days too)
    Q : float/int, coherence of the qpo
    sigma_noise : float/int, standard deviation for the gaussian noise added to the lightcurve. Also used for the error bars.
    timerange : int, total length of the lightcurve you want to simulate in seconds (or in days if everything else is in days too)
    length : int, number of datapoints which you want to simulate. They will approximately be separated by timerange/length time units.
    time_sigma : float/int, standard deviation of the gaussian from which to sample the irregular times.
    exposure_time : float/int, exposure time of each individual datapoint in seconds (or in days if everything else is in days too)
    """
    if model == "DRW+QPO" or model == "QPO+DRW":
        header=f"model={model}, P_qpo={P_qpo}, mean={mean}, P_drw={P_drw}, Q={Q}, sigma_noise={sigma_noise}, timerange={timerange}, length={length}, time_sigma={time_sigma}"
    elif model == "DRW":
        header=f"model={model}, mean={mean}, P_drw={P_drw}, Q={Q}, sigma_noise={sigma_noise}, timerange={timerange}, length={length}, time_sigma={time_sigma}"
    else:
        raise ValueError(f'Model "{model}" is not a valid model. Use either "DRW" or "DRW+QPO".')
    # 2

    # First method : regularly sampled
    times  = np.arange(0, 1000)

    # Second method : irregularly sampled, but always exactly at the same time
    times = np.random.choice(np.arange(0, timerange), size=length, replace=False)
    times = list(times)
    times.sort()
    times = np.array(times)

    # Third method : irregularly sampled in a gaussian way
    times = [0]
    for i in range(length-1):
        delta_time = np.abs(gauss(timerange/length, time_sigma))
        times.append(times[-1]+delta_time)
    times = np.array(times)


    exposure = exposure_time#np.diff(times)[0]

    P_qpo = P_qpo # period of the QPO
    w = 2 * np.pi / P_qpo
    mean = mean
    rms = 0.1
    variance_drw = (mean * rms) ** 2  # variance of the DRW (bending powerlaw)
    P_drw = P_drw
    w_bend = 2 * np.pi / P_drw # angular frequency of the DRW or Bending Powerlaw
    # Define starting parameters
    log_variance_qpo = np.log(variance_drw)
    log_sigma_matern = np.log(np.sqrt(variance_drw))
    P_matern = 10
    log_rho_matern =  np.log(P_matern / 2 / np.pi)
    Q = Q # coherence
    log_Q = np.log(Q)
    log_d = np.log(w)


    # Let's act as if everything is in DAYS (times, period, exposures, etc.):


    print(f"log variance of the QPO: {log_variance_qpo:.2f}, log_Q: {log_Q:.2f}, log omega: {log_d:.2f}")

    if model == "DRW":
        kernel = RealTerm(log_a=np.log(variance_drw), log_c=np.log(w_bend))
    else:
        kernel = Lorentzian(log_S0=log_variance_qpo, log_Q=np.log(Q), log_omega0=log_d) + RealTerm(log_a=np.log(variance_drw), log_c=np.log(w_bend))

    psd_model = kernel.get_psd
    # create simulator object with Gaussian noise
    simulator = Simulator(psd_model, times, np.ones(len(times)) * exposure, mean, pdf="Gaussian", 
                        sigma_noise=sigma_noise, extension_factor = 2)

    # simulate noiseless count rates from the PSD, make the initial lightcurve 2 times as long as the original times
    countrates = simulator.generate_lightcurve()
    # add (Poisson) noise
    noisy_countrates, dy = simulator.add_noise(countrates)

    print("Simulated time range:", np.min(times), "→", np.max(times))
    print("Simulated timestamps:", simulator.sim_timestamps[0], "→", simulator.sim_timestamps[-1])

    print(noisy_countrates)
    print(dy)

    drw_array = np.array([times, noisy_countrates, dy, np.ones(len(times)) * exposure]).T
    if savename is None:
        savename = model
    np.savetxt(f"simulations/{savename}.txt", drw_array, header=header)


#simulate_lc(model="DRW+QPO", savename=f"DRW_QPO_{0}", P_qpo=100,mean=100,P_drw=100,Q=80, sigma_noise=1, timerange=1000, length=150, time_sigma=0.7)