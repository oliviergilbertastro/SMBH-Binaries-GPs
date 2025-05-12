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
import corner
plt.rcParams['figure.figsize'] = [16, 8]


np.random.seed(10)

def simulate_lc(P_qpo=25,mean=100,P_drw=100,Q=50, sigma_noise=1):
    # 2
    times  = np.arange(0, 1000)
    times = np.random.choice(np.arange(0, 3000), size=500, replace=False)
    times = list(times)
    times.sort()
    times = np.array(times)
    exposure = 1#np.diff(times)[0]

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
    print(f"log variance of the QPO: {log_variance_qpo:.2f}, log_Q: {log_Q:.2f}, log omega: {log_d:.2f}")

    labels = ["Lorentzian", "DRW"]
    # You can also use Lorentzian from models.celerite_models (which is defined in terms of variance, Q and omega)
    #kernel = Lorentzian(log_S0=log_variance_qpo, log_Q=np.log(Q), log_omega0=log_d) + RealTerm(log_a=np.log(variance_drw), log_c=np.log(w_bend))
    kernel = RealTerm(log_a=np.log(variance_drw), log_c=np.log(w_bend))
    truth = kernel.get_parameter_vector()
    psd_model = kernel.get_psd

    SIGMA_NOISE = sigma_noise
    # create simulator object with Gaussian noise
    simulator = Simulator(psd_model, times, np.ones(len(times)) * exposure, mean, pdf="Gaussian", 
                        sigma_noise=SIGMA_NOISE, extension_factor = 100)

    # simulate noiseless count rates from the PSD, make the initial lightcurve 2 times as long as the original times
    countrates = simulator.generate_lightcurve()
    # add (Poisson) noise
    noisy_countrates, dy = simulator.add_noise(countrates)

    drw_array = np.array([times, noisy_countrates, dy, np.ones(len(times)) * exposure]).T
    np.savetxt("simulations/DRW.txt", drw_array, header=f"P_qpo={P_qpo},mean={mean},P_drw={P_drw},Q={Q}, sigma_noise={sigma_noise}")



    kernel = Lorentzian(log_S0=log_variance_qpo, log_Q=np.log(Q), log_omega0=log_d) + RealTerm(log_a=np.log(variance_drw), log_c=np.log(w_bend))
    truth = kernel.get_parameter_vector()
    psd_model = kernel.get_psd

    SIGMA_NOISE = 10
    # create simulator object with Gaussian noise
    simulator = Simulator(psd_model, times, np.ones(len(times)) * exposure, mean, pdf="Gaussian", 
                        sigma_noise=SIGMA_NOISE, extension_factor = 100)

    # simulate noiseless count rates from the PSD, make the initial lightcurve 2 times as long as the original times
    countrates = simulator.generate_lightcurve()
    # add (Poisson) noise
    noisy_countrates, dy = simulator.add_noise(countrates)

    drw_qpo_array = np.array([times, noisy_countrates, dy, np.ones(len(times)) * exposure]).T
    np.savetxt("simulations/DRW_QPO.txt", drw_qpo_array, header=f"P_qpo={P_qpo},mean={mean},P_drw={P_drw},Q={Q}, sigma_noise={sigma_noise}")

simulate_lc(P_qpo=25,mean=100,P_drw=100,Q=50, sigma_noise=1)