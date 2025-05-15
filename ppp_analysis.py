"""
Code to calculate the posterior predictive p-value (PPP), which calculates the likelihood ratio test (LRT) distribution
"""

from mind_the_gaps.lightcurves import GappyLightcurve
from mind_the_gaps.gpmodelling import GPModelling
from celerite.terms import Matern32Term, RealTerm, JitterTerm
from mind_the_gaps.models.celerite_models import Lorentzian as Lor
from mind_the_gaps.models.psd_models import BendingPowerlaw, Lorentzian, SHO, Matern32, Jitter
from mind_the_gaps.models.celerite_models import DampedRandomWalk
from mind_the_gaps.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import celerite, corner
from scipy.stats import percentileofscore
from utils import print_color

cpus = 10 # set the number of cores for parallelization
np.random.seed(10)


def plot_lightcurve(input_lc, title=None, units="days"):
    fig = plt.figure()
    if units == "days":
        plt.errorbar(input_lc._times, input_lc._y, yerr=input_lc._dy, ls="None", marker=".")
        plt.xlabel("Time (days)")
    elif units == "seconds":
        plt.errorbar(input_lc._times / 86400, input_lc._y, yerr=input_lc._dy, ls="None", marker=".")
        plt.xlabel("Time (days)")
    plt.ylabel("Rates (ct/s)")
    if title is not None:
        plt.title(title, fontsize=16)



# Define the null-hypothesis
def define_null_hypothesis(input_lc, savefolder=None, units="days"):
    bounds_drw = dict(log_a=(-10, 50), log_c=(-10, 10))
    drw_variance_guess = np.var(input_lc.y)
    drw_c_guess = 2*np.pi/30
    if units == "seconds":
        bounds_drw = dict(log_a=(-10, 50), log_c=(-21.37, -1.37)) # log_c bounds shifted accordingly to 86400 seconds/day factor
        drw_c_guess /= 86400
    null_kernel = celerite.terms.RealTerm(log_a=np.log(drw_variance_guess), log_c=np.log(drw_c_guess), bounds=bounds_drw)
    null_model = GPModelling(input_lc, null_kernel)
    print("Deriving posteriors for null model")
    null_model.derive_posteriors(max_steps=50000, fit=True, cores=cpus)

    corner_fig = corner.corner(null_model.mcmc_samples, labels=null_model.gp.get_parameter_names(), title_fmt='.2f',
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                                levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels

    if savefolder is not None:
        plt.savefig(f"{savefolder}null_corner.png", dpi=100)
    autocorr = null_model.autocorr
    fig = plt.figure()
    n = np.arange(1, len(autocorr) + 1)
    plt.plot(n, autocorr, "-o")
    plt.ylabel("Mean $\\tau$")
    plt.xlabel("Number of steps")
    if savefolder is not None:
        plt.savefig(f"{savefolder}null_autocorr.png", dpi=100)
    return null_model, null_kernel

def define_alternative_model(input_lc, savefolder=None, initial_guess={"P_qpo":50}, units="days"):
    
    
    # Define starting parameters
    log_variance_qpo = np.log(100)
    Q = 80 # coherence
    log_c = np.log(0.5 * w/Q)
    log_d = np.log(w)



    lc_variance = np.var(input_lc.y)

    def bounds_variance(variance, margin=15):
        return np.log(variance/margin), np.log(variance * margin)

    def bounds_bend(duration, dt):
        nyquist = 1 / (2 * dt)
        return np.log(2 * np.pi/duration), np.log(nyquist * 2 * np.pi)
    # You can also use Lorentzian from models.celerite_models (which is defined in terms of variance, Q and omega)

    log_variance = np.log(lc_variance)
    bounds_drw = dict(log_a=(-10, 50), log_c=(-10, 10))
    drw_variance_guess = 100
    drw_c_guess = 2*np.pi/30
    Q_guess = 100
    P_qpo_guess = initial_guess["P_qpo"] # period of the QPO
    w_qpo_guess = 2 * np.pi / P_qpo_guess
    variance_bounds = bounds_variance(lc_variance)
    Q_bounds = (np.log(1.5), np.log(1000))
    bend_bounds = bounds_bend(input_lc.duration, input_lc._exposures[0])
    bounds_qpo = dict(log_S0=variance_bounds, log_Q=Q_bounds, log_omega0=bend_bounds)
    if units == "seconds":
        drw_c_guess /= 86400
        bounds_drw = dict(log_a=(-10, 50), log_c=(-21.37, -1.37)) # log_c bounds shifted accordingly to 86400 seconds/day factor
        w_qpo_guess /= 86400


    print(f"log variance starting values of the QPO: {log_variance:.2f}, log_Q: {np.log(Q_guess):.2f}, log omega0: {np.log(w_qpo_guess):.2f}")
    alternative_kernel = Lor(log_variance, np.log(Q_guess), np.log(w_qpo_guess), bounds=bounds_qpo) + celerite.terms.RealTerm(log_a=log_variance, log_c=np.log(drw_c_guess), bounds=bounds_drw)
    alternative_model = GPModelling(input_lc, alternative_kernel)
    print("Deriving posteriors for alternative model")
    alternative_model.derive_posteriors(max_steps=50000, fit=True, cores=cpus)

    autocorr = alternative_model.autocorr
    fig = plt.figure()
    n = np.arange(1, len(autocorr) + 1)
    plt.plot(n, autocorr, "-o")
    plt.ylabel("Mean $\\tau$")
    plt.xlabel("Number of steps")
    if savefolder is not None:
        plt.savefig(f"{savefolder}alt_autocorr.png", dpi=100)

    corner_fig = corner.corner(alternative_model.mcmc_samples, labels=alternative_model.gp.get_parameter_names(), 
                            title_fmt='.2f',
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                            levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels
    if savefolder is not None:
        plt.savefig(f"{savefolder}alt_corner.png", dpi=100)

    return alternative_model, alternative_kernel

def generate_lightcurves(null_model, Nsims=100):
    Nsims = 100 # typically 10,000
    lcs = null_model.generate_from_posteriors(Nsims, cpus=cpus)
    print_color(f"Done generating {Nsims} lightcurves!")
    return lcs

def fit_lightcurves(lcs, null_kernel, alternative_kernel):
    likelihoods_null = []
    likelihoods_alt = []

    for i, lc in enumerate(lcs):
        print("Processing lightcurve %d/%d" % (i + 1, len(lcs)), end="\r")
        
        # Run a small MCMC to make sure we find the global maximum of the likelihood
        # ideally we'd probably want to run more samples
        null_modelling = GPModelling(lc, null_kernel)
        null_modelling.derive_posteriors(fit=True, cores=cpus, walkers=2 * cpus, max_steps=1000, progress=False)
        likelihoods_null.append(null_modelling.max_loglikelihood)
        alternative_modelling = GPModelling(lc, alternative_kernel)                         
        alternative_modelling.derive_posteriors(fit=True, cores=cpus, walkers=2 * cpus, max_steps=1000, 
                                                progress=False)
        likelihoods_alt.append(alternative_modelling.max_loglikelihood)
    print_color(f"Done fitting lightcurves!")
    return likelihoods_null, likelihoods_alt

def T_LRT_dist(likelihoods_null, likelihoods_alt, null_model, alternative_model, savefolder=None):
    plt.figure()
    T_dist = -2 * (np.array(likelihoods_null) - np.array(likelihoods_alt))
    print(T_dist)
    plt.hist(T_dist, bins=10)
    T_obs = -2 * (null_model.max_loglikelihood - alternative_model.max_loglikelihood)
    print("Observed LRT_stat: %.3f" % T_obs)
    perc = percentileofscore(T_dist, T_obs)
    print("p-value: %.4f" % (1 - perc / 100))
    plt.axvline(T_obs, label="%.2f%%" % perc, ls="--", color="black")

    sigmas = [95, 99.7]
    colors= ["red", "green"]
    for i, sigma in enumerate(sigmas):
        plt.axvline(np.percentile(T_dist, sigma), ls="--", color=colors[i])
    plt.legend()
    #plt.axvline(np.percentile(T_dist, 99.97), color="green")
    plt.xlabel("$T_\\mathrm{LRT}$")

    if savefolder is not None:
        plt.savefig(f"{savefolder}LRT_statistic.png", dpi=100)

    return (1 - perc / 100), T_dist, T_obs


def complete_PPP_analysis(input_lc, save_data=True, infos=None, if_plot=True, save_models=False, units="days"):
    """
    Make the full analysis of LRT distributions and save the important data under "saves/ppp/DD_MM_YYYY_TIME"
    input_lc : GappyLightCurve object of our data (or simulated data)
    save_data : bool, True if we want to save
    infos : string to add infos as a text file in the save folder
    save_models : bool, True to save the null and alternative models as pickle objects (takes a lot of space, around 30mb each)
    units : string, change to "seconds" so the bounds and initial guesses of the model fitting are adjusted
    """
    if infos is not None and not save_data:
        raise ValueError("There are infos to be saved, but save_data=False")
    if save_models and not save_data:
        raise ValueError("There are models to be saved, but save_data=False")
    units = units.lower()
    if units != "days" and units != "seconds":
        raise ValueError("Units must be either seconds or days")
    savefolder = None
    if save_data:
        import pickle
        from datetime import datetime
        import os
        savefolder = "saves/ppp/" + datetime.now().strftime('%Y_%m_%d_%Hh%Mm%Ss') + "/"
        print(savefolder)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        else:
            raise ValueError("Folder already created at this time.")
        with open(f'{savefolder}input_lc.pkl', 'wb') as f:
            pickle.dump(input_lc, f)
        if infos is not None:
            with open(f"{savefolder}info.txt", "a") as f:
                f.write(infos)

    plot_lightcurve(input_lc, units=units)
    if if_plot:
        plt.show()
    null_model, null_kernel = define_null_hypothesis(input_lc, savefolder=savefolder, units=units)
    if if_plot:
        plt.show()
    alternative_model, alternative_kernel = define_alternative_model(input_lc, model="Lorentzian", savefolder=savefolder, units=units)
    # Save the models:
    if save_data and save_models:
        with open(f'{savefolder}null_model.pkl', 'wb') as f:
            pickle.dump(null_model, f)
        with open(f'{savefolder}alt_model.pkl', 'wb') as f:
            pickle.dump(alternative_model, f)
        with open(f'{savefolder}null_kernel.pkl', 'wb') as f:
            pickle.dump(null_kernel, f)
        with open(f'{savefolder}alt_kernel.pkl', 'wb') as f:
            pickle.dump(alternative_kernel, f)
    if if_plot:
        plt.show()
    lcs = generate_lightcurves(null_model, Nsims=100)
    likelihoods_null, likelihoods_alt = fit_lightcurves(lcs, null_kernel, alternative_kernel)
    if save_data:
        np.savetxt(f"{savefolder}likelihoods.txt", np.array([likelihoods_null, likelihoods_alt]).T)
    pval, T_dist, T_obs = T_LRT_dist(likelihoods_null, likelihoods_alt, null_model, alternative_model, savefolder=savefolder)
    if save_data:
        tlrt = list(T_dist)
        tlrt.append(T_obs)
        np.savetxt(f"{savefolder}T_LRT.txt", np.array(tlrt))
        with open(f"{savefolder}info.txt", "a") as f:
                f.write(f"\np-value = {pval}")
                f.write(f"\nObserved T_LRT = {T_obs}")
    if if_plot:
        plt.show()

def load_simulation_to_lc(savename):
    data = np.loadtxt(f"simulations/{savename}.txt", skiprows=1)
    with open(f"simulations/{savename}.txt") as f:
        first_line = f.readline().strip()
    if first_line.startswith("#"):
        first_line = first_line[1:].strip()
    model = (first_line.split(", ")[0]).split("=")[0]
    times, noisy_countrates, dy, exposures = data[:,0], data[:,1], data[:,2], data[:,3]
    input_lc = GappyLightcurve(times, noisy_countrates, dy, exposures=exposures)
    return input_lc, model, first_line

def analyze_simulation(savename):
    input_lc, model, header = load_simulation_to_lc(savename)
    complete_PPP_analysis(input_lc, save_data=True, infos=f"{model}, simulated\n{header}", if_plot=False)

if __name__ == "__main__":
    from simulate_lightcurves import *
    simulate_lc(model="DRW+QPO",
                savename=f"DRW_QPO_{0}",
                P_qpo=25*86400, # 25 days
                mean=100,
                P_drw=100*86400, # 100 days
                Q=80,
                sigma_noise=1,
                timerange=365*86400, 
                length=100,
                time_sigma=0.7*86400,
                exposure_time=60 # exposure times for ASAS-SN and SDSS DR16 are on the order of 1 minute
                )
    # exposures of 2 mins
    lc, model, header = load_simulation_to_lc(f"DRW_QPO_{0}")
    plot_lightcurve(input_lc=lc, title=model)
    plot_lightcurve(input_lc=lc, title=model, units="days")
    plt.show()
    complete_PPP_analysis(lc, save_data=True, infos=f"{model}, simulated\n{header}", if_plot=True, units="days")