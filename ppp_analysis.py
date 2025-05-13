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


def plot_lightcurve(input_lc):
    fig = plt.figure()
    plt.errorbar(input_lc._times, input_lc._y, yerr=input_lc._dy, ls="None", marker=".")
    plt.xlabel("Time (days)")
    plt.ylabel("Rates (ct/s)")



# Define the null-hypothesis
def define_null_hypothesis(input_lc, savefolder=None):
    bounds_drw = dict(log_a=(-10, 50), log_c=(-10, 10))
    null_kernel = celerite.terms.RealTerm(log_a=np.log(100), log_c=np.log(2*np.pi/30), bounds=bounds_drw)
    null_model = GPModelling(input_lc, null_kernel)
    print("Deriving posteriors for null model")
    null_model.derive_posteriors(max_steps=50000, fit=True, cores=cpus)

    corner_fig = corner.corner(null_model.mcmc_samples, labels=null_model.gp.get_parameter_names(), title_fmt='.1f',
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                                levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels

    autocorr = null_model.autocorr
    fig = plt.figure()
    n = np.arange(1, len(autocorr) + 1)
    plt.plot(n, autocorr, "-o")
    plt.ylabel("Mean $\\tau$")
    plt.xlabel("Number of steps")
    if savefolder is not None:
        plt.savefig(f"{savefolder}null_autocorr.png", dpi=100)
    return null_model, null_kernel

def define_alternative_model(input_lc, model="Lorentzian", savefolder=None, initials_guess={"P_qpo":10}):
    bounds_drw = dict(log_a=(-10, 50), log_c=(-10, 10))
    P = initials_guess["P_qpo"] # period of the QPO
    w = 2 * np.pi / P
    # Define starting parameters
    log_variance_qpo = np.log(100)
    Q = 80 # coherence
    log_c = np.log(0.5 * w/Q)
    log_d = np.log(w)
    print(f"log variance starting values of the QPO: {log_variance_qpo:.2f}, log_c: {log_c:.2f}, log omega: {log_d:.2f}")

    lc_variance = np.var(input_lc.y)
    bounds_qpo_complex = dict(log_a=(-10, 50), log_c=(-10, 10), log_d=(-5, 5))

    def bounds_variance(variance, margin=15):
        return np.log(variance/margin), np.log(variance * margin)

    def bounds_bend(duration, dt):
        nyquist = 1 / (2 * dt)
        return np.log(2 * np.pi/duration), np.log(nyquist * 2 * np.pi)
    variance_bounds = bounds_variance(lc_variance)
    bend_bounds = bounds_bend(input_lc.duration, input_lc._exposures[0])
    Q_bounds = (np.log(1.5), np.log(1000))
    bounds_qpo = dict(log_S0=variance_bounds, log_Q=Q_bounds, log_omega0=bend_bounds)
    # You can also use Lorentzian from models.celerite_models (which is defined in terms of variance, Q and omega)
    if model == "Lorentzian":
        alternative_kernel = Lor(np.log(100), np.log(100), np.log(2 * np.pi/10), bounds=bounds_qpo) + celerite.terms.RealTerm(log_a=np.log(100), log_c=np.log(2*np.pi/30), bounds=bounds_drw)
    else:
        alternative_kernel = celerite.terms.ComplexTerm(log_a=log_variance_qpo, log_c=log_c, log_d=log_d, bounds=bounds_qpo_complex) + celerite.terms.RealTerm(log_a=np.log(100), log_c=np.log(2*np.pi/30), bounds=bounds_drw)
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
                            title_fmt='.1f',
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                            levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels
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


def complete_PPP_analysis(input_lc, save_data=True, infos=None, if_plot=True):
    """
    Make the full analysis of LRT distributions and save the important data under "saves/ppp/DD_MM_YYYY_TIME"
    input_lc : GappyLightCurve object of our data (or simulated data)
    save_data : bool, True if we want to save
    infos : string to add infos as a text file in the save folder
    """
    if infos is not None and not save_data:
        raise ValueError("There are infos to be saved, but save_data=False")
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

    plot_lightcurve(input_lc)
    if if_plot:
        plt.show()
    null_model, null_kernel = define_null_hypothesis(input_lc, savefolder=savefolder)
    if if_plot:
        plt.show()
    alternative_model, alternative_kernel = define_alternative_model(input_lc, model="Lorentzian", savefolder=savefolder)
    # Save the models:
    if save_data:
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

if __name__ == "__main__":
    drw_data = np.loadtxt("simulations/DRW.txt", skiprows=1)
    with open('simulations/DRW.txt') as f:
        header_drw = f.readline()
    drw_qpo_data = np.loadtxt("simulations/DRW_QPO.txt", skiprows=1)
    with open('simulations/DRW_QPO.txt') as f:
        header_drw_qpo = f.readline()
    times, noisy_countrates, dy, exposures = drw_data[:,0], drw_data[:,1], drw_data[:,2], drw_data[:,3]
    input_drw_lc = GappyLightcurve(times, noisy_countrates, dy, exposures=exposures)
    times, noisy_countrates, dy, exposures = drw_qpo_data[:,0], drw_qpo_data[:,1], drw_qpo_data[:,2], drw_qpo_data[:,3]
    input_drw_qpo_lc = GappyLightcurve(times, noisy_countrates, dy, exposures=exposures)

    complete_PPP_analysis(input_drw_lc, save_data=True, infos=f"Red noise only, simulated\n{header_drw}", if_plot=False)
    complete_PPP_analysis(input_drw_qpo_lc, save_data=True, infos=f"Red noise + QPO, simulated\n{header_drw_qpo}", if_plot=False)