# 1

import numpy as np
from celerite.terms import Matern32Term, RealTerm
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

# 2

if __name__ == "__main__":

    times  = np.arange(0, 1000)
    exposure = np.diff(times)[0]

    P_qpo = 25 # period of the QPO
    w = 2 * np.pi / P_qpo
    mean = 100
    rms = 0.1
    variance_drw = (mean * rms) ** 2  # variance of the DRW (bending powerlaw)
    P_drw = 40
    w_bend = 2 * np.pi / P_drw # angular frequency of the DRW or Bending Powerlaw
    # Define starting parameters
    log_variance_qpo = np.log(variance_drw)
    log_sigma_matern = np.log(np.sqrt(variance_drw))
    P_matern = 10
    log_rho_matern =  np.log(P_matern / 2 / np.pi)
    Q = 80 # coherence
    log_Q = np.log(Q)
    log_d = np.log(w)
    print(f"log variance of the QPO: {log_variance_qpo:.2f}, log_Q: {log_Q:.2f}, log omega: {log_d:.2f}")

    labels = ["Lorentzian", "DRW", "Matern-3/2"]
    # You can also use Lorentzian from models.celerite_models (which is defined in terms of variance, Q and omega)
    kernel = Lorentzian(log_S0=log_variance_qpo, log_Q=np.log(Q), log_omega0=log_d) \
        + RealTerm(log_a=np.log(variance_drw), log_c=np.log(w_bend)) + Matern32Term(log_sigma_matern, 
                                                                                    log_rho_matern, eps=1e-8)
    truth = kernel.get_parameter_vector()
    psd_model = kernel.get_psd

    # create simulator object with Gaussian noise
    simulator = Simulator(psd_model, times, np.ones(len(times)) * exposure, mean, pdf="Gaussian", 
                        sigma_noise=10, extension_factor = 2)

    # simulate noiseless count rates from the PSD, make the initial lightcurve 2 times as long as the original times
    countrates = simulator.generate_lightcurve()
    # add (Poisson) noise
    noisy_countrates, dy = simulator.add_noise(countrates)
    input_lc = GappyLightcurve(times, noisy_countrates, dy, exposures=exposure)

    fig = plt.figure()
    plt.errorbar(times, noisy_countrates, yerr=dy, ls="None", marker=".")
    plt.xlabel("Time (days)")
    plt.ylabel("Rates (ct/s)")

    # plot PSD
    df = 1 / input_lc.duration
    nyquist = 1 / (2 * exposure)
    freqs = np.arange(df, nyquist, df)
    fig = plt.figure()
    # remember angular freqs for the PSD models
    plt.plot(freqs, psd_model(2 * np.pi * freqs))
    for p in [P_qpo, P_drw, P_matern]:
        plt.axvline(1 / p, ls="--", color="black")

    plt.xlabel("Frequency")
    plt.ylabel("Power Density")
    plt.xscale("log")
    plt.yscale("log")
    for i, term in enumerate(kernel.terms):
        plt.plot(freqs, term.get_psd(2 * np.pi * freqs), label=labels[i])
    plt.legend()

    lc_variance = np.var(input_lc.y)




    # 3
    def bounds_variance(variance, margin=15):
        return np.log(variance/margin), np.log(variance * margin)

    def bounds_bend(duration, dt):
        nyquist = 1 / (2 * dt)
        return np.log(2 * np.pi/duration), np.log(nyquist * 2 * np.pi)


    # 4
    variance_bounds = bounds_variance(lc_variance)
    print(variance_bounds)
    bend_bounds = bounds_bend(input_lc.duration, exposure)
    print(bend_bounds)
    sigma_bounds = bounds_variance(np.sqrt(lc_variance))
    timescale_bounds = ((np.log(exposure), np.log(input_lc.duration)))
    print(timescale_bounds)
    # limit Q lower bound to "periodic" components
    Q_bounds = (np.log(1.5), np.log(1000))

    log_var = np.log(lc_variance)
    realterm = RealTerm(log_var, np.log(2 * np.pi / 50), bounds=[variance_bounds, bend_bounds])
    lorentzian = Lorentzian(log_var, np.log(100), np.log(2 * np.pi/10), 
                        bounds=[variance_bounds, Q_bounds, bend_bounds])
    matern = Matern32Term(np.log(np.sqrt(lc_variance)), np.log(10), bounds=[sigma_bounds, timescale_bounds], 
                        eps=1e-8)
    models = [realterm, 
            matern,
            lorentzian + realterm,
            lorentzian + realterm + matern
            ,]


    # 5
    cpus = 8
    aiccs = []
    pvalues = []
    gps = []
    for kernel in models:
        print(kernel)
        print("------------------------------------------------------------------")
        gp = GPModelling(input_lc, kernel)
        # here we first minimize the likelihood and then run a small MCMC to ensure we find the maximum of the loglikelihood
        gp.derive_posteriors(fit=True, max_steps=2000, walkers=2 * cpus, cores=cpus, progress=False)
        best_pars = gp.max_parameters
        gp.gp.set_parameter_vector(best_pars)
        std_res = gp.standarized_residuals()
        pvalue = ks_1samp(std_res, norm.cdf).pvalue
        AICc = aicc(gp.max_loglikelihood, gp.k, input_lc.n)
        print(f"p-value:{pvalue:.3f} | AICC: {AICc:.2f}")
        pvalues.append(pvalue)
        aiccs.append(AICc)
        gps.append(gp)


    # 6
    best_gp = gps[np.argmin(aiccs)]
    print(best_gp.max_parameters)
    print(f"Best model {models[np.argmin(aiccs)]} has a p-value: {pvalues[np.argmin(AICc)]:.3f}")
    corner_fig = corner.corner(best_gp.mcmc_samples, labels=best_gp.gp.get_parameter_names(), 
                            title_fmt='.1f', truths=truth,
                                quantiles=[0.16, 0.5, 0.84], show_titles=True,
                                title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                                levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels
    plt.savefig("figures/tutorial/corner.png")
    plt.show()
    # 
    best_gp.gp.set_parameter_vector(gp.max_parameters)
    print(best_gp.gp.get_parameter_vector())
    pred_mean, pred_var = gp.gp.predict(input_lc.y, return_var=True)
    # standarized residuals
    plt.figure()
    std_res = gp.standarized_residuals()
    plt.hist(norm.rvs(size=10000), edgecolor="black" , ls="--",
            facecolor="None", density=True, label="$N\sim(0, 1)$")
    plt.hist(std_res, density="True", edgecolor="black", label="Data")
    plt.xlabel("Standarized Residuals")
    plt.ylabel("PDF")
    plt.legend()
    plt.savefig("figures/tutorial/std_res.png")
    plt.show()
    # ACF
    plt.figure()
    plt.acorr(std_res, maxlags=len(std_res) // 4 )
    plt.xlim(0)
    # 2 sigma
    plt.axhspan(-1.96 / np.sqrt(len(std_res)), 1.96 / np.sqrt(len(std_res)), alpha=0.3, color="black")
    plt.ylabel("ACF of Standarized Residuals")
    plt.xlabel("Lags")
    plt.savefig("figures/tutorial/acf.png")
    plt.show()
    # best-fit model
    fig = plt.figure()
    plt.errorbar(times, noisy_countrates, yerr=dy, ls="None", marker=".", color="black")
    plt.xlabel("Time (days)")
    plt.ylabel("Rates (ct/s)")
    plt.plot(times, pred_mean, label="Model", color="C1")
    plt.fill_between(times, pred_mean - np.sqrt(pred_var), pred_mean + np.sqrt(pred_var), 
                    zorder=10, color="C1", alpha=0.5)
    plt.legend()
    plt.savefig("figures/tutorial/model.png")
    plt.show()