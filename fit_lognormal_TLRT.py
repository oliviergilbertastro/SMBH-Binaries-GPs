import pickle
import numpy as np
import matplotlib.pyplot as plt
from ppp_analysis import *
from scipy.optimize import curve_fit
from scipy.stats import lognorm, norm


def plot_original(T_dist, T_obs):
    plt.figure()
    plt.hist(T_dist, bins=10)
    print("Observed LRT_stat: %.3f" % T_obs)
    perc = percentileofscore(T_dist, T_obs)
    print("p-value: %.4f" % (1 - perc / 100))
    plt.axvline(T_obs, label="%.2f%%" % perc, ls="--", color="black")

    sigmas = [95, 99.7]
    colors= ["red", "green"]
    for i, sigma in enumerate(sigmas):
        plt.axvline(np.percentile(T_dist, sigma), ls="--", color=colors[i], label=[r"$2\sigma$",r"$3\sigma$"][i])
    plt.legend()
    #plt.axvline(np.percentile(T_dist, 99.97), color="green")
    plt.xlabel("$T_\\mathrm{LRT}$")

def plot_lognormal(T_dist, T_obs):
    plt.figure()
    hist, bin_edges = np.histogram(T_dist, bins=10, density=True)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers
    y = hist  # PDF estimate
    params = lognorm.fit(T_dist)
    plt.stairs(hist, bin_edges, fill=True)
    x = np.linspace(0,np.max([T_obs,np.max(T_dist),lognorm.ppf(99.7/100, *params)]), 10000)
    lognorm.cdf(T_obs, *params)
    perc = lognorm.cdf(T_obs, *params)*100
    print(perc)
    plt.plot(x,lognorm.pdf(x,*params), color="purple", linestyle="--", label="lognormal fit")
    plt.axvline(T_obs, label="%.2f%%" % perc, ls="--", color="black")

    sigmas = [95, 99.7]
    colors= ["red", "green"]
    for i, sigma in enumerate(sigmas):
        plt.axvline(lognorm.ppf(sigma/100, *params), ls="--", color=colors[i], label=[r"$2\sigma$",r"$3\sigma$"][i])
    plt.legend()
    plt.xlabel("$T_\\mathrm{LRT}$")

    print("p-value: %.8f" % (1 - perc / 100))
    n_sigmas_significance = norm.ppf(perc/100)
    if n_sigmas_significance > 1:
        print(f"QPO is detected with {n_sigmas_significance} sigmas of significance.")
    else:
        print(f"QPO is not significantly detected.")

if __name__ == "__main__":
    from load_ppp_run import *
    filetime = "2025_05_12_12h04m57s"
    T_dist, T_obs = load_T_LRT(filetime)
    #plot_lognormal(T_dist, T_obs)
    if False:
        x = np.linspace(0,10, 1000)
        plt.plot(x,lognormal(x, 1, 0, 1), label=r"$\sigma=1,\mu=0$")
        plt.plot(x,lognormal(x, 1, 1, 1), label=r"$\sigma=1,\mu=1$")
        plt.plot(x,lognormal(x, 2, 0, 1), label=r"$\sigma=2,\mu=0$")
        plt.plot(x,lognormal(x, 1, 0, 2), label=r"$\sigma=1,\mu=0$, scale=2")
        plt.legend()
        plt.show()
    plot_original(T_dist, T_obs)
    plot_lognormal(T_dist, T_obs)
    plt.show()