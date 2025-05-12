import pickle
import numpy as np
import matplotlib.pyplot as plt
from ppp_analysis import *
from load_ppp_run import *
from scipy.optimize import curve_fit

def plot_lognormal(T_dist, T_obs):
    plt.figure()
    plt.hist(T_dist, bins=10)
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
    plt.show()


if __name__ == "__main__":
    filetime = "2025_05_12_12h04m57s"
    T_dist, T_obs = load_T_LRT(filetime)
    plot_lognormal(T_dist, T_obs)