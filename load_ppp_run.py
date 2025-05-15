import pickle
import numpy as np
import matplotlib.pyplot as plt
from ppp_analysis import *
from fit_lognormal_TLRT import plot_lognormal, plot_original

def load_lightcurve(filetime, data_type="simulation"):
    """returns GappyLightCurve"""
    lc = pickle.load(open(f"saves/{data_type}/{filetime}/input_lc.pkl",'rb'))
    return lc

def load_likelihoods(filetime, data_type="simulation"):
    """returns null_likelihoods, alt_likelihoods"""
    likelihoods = np.loadtxt(f"saves/{data_type}/{filetime}/likelihoods.txt")
    return likelihoods[:,0], likelihoods[:,1]

def load_T_LRT(filetime, data_type="simulation"):
    """returns T_dist, T_obs"""
    tlrt = np.loadtxt(f"saves/{data_type}/{filetime}/T_LRT.txt")
    return tlrt[:-1], tlrt[-1]

def load_everything(filetime, data_type="simulation"):
    lc = load_lightcurve(filetime, data_type)
    null_likelihoods, alt_likelihoods = load_likelihoods(filetime, data_type)
    T_dist, T_obs = load_T_LRT(filetime, data_type)
    return lc, null_likelihoods, alt_likelihoods, T_dist, T_obs

if __name__ == "__main__":
    filetime = "2025_05_12_12h04m57s"
    filetime = "2025_05_12_16h38m03s"
    filetime = "2025_05_12_17h06m26s"
    filetime = "2025_05_13_09h32m08s"
    filetime = "2025_05_13_10h01m40s"
    filetime = "2025_05_13_10h45m37s"
    filetime = "2025_05_13_11h17m22s"
    filetime = "2025_05_13_12h09m52s"
    filetime = "2025_05_13_12h35m16s"
    #filetime = "2025_05_13_13h08m21s"
    #filetime = "2025_05_13_13h33m44s"
    filetime = "2025_05_13_15h19m37s"
    filetime = "2025_05_13_16h07m08s"
    #filetime = "2025_05_13_17h13m09s"
    #filetime = "2025_05_14_09h11m26s"
    #filetime = "2025_05_15_10h19m28s"
    filetime = "2025_05_15_13h28m42s"
    filetime2 = "2025_05_15_14h38m31s"
    lc, null_likelihoods, alt_likelihoods, T_dist, T_obs = load_everything(filetime, data_type="real")
    lc2, null_likelihoods2, alt_likelihoods2, T_dist2, T_obs2 = load_everything(filetime2, data_type="real")
    #null_model, _ = define_null_hypothesis(lc)
    #plt.show()
    #alt_model, _ = define_alternative_model(lc, initial_guess={"P_qpo":100})
    #plt.show()
    #p_val, T_dist, T_obs = T_LRT_dist(null_likelihoods, alt_likelihoods, null_model, alt_model)
    #plot_original(T_dist, T_obs)
    #plot_lognormal(T_dist, T_obs)


    plt.figure()
    plt.hist(T_dist, bins=10, color="red", label="90s exposures")
    plt.hist(T_dist2, bins=10, color="blue", label="30s exposures")
    print("Observed LRT_stat: %.3f" % T_obs)
    perc = percentileofscore(T_dist, T_obs)
    print("p-value: %.4f" % (1 - perc / 100))
    plt.axvline(T_obs, label="%.2f%%" % perc, ls="--", color="black")

    #sigmas = [95, 99.7]
    #colors= ["red", "green"]
    #or i, sigma in enumerate(sigmas):
    #    plt.axvline(np.percentile(T_dist, sigma), ls="--", color=colors[i], label=[r"$2\sigma$",r"$3\sigma$"][i])
    plt.legend()
    #plt.axvline(np.percentile(T_dist, 99.97), color="green")
    plt.xlabel("$T_\\mathrm{LRT}$")
    plt.show()