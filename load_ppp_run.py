import pickle
import numpy as np
import matplotlib.pyplot as plt
from ppp_analysis import *
from fit_lognormal_TLRT import plot_lognormal, plot_original

def load_lightcurve(filetime):
    """returns GappyLightCurve"""
    lc = pickle.load(open("saves/ppp/"+filetime+"/input_lc.pkl",'rb'))
    return lc

def load_likelihoods(filetime):
    """returns null_likelihoods, alt_likelihoods"""
    likelihoods = np.loadtxt(f"saves/ppp/{filetime}/likelihoods.txt")
    return likelihoods[:,0], likelihoods[:,1]

def load_T_LRT(filetime):
    """returns T_dist, T_obs"""
    tlrt = np.loadtxt(f"saves/ppp/{filetime}/T_LRT.txt")
    return tlrt[:-1], tlrt[-1]

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
    lc = load_lightcurve(filetime)
    print(lc.exposures)
    print(lc.y)
    print(lc.dy)
    lc._times *= 86400
    print(lc.times)
    plot_lightcurve(lc)
    plt.show()
    null_likelihoods, alt_likelihoods = load_likelihoods(filetime)
    null_model, _ = define_null_hypothesis(lc)
    plt.show()
    alt_model, _ = define_alternative_model(lc, initial_guess={"P_qpo":100})
    plt.show()
    #p_val, T_dist, T_obs = T_LRT_dist(null_likelihoods, alt_likelihoods, null_model, alt_model)
    T_dist, T_obs = load_T_LRT(filetime)
    plot_original(T_dist, T_obs)
    plot_lognormal(T_dist, T_obs)
    plt.show()