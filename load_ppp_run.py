import pickle
import numpy as np
import matplotlib.pyplot as plt
from ppp_analysis import *

def load_lightcurve(filetime):
    lc = pickle.load(open("saves/ppp/"+filetime+"/input_lc.pkl",'rb'))
    return lc

def load_likelihoods(filetime):
    "returns null_likelihoods, alt_likelihoods"
    likelihoods = np.loadtxt(f"saves/ppp/{filetime}/likelihoods.txt")
    return likelihoods[:,0], likelihoods[:,1]

if __name__ == "__main__":
    filetime = "2025_05_12_12h04m57s"
    lc = load_lightcurve(filetime)
    plot_lightcurve(lc)
    plt.show()
    null_likelihoods, alt_likelihoods = load_likelihoods(filetime)
    null_model, _ = define_null_hypothesis(lc)
    alt_model, _ = define_alternative_model(lc)
    T_LRT_dist(null_likelihoods, alt_likelihoods, null_model, alt_model)
    plt.show()