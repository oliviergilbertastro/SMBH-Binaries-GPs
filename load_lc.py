import pickle
import numpy as np
import matplotlib.pyplot as plt
from ppp_analysis import *

def load_lightcurve(filetime):
    lc = pickle.load(open("saves/ppp/"+filetime+"/input_lc.pkl",'rb'))
    return lc

if __name__ == "__main__":
    lc = load_lightcurve("2025_05_12_11h40m09s")
    plot_lightcurve(lc)
    plt.show()