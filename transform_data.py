"""
Transform real data to the template that will be used to create GappyLightCurves (times, countrates, dy, exposures)
At the same time, you can visualize the resulting lightcurve here.
"""

import numpy as np
import matplotlib.pyplot as plt
from mind_the_gaps.lightcurves import GappyLightcurve
import pandas as pd


def load_ASASSN_data(object_name):
    """
    load an ASAS-SN csv lightcurve from the object name

    returns: lc tuple containing object name + data
    """
    #data = np.loadtxt(f"data/raw/{object_name}.csv", skiprows=12)
    data = pd.read_csv(f"data/raw/{object_name}.csv", header=9)
    jd, flux, flux_error = data["JD"], data["Flux"], data["Flux Error"]
    times = (jd-jd[0])*86400 # Put the start time at 0 and convert to seconds
    exposures = np.ones_like(jd)*90 # For ASAS-SN, each epoch consists of 3x90s exposures. Maybe it's only 90s??

    # Not quite sure if flux = count_rates (i.e. counts/s), uncertainties sure seem small though
    return (object_name, np.array([times, flux, flux_error, exposures]))


def plot_lightcurve(input_lc, units="days"):
    """Here input_lc is only the tuple returned by load_data functions"""
    times, y, dy, exposures = input_lc[1]
    fig = plt.figure()
    if units == "days":
        plt.errorbar(times, y, yerr=dy, ls="None", marker=".")
        plt.xlabel("Time (days)")
    elif units == "seconds":
        plt.errorbar(times / 86400, y, yerr=dy, ls="None", marker=".")
        plt.xlabel("Time (days)")
    plt.ylabel("Rates (ct/s)")
    plt.title(input_lc[0], fontsize=16)

def save_lc(lc):
    """
    save a lc (the tuple returned by load_data functions) as numpy array with name of the object
    """
    object_name, lc_array = lc
    np.savetxt(f"data/transformed/{object_name}.txt", lc_array.T)

if __name__ == "__main__":
    lc = load_ASASSN_data("mrk421")
    #plot_lightcurve(lc, units="seconds")

    times = lc[1][0]
    flux = lc[1][1]
    flux_err = lc[1][2]
    data = pd.read_csv(f"data/raw/mrk421.csv", header=9)
    cam = data["Camera"]
    good_indexes = []
    for i in range(len(cam)):
        if cam[i] == "bc":
            good_indexes.append(i)
        

    print(np.mean(flux_err))
    plt.plot(flux, flux_err/flux, "o", color="blue", label="camera=bs")
    plt.plot(flux[good_indexes], flux_err[good_indexes]/flux[good_indexes], "o", color="red", label="camera=bc")
    plt.xlabel("Flux", fontsize=16)
    plt.ylabel("Flux_err/Flux", fontsize=16)
    plt.title("Mrk 421 flux err vs flux")
    plt.legend()
    plt.show()


    plt.plot(flux, flux_err, "o")
    plt.xlabel("Flux", fontsize=16)
    plt.ylabel("Flux_err", fontsize=16)
    plt.show()

    print(np.median(np.diff(times)))
    print(np.mean(np.diff(times)))
    plt.figure()
    plt.plot(np.diff(times))
    plt.axhline(270, linestyle="--", color="red")
    plt.axhline(90+15, linestyle="--", color="blue")
    plt.yscale("log")
    plt.xlabel("Index of observation", fontsize=16)
    plt.ylabel(r"$\log( t_{i+1} - t_i ) [s]$", fontsize=16)
    plt.show()

    save_lc(lc)
