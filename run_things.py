"""
Just a place to run various things without the clutter of ppp_analysis
"""

from ppp_analysis import *
from simulate_lightcurves import simulate_lc
import matplotlib.pyplot as plt
import numpy as np
from load_ppp_run import load_models, load_lightcurve
import sys

if False:

    FILETIME = "2025_05_15_13h28m42s"
    lc = load_lightcurve(FILETIME, data_type="real")
    null_model, alt_model = load_models(FILETIME, data_type="real")
    # Plot the models and the residuals between the models
    times, null_pred_mean, null_pred_var = plot_model_lc(lc, model=null_model, units="seconds", title="Null model")
    _, alt_pred_mean, alt_pred_var = plot_model_lc(lc, model=alt_model, units="seconds", title="Alternative model")


    fig = plt.figure()
    plt.plot(times, null_pred_mean-alt_pred_mean, label="null-alternative")
    plt.xlabel("Time (days)", fontsize=15)
    plt.ylabel("Rates (ct/s)", fontsize=15)
    plt.legend()

    fig = plt.figure()
    plt.plot(times, null_pred_mean, label="null", color="red")
    plt.fill_between(times, null_pred_mean - np.sqrt(null_pred_var), null_pred_mean + np.sqrt(null_pred_var), zorder=10, color="red", alpha=0.5)
    plt.plot(times, alt_pred_mean, label="alternative", color="blue")
    plt.fill_between(times, alt_pred_mean - np.sqrt(alt_pred_var), alt_pred_mean + np.sqrt(alt_pred_var), zorder=10, color="blue", alpha=0.5)
    plt.errorbar(lc.times/86400, lc.y, yerr=lc.dy, label="data", color="black", marker="o")
    plt.xlabel("Time (days)", fontsize=15)
    plt.ylabel("Rates (ct/s)", fontsize=15)
    plt.legend()
    plt.show()

    print(calculate_information_criterions(lc, null_model))
    print(calculate_information_criterions(lc, alt_model))


if __name__ == "__main__":
    # Check the simulated lightcurves
    FILETIME = "2025_05_16_14h12m46s"
    real_lc = load_lightcurve(FILETIME, data_type="real")

    print(find_scale_factor(real_lc))
    
    #sys.exit()

    null_model, alt_model = load_models(FILETIME, data_type="real")
    print(cpus)
    lcs = generate_lightcurves(null_model, Nsims=1)

    for i in range(len(lcs)):
        #plot_lightcurve(lcs[i], title=f"Lightcurve #{i}", units="seconds")
        #plot_lightcurve(real_lc, title=f"Real LC", units="seconds")
        plt.errorbar(real_lc._times / 86400, real_lc._y, yerr=real_lc._dy, ls="None", marker=".", color="black", label="real")
        plt.errorbar(lcs[i]._times / 86400, lcs[i]._y, yerr=lcs[i]._dy, ls="None", marker=".", color="blue", label="generated")
        plt.xlabel("Time (days)")
        plt.ylabel("Rates (ct/s)")

        print(f"Mean y real : {np.mean(real_lc._y)}")
        print(f"Mean y generated : {np.mean(lcs[i]._y)}")

        print(f"Mean dy real : {np.mean(real_lc._dy)}")
        print(f"Mean dy generated : {np.mean(lcs[i]._dy)}")

        print(f"Std y real : {np.std(real_lc._y)}")
        print(f"Std y generated : {np.std(lcs[i]._y)}")

        print(f"Std dy real : {np.std(real_lc._dy)}")
        print(f"Std dy generated : {np.std(lcs[i]._dy)}")

        plt.figure()
        plt.plot(lcs[i]._y, lcs[i]._dy, ls="None", marker=".", color="blue", label="generated")
        plt.plot(real_lc._y, real_lc._dy, ls="None", marker=".", color="black", label="real")
        plt.xlabel("$y$")
        plt.ylabel("d$y$")

    null_model_gen, null_kernel_gen = define_null_hypothesis(lcs[0], units="seconds")
    plot_model_lc(lcs[0], model=null_model_gen, units="seconds")
    alternative_model_gen, alternative_kernel_gen = define_alternative_model(lcs[0], units="seconds")
    plot_model_lc(lcs[0], model=alternative_model_gen, units="seconds")
    plt.show()