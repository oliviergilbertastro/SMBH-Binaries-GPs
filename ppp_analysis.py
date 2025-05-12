"""
Code to calculate the posterior predictive p-value (PPP), which calculates the likelihood ratio test (LRT) distribution
"""

from mind_the_gaps.lightcurves import GappyLightcurve
from mind_the_gaps.gpmodelling import GPModelling
from mind_the_gaps.models.psd_models import BendingPowerlaw, Lorentzian, SHO, Matern32, Jitter
from mind_the_gaps.models.celerite_models import Lorentzian as Lor
from mind_the_gaps.models.celerite_models import DampedRandomWalk
from mind_the_gaps.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import celerite, corner
from scipy.stats import percentileofscore

cpus = 10 # set the number of cores for parallelization
np.random.seed(10)

drw_data = np.loadtxt("simulations/DRW.txt")
drw_qpo_data = np.loadtxt("simulations/DRW_QPO.txt")

times, noisy_countrates, dy, exposures = drw_data[:,0], drw_data[:,1], drw_data[:,2], drw_data[:,3]
input_drw_lc = GappyLightcurve(times, noisy_countrates, dy, exposures=exposures)

fig = plt.figure()
plt.errorbar(times, noisy_countrates, yerr=dy, ls="None", marker=".")
plt.xlabel("Time (days)")
plt.ylabel("Rates (ct/s)")

times, noisy_countrates, dy, exposures = drw_qpo_data[:,0], drw_qpo_data[:,1], drw_qpo_data[:,2], drw_qpo_data[:,3]
input_drw_qpo_lc = GappyLightcurve(times, noisy_countrates, dy, exposures=exposures)

fig = plt.figure()
plt.errorbar(times, noisy_countrates, yerr=dy, ls="None", marker=".")
plt.xlabel("Time (days)")
plt.ylabel("Rates (ct/s)")

plt.show()