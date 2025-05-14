import matplotlib.pyplot as plt
import numpy as np


if False:
    # Plot the p-value as a function of sigma_noise of the simulation
    p_vals = [0.000,0.129,0.000001437,0.44314096,0.06595355,0.00011677,4.482736685377486E-08]
    sigma_noises = [10,5,3,4,4.5,6,3.5]
    plt.plot(sigma_noises,p_vals, "o")
    plt.axhline(0.05, linestyle="--", color="red")
    plt.xlabel(r"$\sigma_\mathrm{noise}$", fontsize=16)
    plt.ylabel(r"p-value", fontsize=16)
    plt.show()


# Plot the TLRT p-values as a function of the number of datapoints
# All of these used P_qpo=100 and a timerange = 1000 days, so all have 10 cycles of data

p_vals = [0.10210821, 0.00000063, 0.00890458, 0.00005573, 0.00000016]
n_sigmas = [1.2696301328349295, 4.8456948482440625, 2.369562107619295, 3.8642001750315975, 5.115379004922388]
n_pts = [100, 150, 200, 250, 300]

plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
ax1.plot(n_pts, p_vals, "o")
ax1.set_xlabel(r"$N$", fontsize=16)
ax1.set_ylabel(r"p-value", fontsize=16)
ax2.plot(n_pts, n_sigmas, "o")
ax2.set_xlabel(r"$N$-datapoints", fontsize=16)
ax2.set_ylabel(r"$n_\sigma$", fontsize=16)
plt.suptitle(r"$P_\mathrm{qpo}=100$, $\Delta t_\mathrm{tot} = 1000$, $P_\mathrm{drw}=100$, $Q=80$, $\sigma_\mathrm{noise}=1$")
plt.subplots_adjust(hspace=0)
plt.show()