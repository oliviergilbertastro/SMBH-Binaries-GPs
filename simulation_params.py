import matplotlib.pyplot as plt
import numpy as np



# Plot the p-value as a function of sigma_noise of the simulation
p_vals = [0.000,0.129,0.000001437,0.44314096,0.06595355,0.00011677]
sigma_noises = [10,5,3,4,4.5,6]
plt.plot(sigma_noises,p_vals, "o")
plt.axhline(0.05, linestyle="--", color="red")
plt.xlabel(r"$\sigma_\mathrm{noise}$", fontsize=16)
plt.ylabel(r"p-value", fontsize=16)
plt.show()
