import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.gaussian_process import kernels,GaussianProcessRegressor
## check version
import sys
import sklearn


np.random.seed(0)
n=50

kernel_ =[kernels.RBF (),

         #kernels.RationalQuadratic(),

         #kernels.ExpSineSquared(periodicity=10.0),

         #kernels.DotProduct(sigma_0=1.0)**2,

         #kernels.Matern()
         ]
print(kernel_, '\n')



for kernel in kernel_:

    # Gaussian process

    gp = GaussianProcessRegressor(kernel=kernel)

    # Prior

    x_test = np.linspace(-5, 10, n).reshape(-1, 1)
    mu_prior, sd_prior = gp.predict(x_test, return_std=True)
    samples_prior = gp.sample_y(x_test, 3)

    # plot

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.plot(x_test, mu_prior)
    plt.fill_between(x_test.ravel(), mu_prior - sd_prior,
                     mu_prior + sd_prior, color='aliceblue')
    plt.plot(x_test, samples_prior, '--')
    plt.title('Prior')

    # Fit

    x_train = np.array([-4, -3, -2, -1, 1,3,6,8,10]).reshape(-1, 1)
    y_train = np.sin(x_train)
    gp.fit(x_train, y_train)

# posterior

    mu_post, sd_post = gp.predict(x_test, return_std=True)
    mu_post = mu_post.reshape(-1)
    samples_post = np.squeeze(gp.sample_y(x_test, 3))

    # plot

    plt.subplot(1, 2, 2)
    plt.plot(x_test, mu_post)
    plt.plot(x_test, np.sin(x_test), color="black")
    plt.fill_between(x_test.ravel(), mu_post - sd_post,
                     mu_post + sd_post, color='aliceblue')
    plt.plot(x_test, samples_post, '--')
    plt.scatter(x_train, y_train, c='blue', s=50)
    plt.title('Posterior')

    plt.show()

    print(gp.kernel_, gp.kernel_)
    print(gp.log_marginal_likelihood, gp.log_marginal_likelihood(gp.kernel_.theta))

    print('-'*50, '\n\n')