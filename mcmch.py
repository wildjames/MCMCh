import scipy.optimize as opt
import emcee
import numpy as np
import matplotlib.pyplot as plt
import seaborn
try:
    import triangle
    # This triangle should have a method corner
    # There are two python packages with conflicting names
    getattr(triangle, "corner")
except (AttributeError, ImportError):
    # We want the other package
    import corner as triangle


def chisq_func(model_args, model_function, x_data, y_data, yerr=None, ln_like=False):
    '''Call the model function on the x_data, compute chisq on the observations.
    Model takes (x, *model_args)'''

    # I need model_args to be a toople
    model_args = tuple(model_args)
    model_data = model_function(x_data, *model_args)

    if yerr is None:
        yerr = np.ones_like(y_data)

    chisq = np.sum( ((y_data - model_data) / yerr)**2 )

    if ln_like:
        return -0.5*chisq

    return chisq

def scipy_opt(model_func, initial_guess, data, *args, **kwargs):
    '''Scipy convenience function. *args and **kwargs are passed to scipy.optimise.'''
    # Construct the argument tuple for chisq_func
    chisq_args = (model_func, *data)

    # Fit with scipy
    result = opt.minimize(chisq_func, initial_guess, args=chisq_args, *args, **kwargs)
    return result

def plot_data_model(func, x, y, ye=None, title='', func_args=(), func_kwargs={}):
    func_args = tuple(func_args)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.errorbar(x, y, ye, linestyle='', marker='x')
    ax.plot(x, func(x, *func_args, **func_kwargs), color='red', zorder=0)

    plt.show()

def thumbPlot(chain, labels, **kwargs):
    seaborn.set(style='ticks')
    seaborn.set_style({"xtick.direction": "in","ytick.direction": "in"})
    fig = triangle.corner(chain, labels=labels, bins=50,
                          label_kwargs=dict(fontsize=18), **kwargs)
    return fig


def MCMCh(model_func, initial_guess, x_data, y_data, err_data=None, nwalkers=10, nsteps=5000, scat_factor=5.0, loud=3):
    '''This is a marriage between scipy optimise and emcee. Suitable only for
    simple functions, designed to be naiive to the model it's fitting.

    The data are first fitted using scipy's optimise routine. The realistic
    error on that fit is then characterised by emcee. This should often work off
    the bat, but DO NOT us this for complex functions or models! A thorough
    understanding of MCMC methods are necessary for anything substantial!

    Inputs:
    -------
      model_func: function
        This is the model that you're testing against. A simple chisq test will
        be perfomed on this output.
      initial_guess: list, array
        The initial values of the parameters to be fitted, in the order they're
        accepted by model_func
      x_data: array
        The x data to be tested.
      y_data: array
        The y data to be tested.
      err_data: array, optional
        The error in the y data.
      nwalkers: int
        The number of walkers to be used in the MCMC chain. In general,
        more=better, but more=takes longer
      nsteps: int
        The number of steps to iterate the ensemble over. In general,
        more=better, but more=takes longer
      scat_factor: float
        The hessian matrix produced by scipy's optimise is used to scatter the
        parameters in prep for the MCMC chain. The matrix is first multiplied
        by scat_factor, then gaussian scatter is applied using this value for
        sigma.
      loud: int
        How loudly to perform the fit.
          0: print nothing

          1: print the initial guess,
             plot the scipy fit,
             report the step of the MCMC,
             plot a corner plot of the variables

          2: Inherit the above,
             print the results of the MCMC

          3: Inherit the above.
             Plot the initial guess,
             print the full report from scipy

    Outputs:
    --------
      mean, lo_error, up_error: np.array, np.array, np.array:
        The mean, down error, and up error of the MCMC results.
        Caution: If the variables are significantly non-gaussian,
        this can be iffy.
    '''
    # # # # # # # # # # # # #
    # Fit with scipy first  #
    # # # # # # # # # # # # #

    chisq_init = chisq_func(initial_guess, model_func, test_x, test_y, y_err)
    if loud > 0:
        print("Initial guess: {} -> chisq = {:.2f}".format(initial_guess, chisq_init))

    if loud > 2:
        plot_data_model(model_func, test_x, test_y, y_err, 'Pre-fit', func_args=initial_guess)

    data = (test_x, test_y, y_err)

    fit = scipy_opt(model_func, initial_guess, data)
    p_0 = fit['x']
    if loud > 2:
        print("Initial pass with scipy:\n", fit)

    if loud > 0:
        plot_data_model(model_func, test_x, test_y, y_err, 'Post-fit', func_args=p_0)

    if not fit['success']:
        print("Scipy's optimise failed to pass! Proceeding may or may not be dangerous...")
        cont = input("Shall I continue, and try fitting with MCMC anyway? y/n: ")
        if not cont.lower().startswith('y'):
            exit()

    # # # # # # # # # # # # # # # # # # #
    # Sample the solution with an MCMC  #
    # # # # # # # # # # # # # # # # # # #

    # p_0 must have (npars) for each walker. i.e. shape must be (nwalkers, ndim)
    ndim = len(p_0)
    p_0 = np.array([np.array(p_0) for _ in range(nwalkers)])

    # Use the Hessian matrix as a guide for the initial scatter.
    scatter = [fit['hess_inv'][i,i] for i in range(ndim)]
    scatter = [(scatter * np.random.randn(ndim))*scat_factor for _ in range(nwalkers)]

    # Apply the scatter matrix
    p_0 += scatter

    # Make a tuple of the args for the chisq function.
    chisq_args = (model_func, *data)

    # Sample from the MCMC
    if loud > 0:
        print('sampling...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, chisq_func, args=chisq_args, kwargs={'ln_like': True})
    for i, _ in enumerate(sampler.sample(p_0, iterations=nsteps, storechain=True)):
        if loud > 0:
            if not i%10:
                print("\r  Step {}".format(i), end='')
    if loud > 0:
        print()

    lo, mean, hi = np.percentile(sampler.flatchain, [16, 50, 84], axis=0)

    if loud > 1:
        for par_lo, par_mean, par_hi in zip(lo, mean, hi):
            print("Found: {:<8.3f}  +{:<8.3f}  -{:<8.3f}".format(par_mean, par_hi-par_mean, par_mean-par_lo))
        print("--------------------------------------------------------")
        print("Done!\n\n")

    if loud > 0:
        thumbPlot(sampler.flatchain, ['m', 'c'])
        plt.show()

    return mean, mean-lo, hi-mean


if __name__ == "__main__":
    np.random.seed(4321)

    def model_func(x, m, c):
        '''Basic straight line model'''
        return (m*x) + c

    # These are the ''true'' parameters
    m_actual = 0.5
    c_actual = 3.5

    # And these are my initial guesses for the above. They MUST be in the
    # same order that they appear in model_func!
    initial_guess = (
        10.0,
        10.0
    )

    # Test data
    test_x = np.linspace(0.0, 10.0, 8)
    test_y = model_func(test_x, m_actual, c_actual)
    # Add error to simulated data
    y_err  = 0.1 * np.ones_like(test_y)
    test_y = test_y + (y_err * np.random.randn(*test_y.shape))


    mean, err_lo, err_hi = MCMCh(model_func, initial_guess, test_x, test_y, y_err, nsteps=1000, loud=3)

    print("Plotting...")
    print(mean)
    plot_data_model(model_func, test_x, test_y, title='Returned from function', func_args=mean)