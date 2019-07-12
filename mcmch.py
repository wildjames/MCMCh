import scipy.optimize as opt
import emcee
import numpy as np
import matplotlib.pyplot as plt


def chisq_func(model_function, x_data, y_data, yerr=None, *args, **kwargs):
    '''Call the model function on the x_data, compute chisq on the observations.
    Model takes (x, *args, **kwargs)'''

    model_data = model_function(x_data, *args, **kwargs)

    if yerr is None:
        yerr = np.ones_like(y_data)

    chisq = np.sum( ((y_data - model_data) / yerr)**2 )

    return chisq

def scipy_opt(model_func, x_data, y_data, yerr=None, *args, **kwargs):
    '''Scipy convenience function'''



if __name__ == "__main__":

    def model_func(x, m, c):
        '''Basic straight line model'''
        return (m*x) + c

    # Test data
    m_actual = 0.5
    c_actual = 3.5

    test_x = np.linspace(0.0, 10.0, 8)
    test_y = model_func(test_x, m_actual, c_actual)
    y_err  = 1.0

    test_y = test_y + (y_err * np.random.randn(*test_y.shape))

    # Plotting area
    fig, ax = plt.subplots()
    ax.errorbar(test_x, test_y, y_err*np.ones_like(test_y), linestyle=None)


    plt.show()

