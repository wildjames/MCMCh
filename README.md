# MCMCh
MCMCh - MCMC Chisq Melding for Characterisation.


Scipy's optimise functions can yield an error, but sometimes this is not trivial. This will characterise the error with an MCMC chain, allowing a fairly easy characterisation of complex covariances with little legwork.

## Usage

```
import mcmch
import numpy as np

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

## Test data ##
test_x = np.linspace(0.0, 10.0, 8)
test_y = model_func(test_x, m_actual, c_actual)

# Add error to simulated data
y_err  = 0.1 * np.ones_like(test_y)
test_y = test_y + (y_err * np.random.randn(*test_y.shape))

mean, err_lo, err_hi = mcmch.MCMCh(model_func, initial_guess, test_x, test_y, y_err)

print("Results:")
print(mean)
```

