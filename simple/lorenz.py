import numpy as np
import scipy as sp
import scipy.optimize as so
import matplotlib.pyplot as plt

# Solving for steady states of the Lorenz model/system using Newton's method

def lorenz(u, sigma, r, b):
    """Defines the Lorenz model of ODEs.

    Args:
        u (ndarray): x, y and z values
        sigma, r, b (floats): model parameters 

    Returns:
        array: values of the lorenz ODEs at given x,y and z
    """

    # define x, y and z variables 
    x = u[0]
    y = u[1]
    z = u[2]

    # define system of ODEs (eq. 3.32)
    dx = sigma*(y-x)
    dy = r*x - y - x*z
    dz = x*y - b*z

    return [dx, dy, dz]


initial_guess = np.array([50., 50., 50.])
roots = so.fsolve(lorenz, initial_guess, args=(10., 28, 8./3.))

print(f"Steady states of the lorenz model: {roots}")