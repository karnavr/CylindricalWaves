import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
import sys

## Solving for the equilibrium position of a mass-spring system using Newton's method

# set constants 
g = 9.81 
m = 0.1
k1 = 10
k2 = 20
L1 = L2 = 0.1
D = 0.1

def spring(u, param):
    """computes force (negative gradient) of the potential energy of the 2D mass-spring system.
    """
    # define x and y 
    x = u[0]
    y = u[1]

    # define parameters 
    g = param[0]
    m = param[1]
    k1 = param[2]
    k2 = param[3]
    L1 = param[4]
    L2 = param[5]
    D = param[6]

    # define dx 
    dx_1 = ( k2*(x-D)*(np.sqrt( (x-D)**2 + y**2 ) - L2)  )/( np.sqrt( (x-D)**2 + y**2) )
    dx_2 = (k1*x*( np.sqrt(x**2 + y**2) - L1 ))/(np.sqrt(x**2 + y**2))

    dx = -(dx_1 + dx_2) 

    # define dy
    dy_1 = (k2*y*(np.sqrt(y**2 + (x-D)**2) - L2))/(np.sqrt(y**2 + (x-D)**2))
    dy_2 = (k1*y*(np.sqrt(y**2 + x**2) - L1))/(np.sqrt(y**2 + x**2))
    dy_3 = -g*m

    dy = -(dy_1 + dy_2 + dy_3)

    return [dx, dy]

# set parameter values and initial guess
initial_guess = [0.05, 0.1]
params = [g, m, k1, k2, L1, L2, D]

equil_positions = so.fsolve(spring, initial_guess, args=params)

print(f"Equilibirum positions: x = {np.round(equil_positions[0], 5)}, y = {np.round(equil_positions[1], 5)}")


## Observing the change in position with variable k2 constand using numerical continuation

# now k2 is the parameter we vary (all other parameters are kept the same as before)
k2 = np.linspace(0, 20, 1000)

equilibirum_positions = np.empty((len(k2),2)) # initialize array for equilibirum_positions

initial_guess = [0.,0.1]

for i in range(len(k2)):
    equilibirum_positions[i,:] = so.fsolve(spring, initial_guess, args=[g, m, k1, k2[i], L1, L2, D])

    initial_guess = equilibirum_positions[i,:]

print(f"Continution computation complete.")

