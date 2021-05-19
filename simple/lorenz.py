import numpy as np
import scipy.optimize as so
import scipy.integrate as spint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# solving phase space trajectory for the lorenz model 

def lorenztraj(t, state):

    # define x, y and z variables 
    x = state[0]
    y = state[1]
    z = state[2]

    # define system of ODEs (eq. 3.32)
    dx = 10*(y-x)
    dy = 28*x - y - x*z
    dz = x*y - (8/3)*z

    return [dx, dy, dz]

initial_condition = [1., 1., 20.]   # initial condition of x, y and z
t_eval = np.linspace(0,14,1000)

# solve ODE problem
solution = spint.solve_ivp(lorenztraj, [0,14], initial_condition, t_eval=t_eval)

# un-pack wanted results
t = solution.t
solution_values = solution.y
x = solution_values[0,:]
y = solution_values[1,:]
z = solution_values[2,:]

print(f"Number of solution points computed: {len(t)}")


# plotting individual solutions
variables = ['x', 'y', 'z']
solutions = (x, y, z)

for i in range(0, 3):

    fig = plt.figure()

    plt.plot(t, solutions[i], color='#00264D')
    plt.xlabel('t', labelpad=10)
    plt.ylabel(variables[i], labelpad=10)
    plt.show()


# plotting phase space plot 

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(x, y, z,'-')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

sys.exit()

# Solving for steady states of the Lorenz model/system using Newton's method

def lorenz(u, param):
    """Defines the Lorenz model of ODEs.

    Args:
        u (ndarray): x, y and z values
        param (ndarray): sigma, r, b (model parameters)

    Returns:
        array: values of the lorenz ODEs at given x,y and z
    """

    # define x, y and z variables 
    x = u[0]
    y = u[1]
    z = u[2]

    # define parameters 
    sigma = param[0]
    r = param[1]
    b = param[2]

    # define system of ODEs (eq. 3.32)
    dx = sigma*(y-x)
    dy = r*x - y - x*z
    dz = x*y - b*z

    return [dx, dy, dz]


initial_guess = np.array([50., 50., 50.])
roots, info_dict, _, _ = so.fsolve(lorenz, initial_guess, args=[10., 28., 8./3.], full_output=True)

print(f"Steady states of the lorenz model: {roots}")

print(f"Number of function calls: {info_dict['nfev']}")