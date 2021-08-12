import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

import sys

# create domain mesh points

N = 90

z = np.zeros(N)
for i in range(1, N+1):
    z[i-1] = (i - 1)/(2*N -2)   # ranges from 0 to 0.5 (inclusive)


def equations(coeffs, z, s):

    N = len(z)              # number of mesh points

    Bstar = coeffs[0]       # get/remove B* from coeffs array 
    coeffs = coeffs[1:]     # now should be N coeffs 

    # convert from fourier space to real space (eq. 5.5)
    F = 0
    for i in range(0, N):
        F += coeffs[i] * np.cos(i*2*np.pi*z)

    # define finite difference derivative (derivative in real space)
    def finDer(point, order, func, domain):
        """Computes finite difference derivative at given point in domain.s 

        Args:
            point (int): index in domain at which to compute derivative
            order (int): derivative order
            func (1D array): function values over domain
            domain (1D array): domain values corresponding to func

        Returns:
            float: derivative of func at point in domain
        """

        N = len(domain)
        h = domain[4] - domain[3]   # set spacing 

        if point == 0: # forward difference derivatives (for first point)
            if order == 1:
                derivative = (func[point + 1] - func[point])/h
            elif order == 2:
                derivative = (func[point+2] - 2*func[point+1] + func[point])/(h**2)
            else:
                raise ValueError('enter a valid derivative order')
        elif point == (N - 1): # backward difference derivatives (for last point)
            if order == 1:
                derivative = (func[point] - func[point-1])/h
            elif order == 2:
                derivative = (func[point] - 2*func[point-1] + func[point-2])/(h**2)
            else:
                raise ValueError('enter a valid derivative order')
        else:
            if order == 1: # centered derivatives (for intermediate points)
                derivative = (func[point+1] - func[point-1])/(2*h)
            elif order ==2:
                derivative = (func[point+1] + func[point-1] - 2*func[point])/(h**2)
            else:
                raise ValueError('enter a valid derivative order')

        return derivative

    
    # define N equations using eq. 5.3 
    equations = np.zeros(N+1)

    for i in range(0, N):
        # define needed derivatives at current mesh point 
        F_z = finDer(i, 1, F, z)
        F_zz = finDer(i, 2, F, z) 

        # define K (eq. 2.6)
        Kone = F_zz/(np.power(1 + F_z**2, 3/2))
        Ktwo = 1/np.sqrt(1 + F_z**2)
        kthree = 1/F[i]

        K = Kone - Ktwo*kthree

        equations[i] = Bstar + K  # eq. 5.3


    # define one more eq. using eq. 5.7 (N + 1 eq's total now)
    equations[N] = F[0] - F[-1] - s

    print(f"mean eq. value = {np.mean(equations)}")


    return equations


# set steepness
s = 0.6

# create initial guess (N coeffs)
initial_guess = np.zeros(N)
initial_guess[0] = 0.17
initial_guess[1] = 0.09

profile_initial = 0
for i in range(0, N):
    profile_initial += initial_guess[i] * np.cos(i*2*np.pi*z)


# add Bstar guess to initial guess
Bstar_initial = [5.8]
initial_guess = np.concatenate([Bstar_initial, initial_guess])

# print(initial_guess[1:])

# solve using fsolve
solution, infodict, ier, msg = so.fsolve(equations, initial_guess, args=(z, s), full_output=True, maxfev=10000)

print("\n\n~SIMULATION DETAILS~")
print(f"INTEGER FLAG: {ier}")
print(f"MESSAGE: {msg}")

print(f"calls = {infodict['nfev']}")
print(f"average function eval at output = {np.average(infodict['fvec'])}\n \n")
# # print(f"fvec = {infodict['fvec']}\n \n")


# convert solution from fourier space to real space 
solution_coeffs = solution[1:]
profile_solution = 0 
for i in range(0, N):
    profile_solution += solution_coeffs[i] * np.cos(i*2*np.pi*z)


# plotting 
plt.plot(z, profile_initial, '--', color='red', label='initial guess')
plt.plot(z, profile_solution, color='#00264D', label='solution')

plt.legend()
plt.show()

# todo:
# * test finite derivative function for correct behaviour (seperate python file with print statements haha)
# * check over derivative formulae for correctness