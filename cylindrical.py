import numpy as np
import scipy.integrate as spint
import scipy.special as sp
import scipy.optimize as so
import matplotlib.pyplot as plt
import sys

# define domain constants
L = np.pi
N = 1000
dz = 2*L/(2*N + 1)

# define domain
z = np.arange(-L, L, dz)
print(len(z))

# define magnetic constants 
B = 1.5
b = 0.1

epsilon = 1 - B/2


## define integral components (eq. 2.18)
# bessel functions
bess_first = sp.jv(1, z)
bess_sec = sp.yn(1, z)


def mainIntegrand(S, z, N, L, b, gamma, rho, V, epsilon):

    # define components in integrand
    kappa = - (S_zz/np.power(1 + S_z**2, 3/2)) + (1/(S*np.sqrt(1 + S_z**2)))
    F = ((gamma*kappa)/rho) - V - epsilon

    # bessel functions 
    def I(domain, order=1):
        # bessel of first kind
        return sp.jv(order, domain)
    
    def K(domain, order=1):
        # bessel of second kind
        return sp.yn(order, domain)


    integrand = np.empty(N) # initialize array of N integrand equations 

    # get k values (101 values but we discrad the eq'n with k=0 in the for loop) 
    k_values = np.arange(-N/2, N/2 + 1, 1)*(np.pi/L)
    i = 0

    for k in k_values:

        if k == 0.0:
            continue # we don't want to include the equation with k = 0 (trivial solution)

        # individual terms
        one = k*S*np.sqrt((1 + S_z**2)*(c**2 - 2*F))
        two = K(k*b)*I(k*S) - I(k*b)*K(k*S)
        three = np.cos(k*z)

        # add eq'n to integrad array  
        integrand[i] = one*two*three
        i += 1

    return integrand

def mainIntegral():

    # define all N integral equations
    np.trapz()

    # append all equations to an array  


    return equations

