import numpy as np
import scipy.integrate as spint
import scipy.special as sp
import scipy.optimize as so
import matplotlib.pyplot as plt
import sys

import funcs

# define domain constants
L = np.pi
N = 50
dz = 2*L/(2*N + 1)

# define domain
# z = np.arange(-L, L, dz) # 2N + 1 domain points
z = np.linspace(-L, L, N) # N points (used for now because I'm not zero padding)
print(len(z))

# define magnetic constants 
B = 1.5
b = 0.1

epsilon = 1 - B/2


## define integral components (eq. 2.18)
# bessel functions
bess_first = sp.jv(1, z)
bess_sec = sp.yn(1, z)


def mainIntegrand(S, z, N, L, b, B, epsilon):

    # define S derivatives (spectral)
    S_z = funcs.fftDeriv(S, z, order=1)
    S_zz = funcs.fftDeriv(S, z, order=2)

    # bessel functions 
    def I(domain, order=1):
        # bessel of first kind
        return sp.jv(order, domain)
    
    def K(domain, order=1):
        # bessel of second kind
        return sp.yn(order, domain)


    integrand = np.empty((N,len(z))) # initialize array of N integrand equations 

    Szsq = 1 + S_z**2 # commonly used value in eqs

    # get k values (101 values but we discard the eq'n with k=0 in the for loop) 
    k_values = np.arange(-N/2, N/2 + 1, 1)*(np.pi/L)
    i = 0

    for k in k_values:

        if k == 0.0:
            continue # we don't want to include the equation with k = 0 (trivial solution)

        # individual terms
        c = 1.07
        one = k*S*np.sqrt((Szsq)*((c**2)/2 - 1/(S*np.sqrt(Szsq)) + S_zz/np.power(Szsq, 3/2) + B/(2*(S**2)) + epsilon))
        two = K(k*b)*I(k*S) - I(k*b)*K(k*S)
        three = np.cos(k*z)

        # add eq'n to integrand array (each row is one eq'n) 
        integrand[i,:] = one*two*three

        # divide integrand by max over one period to maintain well scaled Jacobian
        integrand[i,:] = integrand[i,:] / np.max(integrand[i,:])
        i += 1

    return integrand


def mainIntegral(S, params):

    # # define parameters 
    # parameters = [z, N, L, b, B, epsilon]
    
    # for i in range(0,5):
    #     parameters[i] = params[i]

    z = params[0]
    N = params[1]
    L = params[2]
    b = params[3]
    B = params[4]
    epsilon = params[5]

    # get N integrand equations
    integrands = mainIntegrand(S, z, N, L, b, B, epsilon)

    equations = np.empty(N) # initialize array of N integral equations

    # define all N integral equations (with trapezium rule)
    for n in range(0, N):
        equations[n] = np.trapz(integrands[n,:], z)

    return equations


# set parameter values
params = [z, N, L, b, B, epsilon]

# set initial guess (simple cosine wave in real space for now)
initial_guess = np.cos(z)


solution, infodict, ier, msg = so.fsolve(mainIntegral, initial_guess, args = params, full_output=True)

print(f"Solution computed.")
print(f"Solution length: {solution.size}")
print(f"Number of function calls: {infodict['nfev']}")

print(f"Integer Flag: {ier}")
print(msg)


# plotting 

plt.plot(z, solution, color='#00264D')

plt.xlabel("z", labelpad=5)
plt.ylabel("S", labelpad=5)

plt.show()