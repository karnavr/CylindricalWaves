import numpy as np
import scipy.integrate as spint
import scipy.special as sp
import scipy.optimize as so
import matplotlib.pyplot as plt
import sys

import funcs

# define domain constants
L = np.pi
N = 30
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


def mainIntegrand(S, c, z, N, L, b, B, epsilon):

    # define S derivatives (spectral)
    S_z = funcs.fftDeriv(S, z, order=1)
    S_zz = funcs.fftDeriv(S, z, order=2)

    # bessel functions 
    def I(domain, order=1):
        # modified bessel of first kind
        return sp.iv(order, domain)
    
    def K(domain, order=1):
        # modified bessel of second kind
        return sp.kn(order, domain)


    integrand = np.empty((N,len(z))) # initialize array of N integrand equations 

    Szsq = 1 + S_z**2 # commonly used value in eqs

    # get k values (N + 1 values but we discard the eq'n with k=0 in the for loop) 
    k_values = np.arange(-N/2, N/2 + 1, 1)*(np.pi/L)
    i = 0

    for k in k_values:

        if k == 0.0:
            continue # we don't want to include the equation with k = 0 (trivial solution)

        # individual terms
        one_p = (Szsq)*((c**2)/2 - 1/(S*np.sqrt(Szsq)) + S_zz/np.power(Szsq, 3/2) + B/(2*(S**2)) + epsilon)
        one = k*S*np.sqrt(one_p)
        two = K(k*b)*I(k*S) - I(k*b)*K(k*S)
        three = np.cos(k*z)

        # add eq'n to integrand array (each row is one eq'n) 
        integrand[i,:] = one*two*three

        # divide integrand by max over one period to maintain well scaled Jacobian
        integrand[i,:] = integrand[i,:] / np.max(integrand[i,:])
        i += 1

    return integrand


def mainIntegral(S, params):

    # get initial guess array and c value 
    c = S[0]
    S = S[1:]

    # define parameters
    z = params[0]
    N = params[1]
    L = params[2]
    b = params[3]
    B = params[4]
    epsilon = params[5]

    # convert coeffs to real space
    S = funcs.fourierToReal(S, z)

    # get N integrand equations
    integrands = mainIntegrand(S, c, z, N, L, b, B, epsilon)

    equations = np.empty(N) # initialize array of N integral equations

    # define all N integral equations (with trapezium rule)
    for n in range(0, N):
        equations[n] = np.trapz(integrands[n,:], z)

    return equations


# set parameter values
params = [z, N, L, b, B, epsilon]
# include a2 in params (eventually)

# set initial guess (with a0 = 1, very small a1 and non-zero a2)
# initial_guess = 1 + (1e-3)*np.cos(z) + 0.12*np.cos(2*z)
initial_guess = np.zeros(N)
initial_guess[0:3] = np.array([1, 1e-3, 0.12])

# compute initial guess for wave speed value c0
c0 = [funcs.initial_c0(L, b, B)]
initial_guess = np.concatenate((c0,initial_guess))

solution, infodict, ier, msg = so.fsolve(mainIntegral, initial_guess, args = params, full_output=True)

print(f"Solution computed.")
print(f"Solution length: {solution.size}")
print(f"Number of function calls: {infodict['nfev']}")

print(f"Integer Flag: {ier}")
print(msg)

# plotting 

plt.plot(z, funcs.fourierToReal(solution, z), color='#00264D')
plt.plot(z, funcs.fourierToReal(initial_guess, z), '--', color='red')

plt.xlabel("z", labelpad=5)
plt.ylabel("S", labelpad=5)

plt.show()