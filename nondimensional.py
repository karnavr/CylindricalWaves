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
z = np.linspace(-L, L, N+1) # N points (used for now because I'm not zero padding)
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

    Szsq = 1 + (S_z**2) # commonly used value in eqs

    # get k values (N + 1 values but we discard the eq'n with k=0 in the for loop) 
    # k_values = np.arange(-N/2, N/2 + 1, 1)*(np.pi/L)
    k_values = np.arange(0, N + 1, 1)*(np.pi/L) # N + 1 values but we end up with N eq'ns (discard k=0 in loop)
    i = 0

    for k in k_values:

        if k == 0.0:
            continue # we don't want to include the equation with k = 0 (trivial solution)

        # individual terms
        one_p = (Szsq)*((c**2)/2 - 1/(S*np.sqrt(Szsq)) + S_zz/np.power(Szsq, 1.5) + B/(2*(S**2)) + epsilon)
        one = k*S*np.sqrt(one_p)
        two = K(k*b)*I(k*S) - I(k*b)*K(k*S)
        three = np.cos(k*z)

        # add eq'n to integrand array (each row is one eq'n) 
        integrand[i,:] = one*two*three
        
        # divide integrand by max over one period to maintain well scaled Jacobian
        integrand[i,:] = integrand[i,:] / np.max(integrand[i,:])
        i += 1

    return integrand


def mainIntegral(coeffs, params):

    # get N + 1 fourier coefficients and c value 
    c = coeffs[0]
    coeffs = coeffs[1:]

    # define parameters
    z = params[0]
    N = params[1]
    L = params[2]
    b = params[3]
    B = params[4]
    epsilon = params[5]

    # convert coeffs to real space
    S = funcs.fourierToReal(coeffs, z)

    # get N integrand equations
    integrands = mainIntegrand(S, c, z, N, L, b, B, epsilon)

    equations = np.empty(N+2) # initialize array of N + 2 equations

    # define all N integral equations (with trapezium rule)
    for n in range(0, N):
        equations[n] = np.trapz(integrands[n,:], z)

    # define 2 more equations (needed for the N + 2 unknowns)
    a0 = coeffs[0]
    a2 = coeffs[2]
    equations[N] = a0 - 1 # fix value of a0 to be 1
    equations[N+1] = np.abs(a2 - 0.01) # ensure non-zero a2 value

    return equations


# set parameter values
params = [z, N, L, b, B, epsilon]
# include a2 in params (eventually)

# set initial guess (with a0 = 1, very small a1 and non-zero a2)
# initial_guess = 1 + (1e-3)*np.cos(z) + 0.12*np.cos(2*z)
initial_guess = np.zeros(N+1) # N + 1 coeffs, but we are concatenating with c0 so will be N + 2 unknowns into fsolve 
initial_guess[0:3] = np.array([1.0, 1e-3, 0.12]) # a0, a1, and a2 coefficients 

# compute initial guess for wave speed value c0
# c0 = [funcs.initial_c0(L, b, B)] # computes to approx. 0.815 (for fig. 2 parameters)
c0 = [1.079]

# update initial guess vector
initial_guess = np.concatenate((c0,initial_guess)) # size N + 2 now 

solution, infodict, ier, msg = so.fsolve(mainIntegral, initial_guess, args = params, full_output=True, maxfev=10**5, factor=100)

print(f"Mean difference = {np.mean(solution-initial_guess)}")

print(f"Solution computed.")
print(f"Solution length: {solution.size}")
print(f"Number of function calls: {infodict['nfev']}")

print(f"Integer Flag: {ier}")
print(msg)

# print(infodict['fvec'])


# change solution and initial guess arrays to be exclusively (N + 1) fourier coeffs (discard c value)
solution = solution[1:]
initial_guess = initial_guess[1:]


# plotting 

plt.plot(z, funcs.fourierToReal(solution, z), color='#00264D', label='solution')
plt.plot(z, funcs.fourierToReal(initial_guess, z), '--', color='red', label='initial guess')

plt.xlabel("z", labelpad=5)
plt.ylabel("S", labelpad=5)
plt.legend()

plt.show()

## TO-DO:
# set new eq. like |a_2| - (small number) = 0 
# (and add an extra fourier coeff such that the # of k values and the # of coeffs are the same)