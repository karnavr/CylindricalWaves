import numpy as np
import scipy.integrate as spint
import scipy.special as sp
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

bernoulli_constant = 1 - B/2


## define integral components (eq. 2.18)
# bessel functions
bess_first = sp.jv(1, z)
bess_sec = sp.yn(1, z)


def mainIntegrand():
    return

