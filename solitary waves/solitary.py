import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import sys


## solve for dispersion relation

# define wavespeed function
def wavespeed(k, b, B):
    """Computes the initial guess for the wave speed, c0.

    Args:
        k (int or ndarray): wavenumber(s)
        b (float): dimensionless rod radius
        B (float): dimensionless magnetic bond number

    Returns:
        float: c, initial guess for the wave speed 
    """

    # bessel functions 
    def I(domain, order=1):
        # modified bessel of first kind
        return sp.iv(order, domain)
    
    def K(domain, order=1):
        # modified bessel of second kind
        return sp.kn(order, domain)

    # parts of the equation 
    one = 1/k
    two = ( (I(k)*K(k*b)) - (I(k*b)*K(k)) ) / ( (I(k*b)*K(k, order=0)) + (I(k, order=0)*K(k*b)) )
    three = (k**2) - 1 + B

    c = np.sqrt(one*two*three)

    return c

# plot dispersion relation for a variety of bond numbers

k = np.linspace(0.1, 14, 1000)
B = np.linspace(1, 30, 5)

fig = plt.figure()

for i in B:
    plt.plot(k, wavespeed(k, 0.000001, i), color="#00264D")

plt.xlabel(r"$k$")
plt.ylabel(r"$c$")
# plt.show()



## plot phase portrait of system 

# set up mesh grid
x, y = np.pi*np.linspace(-1, 3, 20), np.pi*np.linspace(0, 3, 20)
# returns two 2D matrices, one with rows of repeated y values and the other with rows over the whole range of x
xmesh, ymesh = np.meshgrid(x, y) 

# set parameter value
B = 30

# define function for system
def system(alpha, S, B):

    dS = np.sin(alpha)
    dalpha = B/2 - 1 + np.cos(alpha)/S - B/(2*S**2)

    return dalpha, dS

# initialize vector arrays to zero
dx = np.zeros(xmesh.shape)
dy = np.zeros(ymesh.shape)

# compute values of the vectors for all values over the mesh grid
for i in range(len(x)):
    # set values of x and y to evaluate the system at
    xeval = xmesh[i, :]
    # yeval = ymesh[-1 + i, :]
    yeval = ymesh[i, :]

    # in each loop, I want to evaluate the system over the whole x domain for a given y value
    dx[i,:], dy[i,:] = system(xeval, yeval, B) 

print(dx.shape)

fig2 = plt.figure()

# plt.quiver(xmesh/np.pi, ymesh, dx, dy, color='#00264D')
plt.streamplot(x, y, dx, dy, color='#00264D')
plt.xlabel(r"$\alpha / \pi $")
plt.ylabel(r"$S$")
plt.show()
