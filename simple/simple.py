import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# define domain 
L = 10*np.pi 
N = 1000

x = np.linspace(-L/2, L/2, N)

S = np.cos(x) * np.exp(-np.power(x,2)/25)


# plotting

plt.plot(x, S, label='Wavepacket', color='#00264D')
plt.legend()
plt.show()