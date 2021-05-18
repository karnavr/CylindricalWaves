import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# define domain 
L = 10*np.pi 
N = 1000

x = np.linspace(-L/2, L/2, N)


S = np.cos(x) * np.exp(-np.power(x,2)/25)
S_prime = -1/25 * S/np.cos(x) * (25*np.sin(x) + 2*x*np.cos(x))

definite_integral = np.trapz(S_prime, x=x)
print(f"Definite Integral {definite_integral}")


# plotting

plt.plot(x, S, label='Wavepacket', color='#00264D')
plt.plot(x, S_prime, label=r'S\'(x)', color='red')
plt.legend()
plt.show()