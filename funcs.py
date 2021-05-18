import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

def fftDeriv(func, domain, method, order = 1):

    if method == 1:
        N = len(domain)
        dx = domain[-1] - domain[-2]

        k = np.fft.fftfreq(N, dx) * 2 * np.pi

        fhat = np.fft.fft(func)

        derivative = np.fft.ifft(k*1j*fhat).real

    elif method == 2:
        fhat = np.fft.fft(func)

        kappa = (2*np.pi/np.abs(x[-1]-x[0]))*np.arange(-len(domain)/2,len(domain)/2)
        kappa = np.fft.fftshift(kappa) 

        dfhat = kappa*fhat*(1j)

        #Inverse Fourier Transform
        derivative = np.real(np.fft.ifft(dfhat))

    else:
        raise RuntimeError("Use a valid method number!")

    return derivative


sys.exit()

## function testing

# Test One 

N = 1000
x = np.linspace(-10, 10, N)

f = 2*np.sin(x) + 2*np.cos(x)
derivative = fftDeriv(f, x, 1)

plt.plot(x, 2*np.cos(x) - 2*np.sin(x), label='Exact value', color='#00264D')
plt.plot(x, derivative, '--', label='Derivative by FFT', color='red')
plt.legend()
plt.show()


# Test Two

N2 = 1000
L = 30
dx = L/N2

x2 = np.arange(-L/2,L/2, dx, dtype="complex_")
f2 = np.cos(x2) * np.exp(-np.power(x2,2)/25)

analytical_derivative2 = -(np.sin(x2) * np.exp(-np.power(x2,2)/25 + (2/25)*x2*f2))
derivative2 = fftDeriv(f2, x2, 3)

plt.plot(x2, analytical_derivative2, label='Exact value', color='#00264D')
plt.plot(x2, derivative2, '--', label='Derivative by FFT', color='red')
plt.legend()
plt.show()


# Test Three

N = 100
x = np.linspace(0, 2*np.pi, N, endpoint=False)

f1 = np.sin(2*x) + np.cos(5*x) 

derivative1 = fftDeriv(f1, x, 2)
analytical_derivative1 = 2*np.cos(2*x)-5*np.sin(5*x)

plt.plot(x, analytical_derivative1, label='Exact value', color='#00264D')
plt.plot(x, derivative1, '--', label='Derivative by FFT', color='red')
plt.legend()
plt.show()