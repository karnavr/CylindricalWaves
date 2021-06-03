import numpy as np
import scipy as sp
import scipy.special as sp
import matplotlib.pyplot as plt
import sys

def fftDeriv(func, domain, order = 1):
    """Computes the nth derivative using the FFT.

    Args:
        func (1D array): the values of the function to be differentiated
        domain (1D array): the domain over which the function is defined
        order (int, optional): derivative order. Defaults to 1.

    Returns:
        1D array: values of nth derivative of function at over domain
    """

    # get number of domain points and spacing
    N = len(domain)
    dx = domain[-1] - domain[-2]

    # compute wavenumbers (spatial frequency)
    k = np.fft.fftfreq(N, dx) * 2 * np.pi

    # compute fourier transform
    fhat = np.fft.fft(func)

    # multiply in fourier space for derivative and return to original space
    derivative = np.fft.ifft( ((k*1j)**order) * fhat).real

    return derivative


def initial_c0(L, b, B):
    """Computes the initial guess for the wave speed, c0.

    Args:
        L (float): length of half of the solution domain
        b (float): dimensionless rod radius
        B (float): dimensionless magnetic bond number

    Returns:
        float: c0, initial guess for the wave speed 
    """

    # define initial parameters
    k1 = np.pi/L 

    # bessel functions 
    def I(domain, order=1):
        # modified bessel of first kind
        return sp.iv(order, domain)
    
    def K(domain, order=1):
        # modified bessel of second kind
        return sp.kn(order, domain)

    # parts of the equation 
    one = 1/k1
    two = (I(k1)*K(k1*b) - I(k1*b)*K(k1)) / (I(k1*b)*K(k1, order=0) + I(k1, order=0)*K(k1*b))
    three = k1**2 - 1 + B

    c0 = np.sqrt(one*two*three)

    return c0


def fourierToReal(coefficients, domain):
    """Computes N points in real space (as cosine series) using N fourier coefficients. 

    Args:
        coefficients (1D array): array of fourier coefficients (a0, a1, ...)
        domain (1D array): domain points in real space

    Raises:
        TypeError: if the number of coefficients is un-equal to number of domain points

    Returns:
        1D array: values in real space
    """

    # we want the same number of points output as the number of coeffs 
    if len(coefficients) != len(domain):
        raise TypeError("there are un-equal number of fourier coefficients and domain points!")

    # initialize output signal to be zero
    S = 0 

    # iterate through all cosine modes corresponding to the coefficients
    for a in range(0, len(coefficients)):
        S += coefficients[a]*np.cos(a*domain)

    # can later perhaps use the np.fft/rfft funcs to generalize this to waves that 
    # are not even around z = 0 (and potentially increase speed)

    return S

# sys.exit()

# ## function testing

# # Test One 

# N = 1000
# x = np.linspace(-10, 10, N)

# f = 2*np.sin(x) + 2*np.cos(x)
# derivative = fftDeriv(f, x, 1)

# plt.plot(x, 2*np.cos(x) - 2*np.sin(x), label='Exact value', color='#00264D')
# plt.plot(x, derivative, '--', label='Derivative by FFT', color='red')
# plt.legend()
# plt.show()

# # sys.exit()

# # Test Two

# N2 = 1000
# L = 30
# dx = L/N2

# x2 = np.arange(-L/2,L/2, dx, dtype="complex_")
# f2 = np.cos(x2) * np.exp(-np.power(x2,2)/25)

# analytical_derivative2 = - ((f4/np.cos(x4))*(25*np.sin(x4)+2*x4*np.cos(x4)))/25
# derivative2 = fftDeriv(f2, x2, 3)

# plt.plot(x2, analytical_derivative2, label='Exact value', color='#00264D')
# plt.plot(x2, derivative2, '--', label='Derivative by FFT', color='red')
# plt.legend()
# plt.show()


# # Test Three

# N = 100
# x = np.linspace(0, 2*np.pi, N, endpoint=False)

# f1 = np.sin(2*x) + np.cos(5*x) 

# derivative1 = fftDeriv(f1, x, 2)
# analytical_derivative1 = 2*np.cos(2*x)-5*np.sin(5*x)

# plt.plot(x, analytical_derivative1, label='Exact value', color='#00264D')
# plt.plot(x, derivative1, '--', label='Derivative by FFT', color='red')
# plt.legend()
# plt.show()


# # Test Four

# N = 1000
# L = 30
# dx = L/N

# x = np.arange(-L/2,L/2, dx, dtype="complex_")
# f = np.cos(x) * np.exp(-np.power(x,2)/25)

# # analytical derivatives
# first_derivative = - ((f/np.cos(x))*(25*np.sin(x)+2*x*np.cos(x)))/25
# second_derivative = ((f/np.cos(x))*(100*x*np.sin(x)+(4*(x**2) - 675)*np.cos(x)))/625

# # fft derivative
# derivative = fftDeriv(f, x, 2)

# plt.plot(x, second_derivative, label='Exact value', color='#00264D')
# plt.plot(x, derivative, '--', label='Derivative by FFT', color='red')
# plt.legend()
# plt.show()