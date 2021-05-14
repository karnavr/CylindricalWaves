import numpy as np
import matplotlib.pyplot as plt
import sys

# define function to approximate using fourier series
def f(x):
    return np.cos(x) + np.cos(2*x)

# def number of points and domain
N = 100
x = np.linspace(0, 2*np.pi,N)

coeffs = np.fft.fft(f(x), n=N) * 1/N
coeffs = coeffs.real
# coeffs = np.fft.fftshift(coeffs)  # fft shift returns the coefficients in order: negative frequencies, zero f, positive frequencies

print(f"Computed {len(coeffs)} fourier coefficients.")
print(f"Fourier coefficients greater than 1e-4: {coeffs[coeffs>1e-4]}")
print(f"Indices of Fourier coefficients greater than 1e-4: {np.argwhere(coeffs>1e-4)}")


# get fourier coefficients in correct order for what we know to be a real function
# create seperate rows in array for positive coeffs and negative coeffs
if N % 2 == 0:
    
    # initialize empty array for fourier coefficients 
    fourier_coeffs = np.empty((2,int(N/2)))

    # get positive frequency coefficients (plus zero)
    positive_coeffs = coeffs[0:int(N/2)]

    # get negative frequency coeffs
    negative_coeffs = np.flip(coeffs[int(N/2 + 1):]); 
    negative_coeffs = np.append(negative_coeffs, 0)     # put negative frequency components in right order ({-1}, {-2} ...)

    # add coefficients to the fourier coeffs array 
    fourier_coeffs[0,:] = positive_coeffs
    fourier_coeffs[1,:] = negative_coeffs
else:
    raise Exception("Use an even number of sample points/fourier coefficients!")


series = np.empty((np.size(fourier_coeffs), len(x)))      # initialize empty array for the series sum elements
n = np.arange(0, int(N/2), 1)                             # define n array 

# create fourier series terms using purely cos terms
# for i in range(len(fourier_coeffs[0,:])):
#     series[i,:] = fourier_coeffs[0,i] * np.cos(n[i]*x)
#     series[i+50,:] = fourier_coeffs[1,i] * np.cos(-(n[i]-1)*x)

# create fourier series terms using exponential terms
for i in range(len(fourier_coeffs[0,:])):
    series[i,:] = fourier_coeffs[0,i] * np.exp(n[i]*x*1j*2*np.pi*(1/(2*np.pi)))
    series[i+50,:] = fourier_coeffs[1,i] * np.exp(-(n[i]-1)*x*1j*2*np.pi*(1/(2*np.pi)))

# sum all series elements (over domain) for final computation of the function as a fourier series
fourier_function = np.sum(series, 0)  

# plotting
plt.plot(x, f(x), label="cos(x) + cos(2x)", color='#00264D')                       # plot real function (defined in f(x))
plt.plot(x, fourier_function, '--', label='Fourier Approximation', color='red')    # plot fourier approximation

plt.xlabel(r"$\theta$ (rad)", labelpad=10)
plt.ylabel("f", labelpad=10)
plt.legend()

plt.show()