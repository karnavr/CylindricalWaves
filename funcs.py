import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def fftDeriv(f, x, order = 1):

    N = len(x)
    dx = x[-1] - x[-2]

    k = np.fft.fftfreq(N, dx) * 2 * np.pi

    fhat = np.fft.fft(f)

    derivative = np.fft.ifft(k*1j*fhat).real

    return derivative

    