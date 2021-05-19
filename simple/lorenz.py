import numpy as np
import scipy as sp
import scipy.optimize as so
import matplotlib.pyplot as plt

# Solving for steady states of the Lorenz model/system using Newton's method

def polynomial(x):
    return x**2 + 2*x

root = so.fsolve(polynomial, -3)

print(f"Root of polynomial: {root}")