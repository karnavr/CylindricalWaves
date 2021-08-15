import numpy as np

# function returning finite difference derivatives
def finDiff(j, i, direction, order, r, domainArray):
    """Finite difference derivative at given mesh point in a given mesh.

    Args:
        j, i (ints): indicies of interest mesh point on r mesh
        direction (string): direction derivative is taken in
        order (int): derivative order (1 or 2)
        r (2D array): the mesh in which the derivative is taken
        domainArray (1D array): derivative is taken wrt this domain array (direction)

    Returns:
        (float): derivative value at given mesh point (j,i) using a finite difference scheme
    """

    # get domain array spacing
    h = domainArray[-3] - domainArray[-4]

    # define finite difference eqns 
    if direction == 'phi':
        
        if order == 1: # first order (centered)
            derivative = (r[j, i+1] - r[j, i-1])/(2*h)
        elif order == 2: # second order (centered)
            derivative = (r[j,i+1] + r[j, i-1] - 2*r[j,i])/(h**2)
        else:
            raise ValueError('invalid order value')
        
    elif direction == 'psi':
        
        if order == 1: # first order (backward)
            derivative = (r[j, i] - r[j-1,i])/h
        elif order == 2: # second order (backward)
            derivative = (r[j, i] - 2*r[j-1, i] + r[j-2, i])/(h**2)
        else:
            raise ValueError('invalid order value')
        
    else:
        raise ValueError('unknown direction string given')


    return derivative

# print(finDiff(1,1, direction='pi', order=1, r=1, domainArray=phi)) # finite difference function test
# sys.exit()

# function to flatten and reverse-flatten array
def sculptArray(option, array, newshape=None):
    """Converts a 2D array to a 1D array or vice-versa, keeping the identical elements.

    Args:
        option (string): 'flatten' or 'unflatten'
        array (array): array to be sculpted
        newshape (tuple): shape of resulting array if option = 'unflatten' 

    Returns:
        [array]: re-shaped array with same elements as input array
    """

    if option == 'flatten':
        array = array.flatten()  # flatten array to 1D

    elif option == 'unflatten': 
        if array.ndim != 1:
            raise TypeError('array must be 1D to sculpt to two dimensions!')
        
        array = np.reshape(array, newshape) # unflatten 1D array to 2D array with given shape
    else:
        raise ValueError('select a valid option for sculpting the array')
    
    return array

# one dimensional finite difference derivatives (in real space)
def finDiff1D(point, order, func, domain):
    """Computes finite difference derivative at given point in domain.s 

    Args:
        point (int): index in domain at which to compute derivative
        order (int): derivative order
        func (1D array): function values over domain
        domain (1D array): domain values corresponding to func

    Returns:
        float: derivative of func at point in domain
    """

    N = len(domain)
    h = domain[4] - domain[3]   # set spacing 

    if point == 0: # forward difference derivatives (for first point)
        if order == 1:
            derivative = (func[point + 1] - func[point])/h
        elif order == 2:
            derivative = (func[point+2] - 2*func[point+1] + func[point])/(h**2)
        else:
            raise ValueError('enter a valid derivative order')
    elif point == (N - 1): # backward difference derivatives (for last point)
        if order == 1:
            derivative = (func[point] - func[point-1])/h
        elif order == 2:
            derivative = (func[point] - 2*func[point-1] + func[point-2])/(h**2)
        else:
            raise ValueError('enter a valid derivative order')
    else:
        if order == 1: # centered derivatives (for intermediate points)
            derivative = (func[point+1] - func[point-1])/(2*h)
        elif order ==2:
            derivative = (func[point+1] + func[point-1] - 2*func[point])/(h**2)
        else:
            raise ValueError('enter a valid derivative order')

    return derivative

# given coeffs, converts from fourier space to real space (eq. 5.5)
def fourierToReal(coeffs, domain):
    """Converts from fourier space to real space over given domain.

    Args:
        coeffs (1D array): array of fourier coefficients (a0, a1, ...)
        domain (1D array): domain points in real space

    Returns:
        1D array: values in real space
    """

    N = len(coeffs)

    # initialize to be zero 
    real = 0 

    # iterate through all cosine modes corresponding to the coefficients (eq. 5.5)
    for i in range(N):
        real += coeffs[i] * np.cos(i*2*np.pi*domain)

    return real