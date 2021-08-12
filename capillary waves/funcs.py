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