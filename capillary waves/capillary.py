import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

from funcs import *
import sys

# set parameter values (steepness and psi extremity)
s = 0.001
Q = 1/(2*np.pi**2)          # eq. 5.1


## create mesh grid
M = 27   # phi direction
N = 34   # psi direction

# initialize mesh arrays
phi = np.zeros(M+2)         # (horizontal axis)
psi = np.zeros(N)           # same as t^2 (vertical axis)

# populate mesh arrays
for i in range(1, M+1):
    # sets the the middle i = 1 to M points over a range of 0 <= phi <= 1/2 (eq. 4.15)
    phi[i] = (i-1)/(2*(M-1)) # note that phi index is identical to i index in notes

for j in range(1,N+1):
    # goes from 0 <= psi <= Q (eq. 4.16)
    psi[j-1] = np.sqrt(Q)*((j-1)/(N-1))
    psi[j-1] = (psi[j-1])**2

print('\n\n~PRE-SIMULATION~')
print(f"Mesh points (unknowns) = {(phi.size - 2) * psi.size}")
print(f"steepness = {s}")
print(f"Q = {Q}")


# define function returning MN + 2 eqs for MN + 2 unknowns
def equations(r, psi, phi, s):

    # get M and N constants
    M = len(phi) - 2
    N = len(psi)

    # check that r is a correctly sized array (the initial state/condition of the solution r(phi, psi))
    if len(r) != (M*N + 2):
        print(f"initial guess array has {len(r)} elements")
        raise ValueError('r does not have the correct number of elements!')

    # pull alpha and B out of r guess vector
    alpha = r[0]
    B = r[1]

    r = r[2:]  # pull alpha and B out of the vector

    # reshape r array to be 2D (NM elements)
    r = sculptArray('unflatten', r, newshape=(N, M))

    # add extra columns to r array (now exactly reflects mesh described in paper and notes)
    r = np.c_[np.zeros(N), r, np.zeros(N)]

    # initialize equations array (2D for now), same dimensions as r array (N, M + 2)
    equations = np.zeros_like(r)
    # note: this should actually be smaller, do not need the i=0 and i=M+1 columns (these are only BCs in the r)
    # these columns are removed near the end of this func to reflect the number of unknowns

    ## define I = 0 and I = M+1 symmetry boundary conditions (eq. 4.18 & 4.19)
    r[:,0] = r[:, 2]
    r[:,-1] = r[:, -3]

    ## get M(N-2) equations in the center of the domain
    for i in range(1, M+1):
        for j in range(1,N-1):

            # define the needed derivatives 
            phi_1 = finDiff(j, i, 'phi', 1, r, phi)
            phi_2 = finDiff(j, i, 'phi', 2, r, phi)

            psi_1 = finDiff(j, i, 'psi', 1, r, psi)
            psi_2 = finDiff(j, i, 'psi', 2, r, psi)

            # define equation at current mesh point (r position maps directly onto equations array)
            equations[j, i] = (r[j, i]**3)*psi_2 + r[j,i]*phi_2 + (r[j, i]**2)*((psi_1)**2) - phi_1**2  # (eq. 4.6)

    ## set free surface condition and bottom surface condition (2M more equations)
    for i in range(1, M+1):
        # define the needed derivatives
        j = N-1 # we only need these at the top surface
        phi_1 = finDiff(j, i, 'phi', 1, r, phi)
        phi_2 = finDiff(j, i, 'phi', 2, r, phi)

        psi_1 = finDiff(j, i, 'psi', 1, r, psi)
        psi_2 = finDiff(j, i, 'psi', 2, r, psi)

        # define second order partial needed in K1
        psi_plus = finDiff(j, i+1, 'psi', 1, r, psi)
        psi_minus = finDiff(j, i-1, 'psi', 1, r, psi)
        h_phi = phi[-3] - phi[-4]
        second_order_partial = (psi_plus - psi_minus)/(2*h_phi)

        # define K (eq. 4.8)
        j = N-1
        K1 = r[j,i]*psi_1*phi_2 - (phi_1**2)*(psi_1) - r[j,i]*phi_1*(second_order_partial)
        K2 = np.power(phi_1**2 + (r[j,i]**2)*(psi_1**2), [-3/2])
        K3 = np.sqrt((phi_1**2 + (r[j,i]**2)*(psi_1**2))**(-1))
        K4 = np.absolute(psi_1)

        K = K1*K2 - K3*K4

        # free/top surface (bottom of r and equations array)
        j = N-1
        equations[N-1,i] = 0.5*((phi_1**2 + (r[j,i]**2)*(psi_1**2))**(-1)) - alpha*K - B # eq. 4.7

        # bottom surface (top of arrays)
        j = 0
        equations[0, i] = r[j,i] - 0 # eq. 4.9


    ## define 2 more equations for last two unknowns (& add to end of the equations vector)
    # fix the steepness of the wave (eq. 4.20), applies to r at the free surface (j = N-1)
    eq1 = r[N-1, 1] - r[N-1, M] - s

    # integral equation (eq. 4.21) (works for any j = const value, see eq. 2.7)
    integrand_r = np.zeros(M)
    integrand_deriv = np.zeros(M)
    for i in range(1, M+1):
        j = 5
        integrand_r[i-1] = r[j,i]
        integrand_deriv[i-1] = finDiff(j, i, 'psi', 1, r, psi)

    integrand = integrand_r*integrand_deriv

    eq2 = np.trapz(integrand, phi[1:M+1]) - 0.5  # take integral to define eq. 4.21


    # remove extra (i = 0 and i = M+1) columns from equations array (MN elements now)
    equations = equations[:,1:-1]

    # reshape equations array to be a vector
    equations = sculptArray('flatten', equations) 

    # add the two extra equations to the equations vector (MN+2 elements now)
    equations = np.append(equations, [eq1, eq2])


    # check if we have the correct number of equations (must be MN + 2)
    if len(equations) != (M*N + 2):
        print(f"{len(equations)} defined equations for {M*N + 2} unknowns")
        raise ValueError('enough equations are not defined for the number of unknowns')

    return equations



# set alpha and B initial guesses
alpha = 0.3042      # good initial guess for the initial (almost) linear wave
R = np.sqrt(2*Q)    # eq. 3.22 (needed to compute R)
B = 0.5 + alpha/R   # eq. 3.3


## create initial guess vector
# input r array (will be converted to same shape as entire mesh grid as notes in the equations func)
r = np.zeros((N, M)) + 0.1

# set free surface (j = N-1) to be a cosine (we know what the solution should look like from phi = 0 to phi = 1/2)
x = np.linspace(0, np.pi, M) # same # of elements as phi mesh points
# profile_guess = 0.001*np.cos(x) + 0.1
# r[N-1, :] = profile_guess

# set initial condition for r as given by r = sqrt(2*psi)
for i in range(0, M):
    r[:,i] = np.sqrt(2*psi)

# uncomment to use a cosine initial condition
x = np.linspace(0, np.pi, M)
r[N-1, :] = 0.001*np.cos(x) + 0.2

profile_guess = r[N-1,:]


 
# plot initial guess array 
fig = plt.figure()
# plt.imshow(r, cmap='gray', vmin=0, vmax=1)
plt.imshow(r, cmap='gray')
plt.colorbar()
plt.title('initial guess (r array)', pad=10)
plt.xlabel(r'$I \ (\phi \ index)$', labelpad=8)
plt.ylabel(r'$J \ (\psi \ index)$', labelpad=8)
# plt.show()
# sys.exit()

# convert to a 1D array (MN elements)
r = sculptArray('flatten', r)



initial_guess = np.concatenate(([alpha, B], r)) # MN + 2 elements

print(f"initial guess has {len(initial_guess)} elements and MN+2 = {M*N + 2}\n\n")
# print(initial_guess)




## solve system using fsolve
solution, infodict, ier, msg = so.fsolve(equations, initial_guess, args=(psi, phi, s), full_output=True, maxfev=10000)
# solution = so.newton(equations, initial_guess, args=(psi, phi, s))

print("~SIMULATION DETAILS~")
print(f"INTEGER FLAG: {ier}")
print(f"MESSAGE: {msg}")

print(f"nfev = {infodict['nfev']}")
print(f"average function eval at output = {np.average(infodict['fvec'])}\n \n")
# print(f"fvec = {infodict['fvec']}\n \n")




# convert solution vector from fsolve into wave profile
alpha_sol = solution[0] # remove alpha and B from solution
B_sol = solution[1]

print("~SOLUTION PARAMETERS~")
print(f"alpha = {alpha_sol} ({(np.abs(alpha-alpha_sol)/alpha)*100} % change)")
print(f"B = {B_sol} ({(np.abs(B-B_sol)/B)*100} % change)\n \n")

profile = solution[-M:] # get profile (last M elements of the solution array, see initial guess array to see exactly why)
profile_domain = np.linspace(0, 0.5, M) # create profile domain (only half domain for now, will reflect later)



# plot results
fig2 = plt.figure()
plt.plot(profile_domain, profile, color='#00264D', label='solution')
plt.plot(profile_domain, profile_guess, '--', color='red', label='initial guess')

plt.xlabel('half domain', labelpad=5)
plt.ylabel("S", labelpad=5)
plt.legend()

plt.show()

bifurcation = False
if bifurcation == True:

    print(f"COMPUTING BIFURCATION BRANCH")

    ## number of points to compute up branch
    branch_points = 6
    

    ## initialize initial guess for first point on branch

    alpha = 0.3042                  # set alpha and B initial guess
    R = np.sqrt(2*Q)                # eq. 3.22 (needed to compute R)
    B = 0.5 + alpha/R               # eq. 3.3

    r = np.zeros((N, M))            # create initial guess vector

    for i in range(0, M):           # set initial condition for r as given by r = sqrt(2*psi)
        r[:,i] = np.sqrt(2*psi)

    # uncomment to use a cosine initial condition
    x = np.linspace(0, np.pi, M)
    r[N-1, :] = 0.001*np.cos(x) + 0.2

    r = sculptArray('flatten', r)   # convert to a 1D array (MN elements)
    initial_guess = np.concatenate(([alpha, B], r)) # MN + 2 elements

    
    # steepness values to use (bounds of bifurcation curve)
    s = np.linspace(0.000001, 0.6, branch_points)

    # create arrays for solutions
    alphas = np.zeros(len(s))
    B = np.zeros(len(s))

    k = 0       # counter for array indices

    ## compute solution up bifurcation branch for each s value 
    for i in s:

        # solve for current s value
        solution, infodict, ier, msg = so.fsolve(equations, initial_guess, args=(psi, phi, i), full_output=True, maxfev=10000)

        # write solutions to array
        alphas[k] = solution[0] 

        # update initial guess along with alpha and B
        initial_guess = solution

        k += 1  # update counter by one
        print(f"progress: {k}/{len(s)}")
        print(f"mean func eval at output = {np.average(infodict['fvec'])}")

    # plotting bifurcation 
    fig = plt.figure()
    plt.plot(alphas, s, 'o')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('s')
    plt.show()



# TO-DO: 
# * check over finite difference formulae for correctness
# * fix integral equation (computing integrands is not done correctly)
# * test trapz with a gaussian curve with unit area
# * compute more solutions up the branch and create bifurcation curve
# * check if the parameters i'm using are reasonable 
# try three-point formulae for finite difference derivatives
# * use a new (guess) value for B (using eq. 3.22 and 3.3)

# things to ask
# free surface location confusion in paper and arrays in numpy 
# reasonable initial guess to use
