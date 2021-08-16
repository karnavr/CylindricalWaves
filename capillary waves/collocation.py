import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from celluloid import Camera

from funcs import *
import sys

# create domain mesh points

N = 90

z = np.zeros(N)
for i in range(1, N+1):
    z[i-1] = (i - 1)/(2*N -2)   # ranges from 0 to 0.5 (inclusive)


def equations(coeffs, z, s):

    N = len(z)              # number of mesh points

    Bstar = coeffs[0]       # get/remove B* from coeffs array 
    coeffs = coeffs[1:]     # now should be N coeffs 

    # convert from fourier space to real space (eq. 5.5)
    F = 0
    for i in range(0, N):
        F += coeffs[i] * np.cos(i*2*np.pi*z)

    
    # define N equations using eq. 5.3 
    equations = np.zeros(N+1)

    for i in range(0, N):
        # define needed derivatives at current mesh point 
        F_z = finDiff1D(i, 1, F, z)
        F_zz = finDiff1D(i, 2, F, z) 

        # define K (eq. 2.6)
        Kone = F_zz/(np.power(1 + F_z**2, 3/2))
        Ktwo = 1/np.sqrt(1 + F_z**2)
        kthree = 1/F[i]

        K = Kone - Ktwo*kthree

        equations[i] = Bstar + K  # eq. 5.3


    # define one more eq. using eq. 5.7 (N + 1 eq's total now)
    equations[N] = F[0] - F[-1] - s

    # print(f"mean eq. value = {np.mean(equations)}")


    return equations


# set steepness
s = 0.18

# create initial guess (N coeffs)
initial_guess = np.zeros(N)
initial_guess[0:2] = [0.17, 0.09]

# convert initial guess to real space (for later plotting)
profile_initial = fourierToReal(initial_guess, z)


# add B* guess to initial guess
Bstar_initial = [5.8]
initial_guess = np.concatenate([Bstar_initial, initial_guess])

# solve using fsolve
solution, infodict, ier, msg = so.fsolve(equations, initial_guess, args=(z, s), full_output=True, maxfev=10000)

print("\n\n~SIMULATION DETAILS~")
print(f"INTEGER FLAG: {ier}")
print(f"MESSAGE: {msg}")

print(f"calls = {infodict['nfev']}")
print(f"average function eval at output = {np.average(infodict['fvec'])}")

print("\n\n~RESULTS~")
print(f"s = {s}")
print(f"B* = {solution[0]}")


# convert solution from fourier space to real space 
solution_coeffs = solution[1:]
profile_solution = fourierToReal(solution_coeffs, z)

# create other half of wave profiles (symmetric)
z_full = np.concatenate([z, z + 0.5])
profile_solution = np.concatenate([profile_solution, np.flip(profile_solution)])
profile_initial = np.concatenate([profile_initial, np.flip(profile_initial)])


# plotting 
plt.plot(z_full, profile_initial, '--', color='red', label='initial guess')
plt.plot(z_full, profile_solution, color='#00264D', label='solution')

plt.legend()


# compute points up bifurcation branch
# note that computing points up the branch is quite sensitive to changes in the number of branch points and 
# the amount that you nudge the B* guess by for each solution on the branch 

bifurcation = True
if bifurcation == True:

    print("\n\n~COMPUTING BIFURCATON BRANCH~")

    branch_points = 111      # number of points on bifrucation branch

    # create initial guess for first branch point 
    s = np.linspace(0.001, 0.5, branch_points)

    initial_guess = np.zeros(N)         # create initial guess (N coeffs)
    initial_guess[0:2] = [0.17, 0.09]

    Bstar_initial = [6.97]              # add Bstar guess to initial guess (for s = 0.001)
    # Bstar_initial = [2*np.pi]
    initial_guess = np.concatenate([Bstar_initial, initial_guess])


    # initialize arrays for solutions (later include wave profiles too)
    Bstars = np.zeros(branch_points)
    profiles_coeffs = np.zeros((branch_points, N))  # coeffs of solution profiles
    guess_coeffs = np.zeros((branch_points, N))     # coeffs of each guess on branch with 1:1 correspondance w above


    # solve up branch for all steepness values
    for i in range(0, branch_points):

        # solve for current s value 
        solution, infodict, ier, msg = so.fsolve(equations, initial_guess, args=(z, s[i]), full_output=True, maxfev=10000)

        # print(f"B* = {solution[0]}")
        print(f"{(i+1)}/{len(s)}")

        # print when the integer flag is anything other than 1
        if ier != 1:
            print(ier)

        # write solutions to array
        Bstars[i] = solution[0]
        guess_coeffs[i,:] = initial_guess[1:]  # capture guess leading to current solution 
        profiles_coeffs[i,:] = solution[1:]    # capture coeffs of current solution

        # update initial guess to be current solution
        solution[0] += -0.0001  # nudge B* guess for better behaviour
        initial_guess = solution

    
    # plot bifurcation branch 
    fig = plt.figure()
    plt.plot(s, Bstars, color='#00264D')
    plt.xlabel('s')
    plt.ylabel(r'$B^*$')


    ## get all solution/guess profiles ready for plotting

    # convert from fourier to real space 
    profiles = np.zeros_like(profiles_coeffs)
    guesses = np.zeros_like(guess_coeffs)

    for i in range(branch_points):
        profiles[i,:] = fourierToReal(profiles_coeffs[i,:], z)
        guesses[i,:] = fourierToReal(guess_coeffs[i,:], z)

    # extend over whole domain
    profiles = np.concatenate([profiles, np.flip(profiles, 1)], axis=1)
    guesses = np.concatenate([guesses, np.flip(guesses, 1)], axis=1)


    ## create bifurcation solutions animation
    create_animation = True
    if create_animation == True:

        print("\n\n~CREATING ANIMATION~")

        # profiles animation
        fig = plt.figure()
        camera = Camera(fig)

        for i in range(branch_points):

            # plot one figure for each branch point and snap using Camera 
            plt.plot(z_full, guesses[i,:], '--', color='red', label='initial guess')
            plt.plot(z_full, profiles[i,:], color='#00264D', label='solution')
            plt.xlabel('z', labelpad=5)
            plt.ylabel('S', labelpad=5)

            camera.snap()

        animation = camera.animate()
        animation.save('profiles.gif')

        # bifurcation branch animation
        fig = plt.figure()
        camera = Camera(fig)

        for i in range(branch_points):

            # plot one figure for each branch point and snap using Camera 
            plt.plot(s[:i], Bstars[:i], '.', color='#00264D')    # plot first i branch points on fig
            plt.xlabel('s', labelpad=5)
            plt.ylabel(r'$B^*$', labelpad=5)

            plt.xlim(min(s)-0.02, max(s)+0.02)
            plt.ylim(min(Bstars)-0.2, max(Bstars)+0.2)

            camera.snap()

        animation = camera.animate()
        animation.save('bifurcation.gif')




plt.show()




# todo:
# * test finite derivative function for correct behaviour (seperate python file with print statements haha)
# * check over derivative formulae for correctness

# * compute solutions up the bifurcation branch using numerical continuation
# colour points in bifurcation depending on how many iterations it took to reach the solution or 
#   maybe even what the average func eval is at those steepness values
# plot the function eval at output for the values up the branch
