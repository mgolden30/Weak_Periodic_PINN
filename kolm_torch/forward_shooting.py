import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import savemat

from integrate_NS import implicit_explicit_step, to_streamfunction

N = 128  #points per side
M = 2**12  #timesteps
every = 32 #plot every
dt = 0.02 #timestep
every = 8
nu = 1.0/40 #viscosity

grid_1d = np.arange(N)/N * 2*np.pi

x,y = np.meshgrid( grid_1d, grid_1d )

forcing = 4*np.cos(4*y)

#initial condition
w = np.sin(3*x) + np.cos(x-2*y+1) - np.cos(5*x-0.7-y)

#store the entire trajectory
ws = np.zeros([N,N,M])



ws[:,:,0] = w
for i in range(M-1):
    w = ws[:,:,i]
    for j in range(every):
        w = implicit_explicit_step(w, dt, nu, forcing)
    ws[:,:,i+1] = w


psi = to_streamfunction(ws)


out_dict = {"w":ws, "dt":dt, "nu":nu, "every":every, "psi": psi}
savemat( "kolm.mat", out_dict )
exit()