'''
This code produces functions for Direct Numeical Simulation (DNS) of Navier-Stokes
'''

import numpy as np
from scipy.io import savemat


def dealias( w ):
    n = w.shape[0]

    k = np.arange(0,n,1)
    bad = k>n/2
    k[bad] = k[bad] - n
    
    kx = np.reshape( k, [n,1] )
    ky = np.reshape( k, [1,n] )


    mask = kx*kx + ky*ky > n*n/9
    wf = np.fft.fft2(w)
    wf[mask] = 0
    w = np.real(np.fft.ifft2(wf))
    return w



def time_deriv( w, nu, forcing ):
    '''
    PURPOSE:
    w - a [n,n] matrix representing the vorticity field
    '''

    n = w.shape[0]

    k = np.arange(0,n,1)
    bad = k>n/2
    k[bad] = k[bad] - n
    
    kx = np.reshape( k, [n,1] )
    ky = np.reshape( k, [1,n] )

    i = 1j
    to_u = i*ky/(kx*kx + ky*ky)
    to_v =-i*kx/(kx*kx + ky*ky)
    to_u[0,0] = 0
    to_v[0,0] = 0
    
    wf = np.fft.fft2(w)
    
    u  = np.real(np.fft.ifft2(to_u*wf))
    v  = np.real(np.fft.ifft2(to_v*wf))
    wx = np.real(np.fft.ifft2(i*kx*wf))
    wy = np.real(np.fft.ifft2(i*ky*wf))

    lap = np.real(np.fft.ifft2( -(kx*kx + ky*ky)*wf ))

    rhs = -u*wx-v*wy + nu*lap + forcing 

    rhs = dealias(rhs)
    return rhs

def rk4_step(w, dt, nu, forcing):
    k1 = time_deriv(w,           nu, forcing)
    k2 = time_deriv(w + dt*k1/2, nu, forcing)
    k3 = time_deriv(w + dt*k2/2, nu, forcing)
    k4 = time_deriv(w + dt*k3,   nu, forcing)
    
    return w + (k1 + 2*k2 + 2*k3 + k4)*dt/6

def generate_traj(w0, dt, nu, forcing, timesteps):
    n = w0.shape[0]
    w = np.zeros( (n, n, timesteps+1) )
    w[:,:,0] = w0

    for i in range(timesteps):
        w[:,:,i+1] = rk4_step(w[:,:,i], dt, nu, forcing)
    return w

# Define the number of points along x and y axes
n = 64  # Adjust this as needed

# Generate the x and y values in the range [0, 2*pi]
x_1d = np.linspace(0, 2*np.pi, n, endpoint=False)  # Exclude endpoint to make the domain periodic
y_1d = np.linspace(0, 2*np.pi, n, endpoint=False)
x, y = np.meshgrid(x_1d, y_1d, indexing='ij')

w0 = np.cos(3*y)*np.sin(x-1) + np.sin(2*x + y)*np.cos(y-2) + np.cos(4*x-1)*np.sin(y)
forcing = 4*np.cos(4*y)

dt = 0.025
nu = 1.0/40
timesteps = 1024*8

w = generate_traj(w0, dt, nu, forcing, timesteps)

#Transform to streamfunction
k = np.arange(0,n,1)
bad = k>n/2
k[bad] = k[bad] - n    
kx = np.reshape( k, [n,1] )
ky = np.reshape( k, [1,n] )

to_psi = 1.0/(kx*kx + ky*ky)
to_psi[0,0] = 0 #get rid of infinity from 1/0
to_psi = np.expand_dims( to_psi, 2 )

wf = np.fft.fft2(w, axes=(0, 1))
psi = np.real( np.fft.ifft2(  to_psi * wf, axes=(0, 1) ) )

my_dict = {"psi": psi, "w": w, "x": x, "y": y, "to_psi": to_psi}
savemat("w_traj.mat", my_dict)