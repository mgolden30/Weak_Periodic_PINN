'''
This code produces functions for Direct Numeical Simulation (DNS) of Navier-Stokes.
Running this script will compute several trajectories with RK4 from random smooth intial conditions.
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

def random_initial_data(n, amplitude):
    '''
    Generate some random fourier modes
    '''

    #generate a whole mesh of random Fourier coefficients
    wf = (2*np.random.rand(n,n)-1) + 1j*(2*np.random.rand(n,n)-1)

    #Set mean vorticity to zero since it has to be
    wf[0,0] = 0
    
    #throw out high mode information
    k = np.arange(0,n,1)
    bad = k>n/2
    k[bad] = k[bad] - n
    kx = np.reshape( k, [n,1] )
    ky = np.reshape( k, [1,n] )
    high_modes = (kx*kx + ky*ky > 16)
    wf[high_modes] = 0

    #transform to real space
    w = np.real( np.fft.ifft2(wf) )
    
    #Rescale to max value is equal to amplitude
    w = amplitude * w / np.max(np.max(np.abs(w)))
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



if __name__ == "__main__":
    # Define the number of points along x and y axes
    n      = 64   # resolution
    trials = 64   # number of initial conditions
    dt     = 0.025
    nu     = 1.0/40

    timesteps = 1024*4 #overall per trial
    transient = 1024*2 #Don't save timesteps less than this
    stride    = 16     #record every "stride" timesteps

    amplitude = 10  #max vorticity value for initial data

    # Generate the x and y values in the range [0, 2*pi]
    x_1d = np.linspace(0, 2*np.pi, n, endpoint=False)  # Exclude endpoint to make the domain periodic
    y_1d = np.linspace(0, 2*np.pi, n, endpoint=False)
    x, y = np.meshgrid(x_1d, y_1d, indexing='ij')
    forcing = 4*np.cos(4*y)

    #allocate memory for all trials
    ss = slice(transient, timesteps, stride) #subsampling index, cut off some of transient
    ns = len(range(ss.start, ss.stop, ss.step)) #count the elements of ss
    w  = np.zeros( (n,n,ns,trials) ) #store all trajectories here

    for tr in range(trials):
        print(f"trial {tr} / {trials}")
        w0 = random_initial_data(n, amplitude)
        w[:,:,:,tr] = generate_traj(w0, dt, nu, forcing, timesteps)[:,:,ss]

    my_dict = { "w": w, "x": x, "y": y }
    savemat("w_traj.mat", my_dict)