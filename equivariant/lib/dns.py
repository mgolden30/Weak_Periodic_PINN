'''
This code produces functions for Direct Numeical Simulation (DNS) of Navier-Stokes.
Running this script will compute several trajectories with RK4 from random smooth intial conditions.
'''

import torch as torch
import numpy as np


from scipy.io import savemat


def wavenumbers(n):
    #Compute the k vectors for a given grid
    k = torch.arange(0,n,1)
    bad = k>n/2
    k[bad] = k[bad] - n
    k = k*1.0

    kx = torch.reshape( k, [n,1] )
    ky = torch.reshape( k, [1,n] )
    return kx, ky

def dealias( w ):
    n = w.shape[-1]
    kx, ky = wavenumbers(n)
    mask = kx*kx + ky*ky > n*n/9

    wf = torch.fft.fft2(w)
    wf[..., mask] = 0

    w = torch.real( torch.fft.ifft2(wf) )
    return w



def random_initial_conditions(n, amplitude, batch):
    '''
    Generate a batched set [b,n,n] of initial conditions
    '''

    #generate a whole mesh of random Fourier coefficients
    wf = (2*np.random.rand(batch,n,n)-1) + 1j*(2*np.random.rand(batch,n,n)-1)
    wf = torch.tensor(wf)

    #Set mean vorticity to zero
    wf[:,0,0] = 0
    
    kx, ky = wavenumbers(n)
    kmax = 4
    high_modes = (kx*kx + ky*ky > kmax**2)
    wf[:, high_modes] = 0

    #transform to real space
    w = torch.real( torch.fft.ifft2(wf) )
    
    #Rescale to max value is equal to amplitude
    w_max = torch.max(torch.abs(w), dim=1, keepdim=True)[0]
    w_max = torch.max( w_max,       dim=2, keepdim=True)[0]
    
    w = amplitude * w / w_max

    return w


def time_deriv( w, nu, forcing, geometric=False ):
    '''
    PURPOSE:
    w - a [...,n,n] matrix representing a set of vorticity fields. 
    I want to use this in my neural network training, in which w will be of size [b,c,n,n] where b and c are arbitrary
    '''

    #Flatten to something of size [b,n,n]
    batch_shape = w.shape[:-2]
    w       = torch.flatten( w, 0, len(batch_shape)-1 )
    forcing = torch.flatten( forcing, 0, len(batch_shape)-1 )

    n = w.shape[-1] #last index of array
    kx, ky = wavenumbers(n)

    kx = torch.unsqueeze(kx, 0) #[1,n,1]
    ky = torch.unsqueeze(ky, 0) #[1,1,n]

    to_u =  1j*ky / (kx*kx + ky*ky)
    to_v = -1j*kx / (kx*kx + ky*ky)
    to_u[:,0,0] = 0
    to_v[:,0,0] = 0

    wf = torch.fft.fft2(w)

    u   = torch.real(torch.fft.ifft2(  to_u*wf ))
    v   = torch.real(torch.fft.ifft2(  to_v*wf ))
    wx  = torch.real(torch.fft.ifft2( 1j*kx*wf ))
    wy  = torch.real(torch.fft.ifft2( 1j*ky*wf ))
    lap = torch.real(torch.fft.ifft2( -(kx*kx + ky*ky)*wf ))

    rhs = -(u*wx + v*wy) + nu*lap + forcing 
    rhs = dealias(rhs)

    if geometric:
        #If the geometric flag is turned on, it means we want to normalize the L2 norm of the time derivative
        #This is a regularization in time and makes the integration have the same path, but with a different "time"
        #This is fine for generating data, but not for forward time integration (unless you compute the transformation back to physical time)

        #I want the mean of rhs^2 to be 1
        scale = torch.mean( rhs*rhs, dim=[1,2], keepdim=True )
        rhs = rhs / torch.sqrt( scale )

    #Restore the batch dimensions
    rhs = torch.unflatten( rhs, 0, batch_shape )

    return rhs


def rk4_step(w, dt, nu, forcing, geometric=False):
    k1 = time_deriv(w,           nu, forcing, geometric=geometric)
    k2 = time_deriv(w + dt*k1/2, nu, forcing, geometric=geometric)
    k3 = time_deriv(w + dt*k2/2, nu, forcing, geometric=geometric)
    k4 = time_deriv(w + dt*k3,   nu, forcing, geometric=geometric)
    
    return w + (k1 + 2*k2 + 2*k3 + k4)*dt/6


def grid(n):
    # Generate the x and y values in the range [0, 2*pi]
    x_1d = np.linspace(0, 2*np.pi, n, endpoint=False)  # Exclude endpoint to make the domain periodic
    y_1d = np.linspace(0, 2*np.pi, n, endpoint=False)
    x, y = np.meshgrid(x_1d, y_1d, indexing='ij')
    return x, y




if __name__ == "__main__":


    # Define the number of points along x and y axes
    n     = 64       # spatial resolution
    batch = 128       # number of initial conditions
    num_outputs = 256  #Number of recorded outputs per trajectory
    
    x, y   = grid(n)
    forcing= 4*np.cos(4*y)

    nu     = 1.0/40.0 # viscosity
    dt     = 1.0/50.0  # timestep
    
    t_vis      = 1.0/(16*nu) #viscous timescale 1/k^2/nu assuming dominant k=4 wavenumber
    transient  = 10*t_vis   #integrate this far forward before recording data
    stride     = 8       #record every "stride" timesteps
    amplitude  = 10   #typical vorticity value for initial data

    trans_step = round( transient/dt )

    print(f"Generating turbulent training data with nu={nu}.")
    print(f"Transient is {transient}, which will require {trans_step} timesteps...\n")

    forcing = torch.tensor(forcing)
    forcing = forcing.repeat((batch,1,1)) #repeat over batch dimension

    w  = torch.zeros( (batch, num_outputs, n, n) )

    #generate initial data
    w0 = random_initial_conditions(n, amplitude, batch)


    #integrate out transient
    for i in np.arange(trans_step):
        w0 = rk4_step( w0, dt, nu, forcing )
    print( "Transient complete." )
    
    #Now integrate out training data
    for i in np.arange(num_outputs):
        print(f"{i}/{num_outputs}")
        w[:,i,:,:] = w0
        for j in np.arange(stride):
            w0 = rk4_step( w0, dt, nu, forcing, geometric=True )

    my_dict = { "w": w, "x": x, "y": y }
    savemat("w_traj.mat", my_dict)