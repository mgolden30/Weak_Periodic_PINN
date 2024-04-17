'''
The goal of this code is to write a Navier-Stokes integrator in pytorch that 
can be used as a physics-informed activation 

without loss of generality, all vorticity fields will be of size [b,c,n,n],
where b is batch size, c is channels, and n is the grid spacing
'''

import torch as torch

def wavenumbers(n):
    #Generate wavenumber kx and ky for a given grid spacing
    #These will be different lengths since we will use torch.fft.rfft2
    k = torch.arange(n)
    k[k>n/2] = k[k>n/2] - n
    k2= k[:n//2+1]

    kx = torch.reshape(k,  [1,1,n,1])
    ky = torch.reshape(k2, [1,1,1,n//2+1])
    return kx, ky

def dealias( w, n ):
    kx, ky = wavenumbers(n)
    mask = kx*kx + ky*ky >= n*n/9 #2/3rds dealiasing
    mask = torch.squeeze(mask)
    mask = mask.flatten()

    wf = torch.fft.rfft2(w)
    wf = wf.flatten(2,3)
    wf[:,:,mask] = 0
    wf = wf.unflatten(2,(n,n//2+1))
    w = torch.fft.irfft2( wf )
    return w



def time_deriv( w, n, nu, forcing=None ):
    kx, ky = wavenumbers(n)

    to_u = 1j*ky/(kx*kx + ky*ky)
    to_v =-1j*kx/(kx*kx + ky*ky)
    to_u[0,0] = 0 #no mean flow
    to_v[0,0] = 0 #no mean flow
    
    wf = torch.fft.rfft2(w)
    
    u   = torch.fft.irfft2( to_u*wf)
    v   = torch.fft.irfft2( to_v*wf)
    wx  = torch.fft.irfft2(1j*kx*wf)
    wy  = torch.fft.irfft2(1j*ky*wf)
    lap = torch.fft.irfft2( -(kx*kx + ky*ky)*wf )

    rhs = -u*wx-v*wy + nu*lap
    if forcing is not None:
        rhs = rhs + forcing
    
    rhs = dealias(rhs, n)
    return rhs

def navier_stokes_rk4(w, dt, steps, nu, forcing=None):
    n = w.shape[3]
    for _ in torch.arange(steps):
        k1 = time_deriv(w,           n, nu, forcing)
        k2 = time_deriv(w + dt*k1/2, n, nu, forcing)
        k3 = time_deriv(w + dt*k2/2, n, nu, forcing)
        k4 = time_deriv(w + dt*k3,   n, nu, forcing)
        w = w + (k1 + 2*k2 + 2*k3 + k4)*dt/6
    return w