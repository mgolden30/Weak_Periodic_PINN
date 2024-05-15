'''
Write an activation function with with no trainable parameters for advection nonlinearity
'''

import torch as torch

from lib.FourierNeuralOperator import FourierNeuralOperator

def advection_activation( f, v, fourier_input=True, fourier_output=False ):
    '''
    f - [b,2*c,n,n] vorticity fields
    v - [b,2*c,n,n] tangent vectors (or None)

    Take a single explicit advective (and linearized advective) step
    '''

    if not fourier_input:
        f = torch.fft.fft2(f)
        v = torch.fft.fft2(v) if v is not None else None
    
    #split the channel dimension into the vorticity that is advected (f1) and the source of the flow (f2)
    c = f.shape[1]
    
    f1 = f[:, :c, :, :]
    f2 = f[:, c:, :, :]
    v1 = v[:, :c, :, :] if v is not None else None
    v2 = v[:, c:, :, :] if v is not None else None

    #Make two FourierNeuralOperators for the pure purpose of changing sptial resolutions
    n1 = f1.shape[-2]
    n2 = 3*(n1//2) #We need to pad with zeros in order to preserve group equivariance under continuous translations
    up   = FourierNeuralOperator(c1=None,c2=None,n2=n2, make_kernel=False)
    down = FourierNeuralOperator(c1=None,c2=None,n2=n1, make_kernel=False)

    #upsample both fields
    f1 = up.upsample( f1, fourier_input=True, fourier_output=True )
    f2 = up.upsample( f2, fourier_input=True, fourier_output=True )
    v1 = up.upsample( v1, fourier_input=True, fourier_output=True ) if v is not None else None
    v2 = up.upsample( v2, fourier_input=True, fourier_output=True ) if v is not None else None

    #Construct derivative and velocity matrices
    k = torch.arange(n2)
    k[k>n2//2] = k[k>n2//2] - n2
    kx = torch.unsqueeze( k, 1 )
    ky = torch.unsqueeze( k, 0 )
    k_sq = kx*kx + ky*ky
    k_sq[0,0] = 1.0 #This removes division by zero in the uncurl
    to_u = 1j*ky/k_sq
    to_v =-1j*kx/k_sq


    u = to_u * f2
    v = to_v * f2
    f1x = 1j*kx*f1 
    f1y = 1j*ky*f1 

    if v is not None:
        du = to_u * v2
        dv = to_v * v2
        df1x = 1j*kx*v1 
        df1y = 1j*ky*v1 

    print(u.shape)
    u   = torch.real( torch.fft.ifft2( u))
    v   = torch.real( torch.fft.ifft2( v))
    f1x = torch.real( torch.fft.ifft2( f1x))
    f1y = torch.real( torch.fft.ifft2( f1y))

    #Take advective step. My rational is that u will also learn a timestep
    f1 = f1 - ( u*f1x +  v*f1y )
    v1 = v1 - (du*f1x + dv*f1y + u*df1x + v*df1y ) if v is not None else None

    #Downsample to original resolution
    f1 = down.downsample(f1, fourier_input=False, fourier_output=fourier_output)
    v1 = down.downsample(v1, fourier_input=False, fourier_output=fourier_output) if v is not None else None

    return f1, v1