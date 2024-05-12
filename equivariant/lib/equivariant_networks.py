'''
This file contains classes for group equivariant operations
'''


import torch
import torch.nn as nn

from lib.FourierNeuralOperator import FourierNeuralOperator
from scipy.io import savemat

device = "cuda"



class EquivariantLayer(nn.Module):
    def __init__(self, c1, c2, n1, n2 ):
        '''
        The goal of this layer is to act as a wrapper layer.
        '''

        super().__init__()

        #Create two FNOs
        self.FNO1 = FourierNeuralOperator(c1,c2,n1,n2)
        self.FNO2 = FourierNeuralOperator(c1,c2,n1,n2)

        

        self.c1 = c1
        self.c2 = c2
        self.n1 = n1
        self.n2 = n2
        
        #Parameters for physics-informed activation
        dt0 = 0.1
        self.dt = torch.nn.Parameter( dt0*torch.ones((1,c2,1,1)) )
        
        #UNCURLING
        #In the new architecture, we go beyond simple translational equivariance.
        #We want to uncurl vorticity fields to construct incompressible vectors, which are then
        #used to create new antisymmetric tensors via the cross product as a physics-informed activation.
        k       = torch.arange(n2)
        k_small = torch.arange(n2//2+1)
        k[k>n2//2] = k[k>n2//2] - n2

        kx = torch.reshape( k,       [-1,1])        
        ky = torch.reshape( k_small, [1,-1])
        
        to_u =  1j*ky/(kx*kx + ky*ky)
        to_v = -1j*kx/(kx*kx + ky*ky)
        to_u[0,0] = 0
        to_v[0,0] = 0
        
        inv_k_sq = 1/(kx*kx + ky*ky)
        inv_k_sq[0,0] = 0

        #add batch and channel dimensions
        self.kx   = torch.unsqueeze(torch.unsqueeze(kx,0),1)
        self.ky   = torch.unsqueeze(torch.unsqueeze(ky,0),1)
        self.to_u = torch.unsqueeze(torch.unsqueeze(to_u,0),1)
        self.to_v = torch.unsqueeze(torch.unsqueeze(to_v,0),1)
        self.inv_k_sq = torch.unsqueeze(torch.unsqueeze(inv_k_sq,0),1)
        
        #move all constant tensors onto gpu
        #device = "cuda"
        self.kx       = self.kx.to(device)
        self.ky       = self.ky.to(device)
        self.to_u     = self.to_u.to(device)
        self.to_v     = self.to_v.to(device)
        self.inv_k_sq = self.inv_k_sq.to(device)
 
    
    def euler_step(self, f1, f2, df1, df2 ):
        '''
        PURPOSE:
        Let's use a Euler scheme of the scalar advection equations.

        INPUT:

        f1 - [b,c,n,n] tensor
        f2 - same
        df1- same size as f, but a tangent vector.
        df2- same size as f, but a tangent vector.

        f  will be evolved according to nonlinear  Euler
        df will be evolved according to linearized Euler
        '''
        
        #f2 becomes the flow velocity
        u  = self.to_u  * f2 
        v  = self.to_v  * f2

        #Spatial derivatives of f1
        fx = 1j*self.kx * f1
        fy = 1j*self.ky * f1

        # Go back to real space
        u = torch.fft.irfft2(u)
        v = torch.fft.irfft2(v)
        fx= torch.fft.irfft2(fx)
        fy= torch.fft.irfft2(fy)
        f1= torch.fft.irfft2(f1)

        #Force time to be positive
        
        f1 = f1 - (u*fx + v*fy)*self.dt #nonlinear advection
        
        if df1 is None:
            return f1, None

        #f2 becomes the flow velocity
        du  = self.to_u  * df2 
        dv  = self.to_v  * df2

        #Spatial derivatives of f1
        dfx = 1j*self.kx * df1
        dfy = 1j*self.ky * df1

        #Transform to real space
        du = torch.fft.irfft2(du)
        dv = torch.fft.irfft2(dv)
        dfx= torch.fft.irfft2(dfx)
        dfy= torch.fft.irfft2(dfy)
        df1= torch.fft.irfft2(df1)

        #Take a tiny step forward in time via advection
        df1 = df1 - (du*fx + dv*fy + u*dfx + v*dfy)*self.dt #linear advection  
    
        return f1, df1



    def forward(self, f, v=None ):
        '''
        PURPOSE:
        In an equivariant way, map some vorticity fields (and tangent vectors) forward

        INPUT:
        f - tensor of size [b,c1,n1,n1]
        v - a tangent vector the same size as f
        '''
        
        #Assume f and v are passed in real space. Take the fft
        f = torch.fft.rfft2( f )
        v = torch.fft.rfft2( v ) if v is not None else None

        #Perform both convolutions
        #Since this is a linear operation, v just has the same operation applied

        f0 = f.clone()
        v0 = v.clone()

        f, f1 = self.FNO1.forward( f0, fourier_input=True, fourier_output=True )
        v, v1 = self.FNO1.forward( v0, fourier_input=True, fourier_output=True ) if v is not None else None
        _, f2 = self.FNO2.forward( f0, fourier_input=True, fourier_output=True )
        _, v2 = self.FNO2.forward( v0, fourier_input=True, fourier_output=True ) if v is not None else None

        #Apply Euler step of the Euler equations
        f3, v3 = self.euler_step( f1, f2, v1, v2 )
        #These come out in real space

        #Map original inputs back to real space. Resolution might have changed!
        f = torch.fft.irfft2(f)
        v = torch.fft.irfft2(v) if v is not None else None

        #Stack output of convolution + activation with input so the network has skip connections
        f_total = torch.cat( (f, f3), dim=1 )
        v_total = torch.cat( (v, v3), dim=1) if v is not None else None

        return f_total, v_total



    def output_dim(self):
        return self.c1 + self.c2

    def save_dt(self, n):
        my_dict = { "dt" : self.dt.cpu().detach() }
        savemat(f"dt_{n}.mat", my_dict)

    def save_kern(self, n):
        my_dict = { "k" : self.symmetric_kernel().cpu().detach() }
        savemat(f"kernel_{n}.mat", my_dict)






    '''
    DEPRECATED: old functions below. User beware...
    '''

    def genetic_activation(self,f):
        '''
        Use a physics-informed activation
        '''
        #f = torch.fft.rfft2(f)
        u = torch.fft.irfft2( f*self.to_u ) #x component of velocity
        v = torch.fft.irfft2( f*self.to_v ) #y component of velocity

        #both components are of size [b,c2,n,n]
        u = torch.unsqueeze(u, dim=1)
        v = torch.unsqueeze(v, dim=2)
        w = u*v
        w = w - torch.transpose(w, 1, 2)
        w = w.flatten(1,2) #combine these two dimensions
        
        #compute a mask to take the unique
        k = torch.arange(self.c2)
        d = torch.unsqueeze(k,dim=0) - torch.unsqueeze(k,dim=1)
        d = d.flatten()
        d = d>0
        #print(w.shape)
        w = w[:,d,:,:]
        #print(w.shape)
        return w
    
    def burgers_activation(self,f):
        '''
        Filling out the Fourier-spectrum is hard when upsampling and activating. What if we used a cheap
        exact solution to the scalar Burgers equation

        \partial_t \phi + \nabla \phi \cdot \nabla \phi = \nu \nabla^2 \phi
        '''
        f    = torch.fft.rfft2(f)
        u = self.to_u * f 
        v = self.to_v * f
        
        u = torch.fft.irfft2(u)
        v = torch.fft.irfft2(v)
        f = torch.fft.irfft2(f)
        
        #Generate an initial scalar field, no need to use a bias. 
        phi = self.c_wsq*f*f + self.c_usq*(u*u+v*v) - self.bias
        
        #Do Cole-hopf transformation
        nu = 1
        phi = torch.exp( -phi/2/nu )

        #Solve the heat equation forward in time
        t = 1 #time to integrate
        phi = torch.fft.rfft2(phi)
        phi = torch.exp( -t*nu*(self.kx**2 + self.ky**2) )*phi
        phi = torch.fft.irfft2(phi)

        #Invert cole-hopf
        phi = -2*nu*torch.log(phi)

        return f