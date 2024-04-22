'''
This file contains classes for group equivariant operations
'''


import torch
import torch.nn as nn

from lib.torch_dns import  navier_stokes_rk4

device = "cuda"

class SymmetryFactory():
    '''
    This class just provides functions for augmenting data via symmetry operations.
    We can use this to check if a network is actually equivariant
    '''

    def __init__(self):
        super().__init__()

    def rot90_kernel(self, tensor):
        '''
        Perform a 90 degree rotation for a doubly periodic domain around the upper left corner [0,0]
        This is the required rotations for kernels in my architecture
        '''
        c1 = 0
        c2 = 0
        needs_reshaping = False
        if len(tensor.shape) > 4:
            #You are of size [b,c1,c2,n,n] likely
            c1 = tensor.shape[1]
            c2 = tensor.shape[2]
            tensor = torch.flatten(tensor,1,2)
            needs_reshaping = True

        # at this point should be of shape [b,c,n,n]
        # a 90 degree rotation is the combination of two things:
        # 1. matrix transpose
        # 2. reverse the order of all columns (except the first) 
        #This form of rotations leaves tensor[:,:,0,0] as a fixed point
        tensor = torch.transpose(tensor, 2, 3)

        #split tensor into first column and the rest
        first_column = tensor[:,:,:,:1]
        rest_columns = tensor[:,:,:,1:]
        #reverse the ordering of rest_columns
        reversed_columns = torch.flip(rest_columns, dims=[3])
        #stack them again
        tensor = torch.cat( (first_column, reversed_columns), dim=3 )

        if needs_reshaping:
            tensor = torch.unflatten(tensor, 1, (c1,c2) )

        return tensor


    def rot90(self, tensor):
        '''
        PURPOSE:
        This applies a 90 degree rotation to a REAL tensor of size [b,c,n,n].
        THIS IS NOT FOR KERNELS
        Because of the way points are stored, we will need to circularly pad before
        '''

        c1 = 0
        c2 = 0
        needs_reshaping = False
        if len(tensor.shape) > 4:
            #You are of size [b,c1,c2,n,n] likely
            c1 = tensor.shape[1]
            c2 = tensor.shape[2]
            tensor = torch.flatten(tensor,1,2)
            needs_reshaping = True

        #define a function for padding to the right and below
        extend = nn.CircularPad2d((0,1,0,1))
        n = tensor.shape[2]

        #pad
        tensor = extend(tensor)
        
        #rotate
        tensor = torch.rot90( tensor, k=1, dims=(2,3) )
        
        #unpad
        tensor = tensor[:,:,:n,:n]

        if needs_reshaping:
            tensor = torch.unflatten(tensor, 1, (c1,c2) )

        return tensor
    
    def transpose(self, tensor):
        c1 = 0
        c2 = 0
        needs_reshaping = False
        if len(tensor.shape) > 4:
            #You are of size [b,c1,c2,n,n] likely
            c1 = tensor.shape[1]
            c2 = tensor.shape[2]
            tensor = torch.flatten(tensor,1,2)
            needs_reshaping = True

        tensor = torch.transpose(tensor, 2, 3)
        
        if needs_reshaping:
            tensor = torch.unflatten(tensor, 1, (c1,c2) )
        
        return tensor



class EquivariantLayer(nn.Module):
    def __init__(self, c1, c2, n1, n2 ):
        '''
        The goal of this network is to map a [b,c1,n1,n1] tensor to a [b,c2,n2,n2]
        b - batch size
        c1- input channels
        c2- output channels
        n1- input grid resolution
        n2- output grid resolution 

        We will also apply a physics-informed activation to the [b,c2,n2,n2] output to obtain
        a new output tensor of size [b,c3,n2,n2], where c3 = c2*(c2-1)/2 + c1
        '''

        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.n1 = n1
        self.n2 = n2

        print(f"Creating convolutional layer with resolution change {n1} -> {n2} and channel {c1} -> {c2}")

        n_min = min( (n1,n2) ) 
        n_max = max( (n1,n2) )

        #The only trainable parameter is the kernel, which will be stored at the minimum resolution
        #We will check if we are upsampling or downsampling and change the order of convolution as needed
        #to preserve memory
        epsilon = 0.1 #amplitude of initialized field
        self.kernel = torch.nn.Parameter(  epsilon*(2*torch.rand(1, c1, c2, n_min, n_min)-1) )
        
        #parameters for the physics-informed activation
        self.c_wsq = torch.nn.Parameter(  epsilon*(2*torch.rand(1, c2, 1, 1) - 1) )
        self.c_usq = torch.nn.Parameter(  epsilon*(2*torch.rand(1, c2, 1, 1) - 1) )
        self.c_p   = torch.nn.Parameter(  epsilon*(2*torch.rand(1, c2, 1, 1) - 1) )
        self.bias  = torch.nn.Parameter(  epsilon*(2*torch.rand(1, c2, 1, 1) - 1) )

        #We will need several masks for handling upsampling/downsampling.
        #we need two because of the way the real fft (rfft) stores output
        k = torch.arange(n_max)
        k[k>n_max//2] = k[k>n_max//2] - n_max
        self.mask  = (torch.abs(k) <= n_min/2)
        self.mask2 = self.mask[0:(n_max//2+1)]

        self.nyquist_pos = (k== n_min/2)
        self.nyquist_pos2= self.nyquist_pos[0:(n_max//2+1)]
        self.nyquist_neg = (k==-n_min/2)
        self.mask[ self.nyquist_neg ] = False #Make sure there is only one nyquist frequency
        
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
        
        #create a third mask for unique antisymmetric combinations for cross products
        idx1   = torch.unsqueeze( torch.arange(c2), 0 )
        idx2   = torch.unsqueeze( torch.arange(c2), 1 )
        self.mask3 = torch.reshape( idx1-idx2 > 0, [-1] )

        #move all constant tensors onto gpu
        #device = "cuda"
        self.kx       = self.kx.to(device)
        self.ky       = self.ky.to(device)
        self.to_u     = self.to_u.to(device)
        self.to_v     = self.to_v.to(device)
        self.inv_k_sq = self.inv_k_sq.to(device)
        self.mask     = self.mask.to(device)
        self.mask2    = self.mask2.to(device)
        self.mask3    = self.mask3.to(device)


    def symmetric_kernel(self):
        '''
        Symmetrize the kernel with respect to D4, the symmetry group of a square
        I'm usin gthe fact that D4 is generated by a reflection about the diagonal and 90 degree rotations
        '''

        #Use this to apply symmetry operations since the storage of our data on a 
        #periodic grid makes applying these symmetries nontrivial
        symm = SymmetryFactory()

        #all rotations
        k1 = self.kernel
        k2 = symm.rot90_kernel( k1 )
        k3 = symm.rot90_kernel( k2 )
        k4 = symm.rot90_kernel( k3 )

        #one reflection + rotations
        k5 = symm.transpose(k1) #reflection about diagonal
        k6 = symm.rot90_kernel( k5 )
        k7 = symm.rot90_kernel( k6 )
        k8 = symm.rot90_kernel( k7 )
        
        #take the group invariant mean
        k = (k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8)/8
        return k



    def downsample(self, f):
        '''
        Reduce the spatial resolution by using a truncated Fourier series
        '''
        #There is care needed in handling the Nyquist frequency for maintaining equivariance. 
        #Rather than express that care, Let's just kill it completely.
        f[:,:,self.nyquist_pos,:]  = 0 * f[:,:,self.nyquist_pos,:]
        f[:,:,:,self.nyquist_pos2] = 0 * f[:,:,:,self.nyquist_pos2]
        
        #Torch wants me to do the logical indexing one axis at a time
        f = f[:,:,self.mask ,         :]
        f = f[:,:,         :,self.mask2]

        #renormalize
        f = f * (self.n2 / self.n1)**2

        return f

    def upsample(self, f):
        b = f.shape[0]
        c = f.shape[1]

        #Just as in the downsampling function, let's just kill the nyquist frequency for safety
        nyq = self.n1//2 #No mask needed
        
        f1 = f[:,:,0:nyq,:]
        f2 = f[:,:,nyq,:]
        f3 = f[:,:,nyq+1:]
        f2 = torch.unsqueeze(f2,2)
        f = torch.cat( (f1,0*f2,f3), dim=2 )
        
        f1 = f[:,:,:,0:nyq]
        f2 = f[:,:,:,nyq]
        f2 = torch.unsqueeze(f2,3)
        f = torch.cat( (f1,0*f2), dim=3 )

        #Torch hates multiple row Boolean indexing, so I have to do this insane work-around
        temp = f
        #device = "cuda"
        f = torch.zeros( (b,c,self.n1,self.n2//2+1), dtype=torch.complex64 ).to(device)
        f[:,:,:,self.mask2] = temp
        
        self.mask

        temp = f
        f = torch.zeros( (b,c,self.n2,self.n2//2+1), dtype=torch.complex64 ).to(device)
        f[:,:,self.mask,:] = temp

        #renormalize
        f = f * (self.n2 / self.n1)**2

        return f

    def local_activation( self, f ):
        #A local, equivariant activation that is lightly "physics-informed"
        '''
        f    = torch.fft.rfft2(f)
        u = self.to_u * f 
        v = self.to_v * f

        #flow gradients
        ux = 1j*self.kx * u
        uy = 1j*self.ky * u
        vx = 1j*self.kx * v
        vy = 1j*self.ky * v

        #pressure
        p = self.inv_k_sq * (ux*ux + vy*vy + 2*uy*vx)
        
        u = torch.fft.irfft2(u)
        v = torch.fft.irfft2(v)
        p = torch.fft.irfft2(p)
        f = torch.fft.irfft2(f)
        '''
        sig = nn.Sigmoid()

        arg = self.c_wsq*f*f  - self.bias #+ self.c_usq*(u*u+v*v) + self.c_p*p - self.bias
        f = f*sig( arg )
        return f

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

    def euler_step(self,f):
        #an Euler step of the Euler equations
        f    = torch.fft.rfft2(f)
        
        u  = self.to_u * f 
        v  = self.to_v * f
        fx = 1j*self.kx * f
        fy = 1j*self.ky * f
        
        u = torch.fft.irfft2(u)
        v = torch.fft.irfft2(v)
        fx= torch.fft.irfft2(fx)
        fy= torch.fft.irfft2(fy)
        f = torch.fft.irfft2(f)

        #Take a tiny step forward in time via advection
        dt = 0.1 #Take a tiny timestep
        return f - (u*fx - v*fy)*dt

    def convolution(self,f):
        # Take f of size [b,c1,n1,n1] and convolve via fft2 + change sampling 
        # output [b,c2,n2,n2] 
        #
        # Both input and output will be in Fourier space using torch.fft.rfft2

        #Don't access the kernel directly. Use the symmetrized variant
        #This kernel is the group average over D4, the symmetry group of a square
        k = self.symmetric_kernel()
        k  = torch.fft.rfft2( k )
        
        #Convolve == multiply in Fourier space
        f    = torch.unsqueeze(f,2)  #[b,c1,1,n,n] to make room for c2 dimension
        conv = torch.sum(f*k, dim=1) #convolved f
        
        return conv

    def forward(self, f ):
        # Perform a convolution and change resolution of input f

        f  = torch.fft.rfft2( f )
        #Do downsampling if needed
        if self.n2 < self.n1:
            f = self.downsample(f)

        #output is in Fourier space
        conv = self.convolution( f )

        #Do upsampling if needed
        if self.n2 > self.n1:
            conv = self.upsample(conv)
            f    = self.upsample(f)

        f    = torch.fft.irfft2(f)
        conv = torch.fft.irfft2(conv)
        return f, conv

    def forward2(self, f ):
        # Perform a convolution and change resolution of input f

        f  = torch.fft.rfft2( f )
        #Do downsampling if needed
        if self.n2 < self.n1:
            f = self.downsample(f)

        #output is in Fourier space
        conv = self.convolution( f )

        #Do upsampling if needed
        if self.n2 > self.n1:
            conv = self.upsample(conv)
            f    = self.upsample(f)

        #Compute the children velocity fields
        w = self.genetic_activation(conv)
        #Apply a nonlinear function to w to keep the values reasonable
        w = torch.tanh(w)

        f = torch.fft.irfft2(f)
        return torch.cat( (f, w), dim=1 )

    def output_dim(self):
        return self.c1 + (self.c2*(self.c2-1))//2

class EquivariantAutoencoder(nn.Module):
    def __init__(self, latent_c, enc_res, dec_res, enc_c, dec_c ):
        super().__init__()
        #device = "cuda"

        ########################
        # Make encoding layers
        ########################
        enc = nn.ModuleList() #convolutional layers
        c = enc_c[0] #keep track of true channels since we are adding skip connections
        for i in range(len(enc_res)-1):        
            layer = EquivariantLayer( c1=c, c2=enc_c[i+1], n1=enc_res[i], n2=enc_res[i+1] ).to(device)
            self.add_module(f"enc_{i}", layer) #keep track of weights!
            c = layer.output_dim() #since skip connection
            enc.append(layer)
        self.enc = enc
        #A linear layer to go to latent_c
        self.elin = nn.Linear(c, latent_c, bias=False)
        print(f"Linear layer {c} -> {latent_c}")
        self.elin = self.custom_init(self.elin)

        ########################
        # Make decoding layers
        ########################
        dec = nn.ModuleList() #convolutional layers
        c = latent_c #keep track of true channels since we are adding skip connections
        for i in range(len(dec_res)-1):        
            layer = EquivariantLayer( c1=c, c2=dec_c[i+1], n1=dec_res[i], n2=dec_res[i+1] ).to(device)
            self.add_module(f"dec_{i}", layer)
            c = layer.output_dim() #since skip connection
            dec.append(layer)
        self.dec = dec
        #A linear layer to go to vorticity
        self.dlin = nn.Linear(c, 1, bias=False)
        print(f"Linear layer {c} -> 1")
        self.dlin = self.custom_init(self.dlin)

    def custom_init(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.fill_(0)
                m.weight[0, 0] = 1
        return m

    def encode(self, input, dropout=0):
        drop = nn.Dropout2d(dropout)
        for i in range( len(self.enc) ):
            #Apply convolutional layer
            #Take the input back as well since the layer will upsample/downsample as needed!
            #input, output = self.enc[i](input)
            #Apply nonlinear activation 
            #output = self.enc[i].local_activation(output)
            #Dropout to prevent overfitting
            #output = drop(output)
            #Add skip connection (append input to output)
            #input = torch.cat( (input, output), dim=1 )
            input = self.enc[i].forward2(input)

        #After all of that, apply a linear layer to channels
        #f is [b,c,n,n]. Need to swap c to the end
        input = torch.permute(input, [0,3,2,1])
        input = self.elin(input)
        input = torch.permute(input, [0,3,2,1])
        
        return input
    
    def decode(self, input, dropout=0):
        '''
        Go from latent space to vorticity
        '''
        drop = nn.Dropout2d(dropout)
        for i in range( len(self.dec) ):
            #Apply convolutional layer
            #Take the input back as well since the layer will upsample/downsample as needed!
            #input, output = self.dec[i](input)
            #Apply nonlinear activation 
            #output = self.dec[i].local_activation(output)
            #output = self.dec[i].burgers_activation(output)
            #if i == (len(self.dec)-1):
                #last layer
                #output = self.dec[i].euler_step(output)

            #Apply dropout to prevent overfitting
            #output = drop(output)
            #Add skip connection (append input to output)
            #input = torch.cat( (input, output), dim=1 )
            input = self.dec[i].forward2(input)


        #After all of that, apply a linear layer to channels
        #f is [b,c,n,n]. Need to swap c to the end
        input = torch.permute(input, [0,3,2,1])
        input = self.dlin(input)
        input = torch.permute(input, [0,3,2,1])
        
        return input
        
