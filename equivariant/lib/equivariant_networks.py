import torch
import torch.nn as nn


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
    def __init__(self, c1, c2, n1, n2, apply_activation):
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
        self.apply_activation = apply_activation

        n_min = min( (n1,n2) ) 
        n_max = max( (n1,n2) )

        #The only trainable parameter is the kernel, which will be stored at the minimum resolution
        #We will check if we are upsampling or downsampling and change the order of convolution as needed
        #to preserve memory
        epsilon = 0.1
        self.kernel = torch.nn.Parameter(  epsilon*(2*torch.rand(1, c1, c2, n_min, n_min)-1) )
        
        #We will need several masks for handling upsampling/downsampling.
        #we need two because of the way the real fft (rfft) stores output
        k = torch.arange(n_max)
        k[k>n_max//2] = k[k>n_max//2] - n_max
        self.mask  = torch.abs(k) <= n_min/2
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
        
        #add batch and channel dimensions
        self.to_u = torch.unsqueeze(torch.unsqueeze(to_u,0),1)
        self.to_v = torch.unsqueeze(torch.unsqueeze(to_v,0),1)

        #create a third mask for unique antisymmetric combinations for cross products
        idx1   = torch.unsqueeze( torch.arange(c2), 0 )
        idx2   = torch.unsqueeze( torch.arange(c2), 1 )
        self.mask3 = torch.reshape( idx1-idx2 > 0, [-1] )



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
        #k = (k1+k2+k3+k4)/4
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
        f = f * self.n2 / self.n1

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
        f = torch.zeros( (b,c,self.n1,self.n2//2+1), dtype=torch.complex64 )
        f[:,:,:,self.mask2] = temp
        
        self.mask

        temp = f
        f = torch.zeros( (b,c,self.n2,self.n2//2+1), dtype=torch.complex64 )
        f[:,:,self.mask,:] = temp

        #renormalize
        f = f * self.n2 / self.n1

        return f

    def uncurl_and_cross(self, conv):
        '''
        PURPOSE:
        Physics-informed activation function. Many channels of vorticity fields are uncurled to obtain
        the corresponding flows and these flows are cross-producted to find new antisymmetric tensors
        '''

        #construct the flow velocities by uncurling
        u = self.to_u * conv
        v = self.to_v * conv
        u = torch.fft.irfft2(u)
        v = torch.fft.irfft2(v)

        #compute the cross product of u and v
        u = torch.unsqueeze(u,2)
        v = torch.unsqueeze(v,1)
        #print(f"magnitude of u is {torch.mean(torch.abs(u), dim=[0,1,2,3,4])}")
        #print(f"magnitude of v is {torch.mean(torch.abs(v), dim=[0,1,2,3,4])}")
        cross = u*v
        cross = cross - cross.transpose(1,2)
        #print(f"magnitude of cross is {torch.mean(torch.abs(cross), dim=[0,1,2,3,4])}")

        #combine second two indices into many channels
        cross = torch.flatten(cross, 1, 2)

        #restrict to unique combinations since the cross product is antisymmetric
        f = cross[:,self.mask3,:,:]
        return f

    def forward(self, f ):
        k = self.symmetric_kernel()
        
        #Change f and k to Fourier space
        f  = torch.fft.rfft2( f )
        k  = torch.fft.rfft2( k )

        #Do downsampling if needed
        if self.n2 < self.n1:
            f = self.downsample(f)

        #Convolve == multiply in Fourier space
        f    = torch.unsqueeze(f,2)  #[b,c1,1,n,n] to make room for c2 dimension
        #print(f.shape)
        #print(k.shape)
        conv = torch.sum(f*k, dim=1) #convolved f
        
        #get rid of the dimension we added, but don't kill channel dimension even if 
        #channel = 1! This flatten trick is a good workaround
        f = torch.flatten(f, 2, 3)

        #Do upsampling if needed
        if self.n2 > self.n1:
            f    = self.upsample(f)
            conv = self.upsample(conv) 

        if( self.apply_activation == False ):
            #Don't apply the activation, just return the convolution output
            #No activation, no stacking with input
            return torch.fft.irfft2(conv)

        #physics-informed activation
        cross = self.uncurl_and_cross(conv)

        #stack these new features with the input
        f = torch.fft.irfft2(f)
        f = torch.cat( (f,cross), dim=1 )

        return f

class EquivariantAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        #Parameters
        c  = 4 #number of kernel output channels in general layers 
        ce = 2 #number of kernel output channels to the latent space
        output_dim = lambda c1,c2 : c1 + (c2*(c2-1))//2
        
        #encoding layers
        self.elayer1 = EquivariantLayer(c1=2,  c2=c,  n1=64, n2=32, apply_activation=True) # take in vorticity and force, so c1=2
        ci = output_dim(2, c)
        self.elayer2 = EquivariantLayer(c1=ci, c2=c,  n1=32, n2=16, apply_activation=True)
        ci = output_dim(ci,c)
        self.elayer3 = EquivariantLayer(c1=ci, c2=c,  n1=16, n2=8,   apply_activation=True)
        ci = output_dim(ci,c)
        self.elin = nn.Linear(ci, ce, bias=False)

        #decoding layers
        self.dlayer1 = EquivariantLayer(c1=ce, c2=c, n1= 8, n2=16, apply_activation=True)
        ci = output_dim(ce,c)
        self.dlayer2 = EquivariantLayer(c1=ci, c2=c, n1=16, n2=32, apply_activation=True) 
        ci = output_dim(ci,c)
        self.dlayer3 = EquivariantLayer(c1=ci, c2=c, n1=32, n2=64, apply_activation=True) 
        ci = output_dim(ci,c)
        self.dlayer4 = EquivariantLayer(c1=ci, c2=c, n1=64, n2=64, apply_activation=True)
        ci = output_dim(ci,c)
        self.dlin = nn.Linear( ci, 1, bias=False)
        
    def encode(self, f):
        f = self.elayer1(f)
        f = self.elayer2(f)
        f = self.elayer3(f)

        #f is [b,c,n,n]. Need to swap c to the end
        f = torch.permute(f, [0,3,2,1])
        f = self.elin(f)
        f = torch.permute(f, [0,3,2,1])
        
        return f
    
    def decode(self, f):
        
        f = self.dlayer1(f)
        f = self.dlayer2(f)
        f = self.dlayer3(f)
        f = self.dlayer4(f)

        #f is [b,c,n,n]. Need to swap c to the end
        f = torch.permute(f, [0,3,2,1])
        f = self.dlin(f)
        f = torch.permute(f, [0,3,2,1])
        
        return f
        
