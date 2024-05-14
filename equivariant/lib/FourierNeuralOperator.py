'''
Contain a single class: the  Fourier Neural Operator for symmetry covariant convolutions + downsampling + upsampling
'''


import torch
import torch.nn as nn

from lib.SymmetryFactory import SymmetryFactory

from scipy.io import savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FourierNeuralOperator(nn.Module):
    def __init__(self, c1, c2, n1, n2 ):
        '''
        The goal of this network is to map a [b,c1,n1,n1] tensor to a [b,c2,n2,n2]
        b - batch size
        c1- input channels
        c2- output channels
        n1- input grid resolution
        n2- output grid resolution 
        '''

        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.n1 = n1
        self.n2 = n2

        print(f"Creating Fourier Neural Operator with resolution change {n1} -> {n2} and channel {c1} -> {c2}")

        n_min = min( (n1,n2) ) 
        n_max = max( (n1,n2) )

        #The only trainable parameter is the kernel, which will be stored at the minimum resolution
        #We will check if we are upsampling or downsampling and change the order of convolution as needed
        #to preserve memory
        
        #amplitude of initialized kernels
        epsilon = 0.01 #Just make it small initially to break symmetry
        self.kernel = torch.nn.Parameter(  epsilon*(2*torch.rand(1, c1, c2, n_min, n_min)-1) )
        
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
        
        self.mask3 = (torch.abs(k) >= n_min/4)
        self.mask4 = self.mask3[0:(n_max//2+1)]

        #move all constant tensors onto gpu
        #device = "cuda"
        self.mask     = self.mask.to(device)
        self.mask2    = self.mask2.to(device)
        self.mask3    = self.mask3.to(device)
        self.mask4    = self.mask4.to(device)


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
        Reduce the spatial resolution by truncating a Fourier series
        '''

        #There is care needed in handling the Nyquist frequency for maintaining equivariance. 
        #Rather than express that care, Let's just kill it completely.
        f[:,:,self.nyquist_pos,:]  = 0 * f[:,:,self.nyquist_pos,:]
        f[:,:,:,self.nyquist_pos2] = 0 * f[:,:,:,self.nyquist_pos2]
        
        #Throw out high modes for advection to maintain 
        f[:,:,self.mask3,:] = 0.0
        f[:,:,:,self.mask4] = 0.0

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



    def forward(self, f, fourier_input=False, fourier_output=False ):
        '''
        PURPOSE:
        Fourier convolution

        INPUT:
        f - tensor of size [b,c1,n1,n1]

        fourier_input  - Did you apply torch.fft.rfft2 already? Are you in Fourier space?
        fourier_output - Did you want to apply torch.fft.irfft2 before returning?

        OUTPUT:
        The 

        '''
        
        #Transform to Fourier space if input isn't already
        if not fourier_input:
            f = torch.fft.rfft2( f )

        #Do downsampling if needed
        if self.n2 < self.n1:
            f = self.downsample(f)

        #output of convolution is in Fourier space
        f0 = f #copy the original input (in fourier space)
        f  = self.convolution( f )

        #Do upsampling if needed
        if self.n2 > self.n1:
            f0 = self.upsample(f0)
            f  = self.upsample(f)
        
        #Inverse fft
        if not fourier_output:
            f = torch.fft.irfft2(f)
            f0= torch.fft.irfft2(f0)
            
        return f0, f


    def output_dim(self):
        return self.c2

    def save_dt(self, n):
        my_dict = { "dt" : self.dt.cpu().detach() }
        savemat(f"dt_{n}.mat", my_dict)

    def save_kern(self, n):
        my_dict = { "k" : self.symmetric_kernel().cpu().detach() }
        savemat(f"kernel_{n}.mat", my_dict)