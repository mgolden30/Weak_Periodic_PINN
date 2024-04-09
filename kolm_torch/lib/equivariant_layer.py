import torch
import torch.nn as nn

class EquivariantLayer(nn.Module):
    def __init__(self, c1, c2, n1, n2):
        '''
        The goal of this network is to map a [b,n1,n1,c1] tensor to a [b,n2,n2,c2]
        b - batch
        c1- input channels
        c2- output channels
        n1- input grid resolution
        n2- output grid resolution 
        '''

        super().__init__()
        
        #The kernel for convolutions only needs to be this size
        n_min = min( (n1,n2) ) 
        n_max = max( (n1,n2) )

        #Only two trainable paramaters: the trainable
        self.kernel = torch.nn.Parameter(  torch.randn(1, n_min, n_min, c1, c2))
        self.bias   = torch.nn.Parameter(  torch.randn(1,     1,     1, c2))
        #self.linear = nn.Linear( c1, c2 ) #The linear operator acts on the channels

        self.n1 = n1
        self.n2 = n2
        
        #construct two masks for changing grid resolution
        k = torch.arange(n_max)
        k[k>n_max//2] = k[k>n_max//2] - n_max
        self.mask  = torch.abs(k) <= n_min/2
        self.mask2 = self.mask[0:(n_max//2+1)]
        self.mask[k==-n_min/2] = False #Make sure there is only one nyquist frequency

    def forward(self, f ):
        #Apply 90 degree rotational symmetry to the kernels
        rot_1 = lambda k: torch.flip(torch.permute_copy(k, [0,2,1,3,4]),dims=[2])
        rot_2 = lambda k: torch.flip(torch.flip(k, dims=[1]),dims=[2])
        rot_3 = lambda k: torch.flip(torch.permute_copy(k, [0,2,1,3,4]),dims=[1])
        k = self.kernel
        k = (k + rot_1(k) + rot_2(k) + rot_3(k))/4

        #Change f and k to Fourier space
        f = torch.fft.rfft2( f, dim=[1,2]) #default
        k = torch.fft.rfft2( k, dim=[1,2]) #default

        #Do downsampling if needed
        if self.n2 < self.n1:
            #Torch wants me to do the logical indexing one axis at a time
            f = f[:,        :,self.mask2,:]
            f = f[:,self.mask,         :,:]

        #Convolve == multiply in Fourier space
        f = torch.unsqueeze(f,4)
        f = f*k
        f = torch.sum(f, dim=3) - self.bias

        #Do upsampling if needed
        if self.n2 > self.n1:
            b = f.shape[0]
            c = f.shape[3]

            #Torch hates multiple row Boolean indexing, so I have to do this insane work-around

            temp = f
            f = torch.zeros( (b,self.n1,self.n2//2+1,c), dtype=torch.complex64 )
            f[:,:,self.mask2,:] = temp

            temp = f
            f = torch.zeros( (b,self.n2,self.n2//2+1,c), dtype=torch.complex64 )
            f[:,self.mask,:,:] = temp

        #transform back to real space
        f = torch.fft.irfft2(f,dim=[1,2])
        
        return f


class EquivariantDenseBlock(nn.Module):
    '''
    Inspired by "Densely Connected Convolutional Networks" by Huang et al.
    '''
    def __init__(self, n1, n2, c1, c2, num_layers, activation ):
        '''
        The goal of this block is to apply exactly translational equivariant convolutions, 
        but in the style of a "DenseBlock" in the sense of Huang.

        INPUT:
        nl - number of layers. 
        activation - activation function
        '''
        super().__init__()

        self.activation = activation

        layers = []
        num_features = c1 #this will change
        for i in range(num_layers-1):
            layers.append( EquivariantLayer(num_features, c2, n1, n1) )
            num_features = num_features + c2 #since we concat input with output

        #At the last layer, change resolution
        layers.append( EquivariantLayer(num_features, c2, n1, n2) )
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            output = self.activation(layer(x))
            x = torch.cat( (x,output), dim=3 )
        return x

class EquivariantAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        c = 32
        #encoding layers
    
        gelu = nn.GELU()

        self.elayer1 = EquivariantDenseBlock(n1=64, n2=64, c1=2, c2=8, num_layers=4, activation=gelu) #EquivariantLayer(c1=2,c2=c,n1=64,n2=32) # take in vorticity and force, so c1=2
        self.elayer2 = EquivariantLayer(c1=1,c2=c,n1=64,n2=16)
        self.elayer3 = EquivariantLayer(c1=1,c2=1,n1=16,n2=8)
        
        #decoding layers
        self.dlayer1 = EquivariantLayer(c1=1,c2=c,n1= 8,n2=16)
        self.dlayer2 = EquivariantLayer(c1=1,c2=c,n1=16,n2=32) 
        self.dlayer3 = EquivariantLayer(c1=1,c2=1,n1=32,n2=64) 
        
    def encode(self, f):
        '''
        f[:,:,:,0] - vorticity
        f[:,:,:,1] - forcing
        '''
        #activation = lambda x: torch.cos(x)
        activation = torch.nn.GELU()
        
        f = activation(self.elayer1(f))
        f = torch.mean(f,dim=3,keepdim=True)
        f = activation(self.elayer2(f))
        f = torch.mean(f,dim=3,keepdim=True)
        f = activation(self.elayer3(f))
 
        return f
    
    def decode(self, f):
        activation = torch.nn.GELU()

        f = activation(self.dlayer1(f))
        f = torch.mean(f,dim=3,keepdim=True)
        #f = torch.max(f,dim=3,keepdim=True)[0]
        f = activation(self.dlayer2(f))
        #f = torch.max(f,dim=3,keepdim=True)[0]
        f = torch.mean(f,dim=3,keepdim=True)
        f =            self.dlayer3(f)
        #no activation to end decoding

        return f
        

if __name__ == "__main__":
    #Do some quick testing
    n1 = 64
    c1 = 2  #w and forcing
    b  = 24 #batch size

    n2 = 32 #try upsampling and downsampling
    c2 = 7


    w = torch.randn(b, n1, n1, c1)

    eq = EquivariantLayer(c1, c2, n1, n2)
    w2 = eq.forward(w)

    enc = EquivariantAutoencoder()

    l = enc.encode(w)
    print( l.shape )

    w3 = enc.decode(l)
    print( w3.shape )