import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, channels, latent):
        super().__init__()
        self.latent   = latent
        self.channels = channels

        # Declare the convolutional layers
        self.conv = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=8, stride=4, padding=(4,4))
        self.fc   = nn.Linear( 17*17*self.channels, latent)  # Define the fully connected layer

    def forward(self, w):
        #Input w is of size [b,n,n], where b is the number of samples (batch size) and n is the grid resolution 
        # Reshape input to [batch_size, channels, height, width], with channels = 1
        w = w.unsqueeze(1)
        # Apply convolutional layers with ReLU activation
        w = torch.sigmoid( self.conv(w) )
        # Flatten the output for the fully connected layer
        w = w.view(-1, 17*17*self.channels )
        # Apply the fully connected layer to get the latent dimension
        latent = torch.sigmoid(self.fc(w))

        return latent


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim + 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, l, xs):
        '''
        l - a [bl, latent] vector in the latent space
        xs- a [bx, 2] vector of (x,y) coordinates to decode to w(xs)

        output should then be of size [bl, bx]
        '''

        # Take the periodic part of coordinates
        xs = torch.cat((torch.cos(xs), torch.sin(xs)), dim=1)  # Concatenate cosine and sine
        xs = xs.unsqueeze(1)  # Shape: [bx, 1, 4]
        l = l.unsqueeze(0)    # Shape: [1, bl, latent]

        # Stack the trigonometric functions with the latent embedding
        bx = xs.shape[0]
        bl = l.shape[1]

        xs = xs.repeat(1,bl,1)
        l  = l.repeat(bx,1,1)
        

        #print(xs.shape)
        #print(l.shape) 
        x = torch.cat((xs, l), dim=2)  # Shape: [bx, bl, latent+4]

        # Apply linear layers with torch.cos activations
        x = torch.cos(self.fc1(x))
        x = torch.cos(self.fc2(x))
        x = torch.cos(self.fc3(x))
        x = self.fc4(x)  

        # Transpose x to have dimensions [bl, bx]
        x = x.transpose(0, 1)

        return x
    
    def forward2(self, l, xs):
        '''
        l - a [bl, latent] vector in the latent space
        xs- a [bx, 2] vector of (x,y) coordinates to decode to w(xs)

        This variant assumes that bl = bx
        '''

        # Take the periodic part of coordinates
        xs = torch.cat((torch.cos(xs), torch.sin(xs)), dim=1)  # Concatenate cosine and sine
        #xs = xs.unsqueeze(1)  # Shape: [bx, 1, 4]
        #l = l.unsqueeze(0)    # Shape: [1, bl, latent]
        
        #print(xs.shape)
        #print(l.shape) 
        x = torch.cat((xs, l), dim=1)  # Shape: [bx, latent+4]

        # Apply linear layers with torch.cos activations
        x = torch.cos(self.fc1(x))
        x = torch.cos(self.fc2(x))
        x = torch.cos(self.fc3(x))
        x = self.fc4(x)  

        # Transpose x to have dimensions [bl, bx]
        #x = x.transpose(0, 1)

        return x

class StreamfunctionNetworkDecoder(nn.Module):
    '''
    The purpose of this class is to use a pretrained decoder to constrain the spatial profile of learned streamfunction.
    '''
    
    def __init__(self, decoder, latent, mask, mean):
        super().__init__()

        #Clone and freeze the decoder so the weights will not change
        for param in decoder.parameters():
            param.requires_grad = False
        self.decoder = decoder

        self.latent = latent #latent dimension
        self.mask = mask
        self.mean = mean

        self.fc1 = nn.Linear(  2,  8 )
        #self.fc2 = nn.Linear(  8, 16 )
        self.fc3 = nn.Linear( 8, latent )

    def forward( self, xs ):
        """
        INPUT:
        xs - size [N, 3] in the ordering (x,y,t)
        """
        t = xs[:,2].unsqueeze(1)
        x = xs[:,0:2]

        #Take sin and cos of these inputs
        t = torch.cat( (torch.sin(t), torch.cos(t)), dim=1 )

        #apply a feed forward network to get a latent embedding
        t = torch.cos(self.fc1(t))
        #l = torch.sigmoid(self.fc2(t))
        l = torch.sigmoid(self.fc3(t)) #important to apply sigmoid so the output is in the bounded interval [0,1]

        #Mask out unphysical degrees of freedom (ones that do not vary in turbulence)
        #print(self.mask.shape)
        #print(self.mean.shape)
        #print(l.shape)
        #l[:, self.mask] = self.mean
        
        # Mask out unphysical degrees of freedom (ones that do not vary in turbulence)  
        #'''
        mask_broadcasted = self.mask.expand(l.size(0), -1)  # Broadcast mask to match the shape of l'
        mean = torch.zeros(64)
        mean[self.mask] = self.mean
        l = torch.where(mask_broadcasted, mean, l)
        #'''
        # l is [N, latent] and x is [N,4]. Since lx and lb are the same, use forward2 to avoid unneeded computation
        psi = self.decoder.forward2( l, x )

        return psi
        
