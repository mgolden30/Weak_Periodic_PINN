import torch
import torch.nn as nn

#from lib.SymmetryFactory import SymmetryFactory
#from lib.equivariant_networks import EquivariantLayer
from lib.FourierNeuralOperator import FourierNeuralOperator
from lib.advection_activation import advection_activation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            layer = FourierNeuralOperator( c1=c, c2=2*enc_c[i+1], n2=enc_res[i+1] ).to(device)            
            
            #Since we have skip connections all layers can see all previous layers
            c = c + layer.output_dim()//2

            self.add_module(f"enc_{i}", layer) #keep track of weights!
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
        for i in range(len(enc_res)-1):
            layer = FourierNeuralOperator( c1=c, c2=2*dec_c[i+1], n2=dec_res[i+1] ).to(device)            
            
            #Since we have skip connections all layers can see all previous layers
            c = c + layer.output_dim()//2

            self.add_module(f"dec_{i}", layer) #keep track of weights!
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

    def encode(self, f, v=None, dropout=0): 
        #Loop over each encoding layer
        for i in range( len(self.enc) ):
            
            #Take the fourier transform
            f = torch.fft.fft2(f)
            v = torch.fft.fft2(v) if v is not None else None

            #Apply the convolution layer
            fc = self.enc[i].forward(f, fourier_input=True, fourier_output=True)
            vc = self.enc[i].forward(v, fourier_input=True, fourier_output=True) if v is not None else None

            #Compute the activation function (and linearized activation function)
            fc, vc = advection_activation( fc, vc, fourier_input=True, fourier_output=False )

            #Invert the  fourier transform
            f = torch.fft.ifft2(f)
            v = torch.fft.ifft2(v) if v is not None else None

            #Stack input and output
            f = torch.cat( (f,fc), dim=1 )
            v = torch.cat( (v,vc), dim=1 ) if v is not None else None


        #Apply dropout before the final projection
        drop = nn.Dropout2d(dropout)
        f = drop(f)

        #Project to final encoding. Since this is linear, the same function 
        #is applied to both
        f = self.project_channels( self.elin, f )
        v = self.project_channels( self.elin, v ) if v is not None else None
 
        #Return the nonlinear map and the action of the Jacobian on v
        return input, v
    


    def decode(self, f, v=None, dropout=0):
        '''
        Go from latent space to vorticity
        '''
        drop = nn.Dropout2d(dropout)
        for i in range( len(self.dec) ):
            #Take the fourier transform
            f = torch.fft.fft2(f)
            v = torch.fft.fft2(v) if v is not None else None

            #Apply the convolution layer
            fc = self.dec[i].forward(f, fourier_input=True, fourier_output=True)
            vc = self.dec[i].forward(v, fourier_input=True, fourier_output=True) if v is not None else None

            #Compute the activation function (and linearized activation function)
            fc, vc = advection_activation( fc, vc, fourier_input=True, fourier_output=False )

            #Invert the  fourier transform
            f = torch.fft.ifft2(f)
            v = torch.fft.ifft2(v) if v is not None else None

            #Stack input and output
            f = torch.cat( (f,fc), dim=1 )
            v = torch.cat( (v,vc), dim=1 ) if v is not None else None
        
        f = self.project_channels( self.dlin, f )
        v = self.project_channels( self.dlin, v ) if v is not None else None
        
        #Return the nonlinear map and the action of the Jacobian on v
        return f, v
        

    def project_channels( self, lin, data ):
        #The last layer of the encoder/decoder is a linear projection over channels
        #I thought I would simplify my life and write a common function to handle this
    
        #data has dimensions [b,c,n,n]. Need to swap c to the end to apply linear layer
        data = torch.permute(data, [0,3,2,1])
        data = lin(data)
        data = torch.permute(data, [0,3,2,1])
        return data

    def save_dt(self):
         for i in range( len(self.dec) ):
            self.dec[i].save_dt(i)
            self.dec[i].save_kern(i)