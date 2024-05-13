import torch
import torch.nn as nn

#from lib.SymmetryFactory import SymmetryFactory
from lib.equivariant_networks import EquivariantLayer

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

    def encode(self, input, v=None, dropout=0): 
        #Loop over each encoding layer
        for i in range( len(self.enc) ):
            input, v = self.enc[i].forward(input, v)

        #Apply dropout before the final projection
        drop = nn.Dropout2d(dropout)
        input = drop(input)

        #Project to final encoding. Since this is linear, the same function 
        #is applied to both
        input = self.project_channels( self.elin, input )
        v     = self.project_channels( self.elin, v ) if v is not None else None
 
        #Return the nonlinear map and the action of the Jacobian on v
        return input, v
    


    def decode(self, input, v=None, dropout=0):
        '''
        Go from latent space to vorticity
        '''
        drop = nn.Dropout2d(dropout)
        for i in range( len(self.dec) ):
            input, v = self.dec[i].forward(input, v)
        
        input = self.project_channels( self.dlin, input )
        v     = self.project_channels( self.dlin, v ) if v is not None else None
        
        #Return the nonlinear map and the action of the Jacobian on v
        return input, v
        

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