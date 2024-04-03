'''
Use a pretrained autoencoder in the streamfunction network to constrain the spatial profile
'''

import numpy as np
import torch

import torch.optim as optim
from torch import nn
from lib.model import  HydroNetwork, WeakPINN
from lib.utils import generate_uniform_grid, save_network_output, generate_samples
from lib.autoencoder import Decoder, StreamfunctionNetworkDecoder

from scipy.io import savemat, loadmat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Set PyTorch seed for reproducibility
seed_value = 420
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

epochs = 4*1024

nu = 1.0/40 #fluid viscosity
p  = 6      #points in each direction for Gaussian quadrature
n  = 4*4*4    #number of subdomain to train on


#load the mask data and decoder we trained on turbulence
latent_dim = 64
dec = Decoder(latent_dim=latent_dim)
dec.load_state_dict(torch.load("decoder.pth"))

mask_data = loadmat("latent_mask.mat")
mask = mask_data["static_mask"]
mean = mask_data["static_mean"]

mask = torch.tensor(mask, dtype=torch.bool).squeeze()
mean = torch.tensor(mean)

#construct nested networks
stream_model = StreamfunctionNetworkDecoder( dec, latent_dim, mask, mean )
hydro_model  = HydroNetwork( stream_model )
pinn         = WeakPINN( hydro_model, nu, p )


# Define the Mean Squared Error (MSE) loss
criterion = nn.MSELoss()

# Use an optimizer (e.g., Adam) to update the model parameters
optimizer = optim.Adam(hydro_model.parameters(), lr=0.01)

loss_history = torch.zeros( (epochs) )

for epoch in range(epochs):
    print(epoch)
    xs = generate_samples(n)

    # Forward pass
    err = pinn.forward(xs)

    # Compute the MSE loss
    loss = criterion(err, torch.zeros_like(err))  # assuming you want to minimize pinn.forward(xs) to zero

    loss_history[epoch] = loss.detach()

    # Backward pass and optimization step
    optimizer.zero_grad()  # clear previous gradients
    loss.backward(retain_graph=True)   # compute gradients
    optimizer.step()  # update model parameters

    # Print the loss every few epochs
    if epoch % 100 == 0:
        mat_name = "torch_output_%d.mat" % (epoch)
        mod_name = "torch_output_%d.pth" % (epoch)
        save_network_output( hydro_model, mat_name, mod_name, loss_history, xs, xs )
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

torch.save( hydro_model, 'trained_hydro_model.pth' )