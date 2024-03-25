import numpy as np
import os
import torch

import torch.optim as optim
from torch import nn


from lib.hail_mary import PotentialNetwork, PotentialHydroNetwork, PotentialPINN
from lib.utils import generate_uniform_grid, save_potential_network_output



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Set PyTorch seed for reproducibility
seed_value = 420
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

epochs = 1024

nu = 1.0/40   #fluid viscosity
L  = 16       #number of nodes in hidden layers
n  = 32       #number of points per side to train on (n^3 total)
lr = 0.0001    #learning rate


#Pick a random set of points
#xs = generate_uniform_grid(n)
xs = 2 * torch.pi * torch.rand( (n*n*n,3), requires_grad=True ) #need to diff with respect to these for flow
xs = xs.to(device)

#guess of a period
T0 = 2*torch.pi

#initialize models
pot   = PotentialNetwork( L ).to(device)
hydro = PotentialHydroNetwork( pot, nu, T0 ).to(device)
pinn  = PotentialPINN( hydro ).to(device)

#Test the network evaluation
potentials = pot.forward(xs)
print(potentials.shape)

hydro_out  = hydro.forward(xs)
print(hydro_out.shape)

err  = pinn.forward(xs)
print(err.shape)

# Define the Mean Squared Error (MSE) loss
criterion = nn.L1Loss()

# Use an optimizer (e.g., Adam) to update the model parameters
optimizer = optim.Adam( hydro.parameters(), lr )

loss_history = torch.zeros( (epochs) )

for epoch in range(epochs):
    #xs = 2 * torch.pi * torch.rand( (n*n*n,3), requires_grad=True ) #need to diff with respect to these for flow
    #xs = xs.to(device)

    # Forward pass
    err = pinn.forward(xs)

    # Compute the MSE loss
    loss = criterion(err, torch.zeros_like(err))  # assuming you want to minimize pinn.forward(xs) to zero

    loss_history[epoch] = loss.detach()

    # Backward pass and optimization step
    optimizer.zero_grad()  # clear previous gradients
    loss.backward(retain_graph=True)   # compute gradients
    optimizer.step()  # update model parameters

    loss_history[epoch] = loss.detach()

    # Print the loss every few epochs
    if epoch % 100 == 0:
        save_potential_network_output( hydro, "torch_output_hm_%d.mat" % (epoch), loss_history )
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

torch.save( hydro, 'trained_model_hm.pth' )