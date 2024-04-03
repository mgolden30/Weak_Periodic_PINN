import numpy as np
import torch

import torch.optim as optim
from torch import nn

from lib.dual_potentials import DualPotential
from lib.model import StreamfunctionNetwork, HydroNetwork, WeakPINN
from lib.utils import save_dual_network_output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# Set PyTorch seed for reproducibility
seed_value = 420
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

num_samples = 1024

epochs = 128
learning_rate = 1e-2

num_features = 256
network = DualPotential( input_dim=3, output_dim=4, num_layers=2, num_features=num_features, T_init=20.0, nu=1.0/40 )

#network = torch.load('trained_potential_model.pth')

# Define the Mean Squared Error (MSE) loss
criterion = nn.L1Loss()

# Use an optimizer (e.g., Adam) to update the model parameters
optimizer = optim.Adam( network.parameters(), lr=learning_rate)
loss_history = torch.zeros( (epochs) )

x = 2*torch.pi* torch.rand( num_samples, 3)
x.requires_grad = True

for epoch in range(epochs):
    x = 2*torch.pi* torch.rand( num_samples, 3)
    x.requires_grad = True

    # Forward pass to compute error (in weak-form)
    p = 8 #number of points for each integral
    domain_size = 2*torch.pi/4 #physical domain size
    err = network.weak_forward( x, p, domain_size)

    # Compute the loss
    loss = criterion(err, torch.zeros_like(err))
    loss_history[epoch] = loss.detach()

    # Backward pass and optimization step
    optimizer.zero_grad()  # clear previous gradients
    loss.backward(retain_graph=True)   # compute gradients
    optimizer.step()  # update model parameters

    # Print the loss every few epochs
    if epoch % 100 == 0:
        #save_network_output( hydro_model, "torch_output_%d.mat" % (epoch) )
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, T = {network.period()}")


torch.save( network, 'trained_potential_model_{num_features}.pth' )

save_dual_network_output(network, points_per_side=64, matfile_name_prefix="dual_network", loss_history=loss_history)