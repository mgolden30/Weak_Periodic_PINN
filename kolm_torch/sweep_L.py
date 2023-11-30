'''
The purpose of this script is to train several networks with varying hidden layer size L
'''

import numpy as np
import os
import torch

import torch.optim as optim
from torch import nn
from lib.model import StreamfunctionNetwork, HydroNetwork, WeakPINN, PickDomains

from lib.utils import save_network_output, generate_samples_NN, reset_torch_seed, generate_uniform_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#define the layer sizes we want to check
Ls = [32]#, 64, 128, 256, 512]

output_folder = "network_output/L_sweep/"

epochs = 1024*8 + 1 #add the +1 so the final state is output

L_pick = 32

learning_rate = 0.025
learning_rate_pick = 0.05

nu = 1.0/40 #fluid viscosity
p  = 6      #points in each direction for Gaussian quadrature
n  = 8**3   #number of subdomain to train on each step

xs_uniform = generate_uniform_grid(5)

for L in Ls:
    print("L = %d" % (L) )

    #reset the torch seed every iteration for reproducibility
    reset_torch_seed()

    #build your networks (with L dependence)
    stream_model = StreamfunctionNetwork(L).to(device)
    hydro_model  = HydroNetwork( stream_model ).to(device)
    pinn         = WeakPINN( hydro_model, nu, p ).to(device)

    pick = PickDomains(L_pick).to(device)

    # Define the Mean Squared Error (MSE) loss
    criterion = nn.L1Loss()

    # Use an optimizer (e.g., Adam) to update the model parameters
    optimizer      = optim.Adam(hydro_model.parameters(), lr=learning_rate )
    optimizer_anti = optim.Adam(       pick.parameters(), lr=learning_rate_pick, maximize=True)
    
    loss_history = torch.zeros( (epochs) )

    for epoch in range(epochs):
        #generate new training data
        xs, xs_NN = generate_samples_NN(n, pick)

        # Forward pass (with output of PickDomains!)
        err = pinn.forward(xs_NN, xs_uniform)

        # Compute the MSE loss
        loss = criterion(err, torch.zeros_like(err))  # assuming you want to minimize pinn.forward(xs) to zero
        loss_history[epoch] = loss.detach()

        # clear previous gradients
        optimizer.zero_grad()
        optimizer_anti.zero_grad()

        # compute gradients
        loss.backward(retain_graph=True)   
        
        # update model parameters
        optimizer.step()
        optimizer_anti.step()

        #save state every so often
        if epoch % 1024 == 0:
            matlb_file = output_folder + "torch_output_L_%d_epoch_%d.mat" % (L, epoch)
            model_file = output_folder + "hydromodelNN_L_%d_epoch_%d.pth" % (L, epoch)

            save_network_output( hydro_model, matlb_file, model_file, loss_history, xs, xs_NN )
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

            #reset optimizers
            learning_rate = learning_rate/5
            optimizer      = optim.Adam(hydro_model.parameters(), lr=learning_rate )
            optimizer_anti = optim.Adam(       pick.parameters(), lr=learning_rate_pick, maximize=True)
