'''
The purpose of this script is to train several networks with varying hidden layer size L
'''

import numpy as np
import os
import torch

import torch.optim as optim
from torch import nn
from lib.model import StreamfunctionNetwork, HydroNetwork, WeakPINN, PickDomains

from lib.utils import save_network_output, generate_samples_NN, reset_torch_seed, generate_uniform_grid, generate_uniform_grid2

from scipy.io import loadmat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#define the layer sizes we want to check
L = 256#, 64, 128, 256, 512]

output_folder = "network_output/pretrain_test/"

epochs = 1024*4 + 1

L_pick = 32


nu = 1.0/40 #fluid viscosity
p  = 5      #points in each direction for Gaussian quadrature
n  = 8**3   #number of subdomain to train on each step

xs_uniform = generate_uniform_grid(5)


if __name__ == "__main__":
    seed = 0

    learning_rate = 0.025
    learning_rate_pick = 0.05

    print("seed = %d" % (seed) )

    #reset the torch seed every iteration for reproducibility
    reset_torch_seed(seed_value=seed)

    #build your networks (with L dependence)
    stream_model = StreamfunctionNetwork(L).to(device)
    hydro_model  = HydroNetwork( stream_model ).to(device)
    pinn         = WeakPINN( hydro_model, nu, p ).to(device)

    pick = PickDomains(L_pick).to(device)

    # Define the Mean Squared Error (MSE) loss
    criterion = nn.L1Loss()

    # pretrain the network from turbulence
    mat_contents = loadmat("kolm.mat")

    turb_w = mat_contents["w"]
    dt = mat_contents["dt"]
    every = mat_contents["every"]
    
    dt = dt*every

    t0 = 128
    start= 800
    stop = 100

  #  target = turb_w[ 0::4, 0::4, (t0+start):(t0+start+stop):4 ]
    target = turb_w[ 0::4, 0::4, t0+start ]

    a = torch.tensor(0) #zero shift
    T = torch.tensor( (stop-start)*dt )
    target = torch.tensor(target)
    target = target[..., np.newaxis]
    target = target.float()
    target = target.to(device)
    #target = torch.cat( (target, T*torch.ones_like(target), a*torch.ones_like(target)), axis=3 ) 
    
    xs_pretrain = generate_uniform_grid2( [16,16,1] ).to(device)

    # Define the Mean Squared Error (MSE) loss for pretraining
    pretrain_criterion = nn.MSELoss()

    # Set the number of pretraining epochs
    pretrain_epochs = 10000

    # Set the pretraining learning rate
    pretrain_learning_rate = 0.01

    # Use a separate optimizer for pretraining
    pretrain_optimizer = optim.SGD( stream_model.parameters(), lr=pretrain_learning_rate)

    for pretrain_epoch in range(pretrain_epochs):
        # Forward pass
        pretrain_output = pinn.forward(xs_pretrain, xs_uniform)

        # Compute the MSE loss for pretraining
        pretrain_loss = pretrain_criterion(pretrain_output, target)

        # Clear previous gradients
        pretrain_optimizer.zero_grad()

        # Compute gradients
        pretrain_loss.backward(retain_graph=True)

        # Update model parameters
        pretrain_optimizer.step()

        # Print or log the pretraining loss
        print(f"Pretraining Epoch {pretrain_epoch}/{pretrain_epochs}, Loss: {pretrain_loss.item()}")

    matlb_file = output_folder + "torch_output_L_%d_seed_%d.mat" % (L, seed)
    model_file = output_folder + "hydromodelNN_L_%d_seed_%d.pth" % (L, seed)
    loss_history = torch.tensor(0)

    save_network_output( hydro_model, matlb_file, model_file, loss_history, xs_pretrain, xs_pretrain )

    print("Pretraining finished.")
    exit()





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
            #reset optimizers
            learning_rate = learning_rate/5
            optimizer      = optim.Adam(hydro_model.parameters(), lr=learning_rate )
            optimizer_anti = optim.Adam(       pick.parameters(), lr=learning_rate_pick, maximize=True)

        #save state every so often
        if epoch  == 4*1024:
            matlb_file = output_folder + "torch_output_L_%d_epoch_%d_seed_%d.mat" % (L, epoch, seed)
            model_file = output_folder + "hydromodelNN_L_%d_epoch_%d_seed_%d.pth" % (L, epoch, seed)

            save_network_output( hydro_model, matlb_file, model_file, loss_history, xs, xs_NN )
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")