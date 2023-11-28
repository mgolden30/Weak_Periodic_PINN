import numpy as np
import os
import torch

import torch.optim as optim
from torch import nn
from lib.model import StreamfunctionNetwork, HydroNetwork, WeakPINN
from scipy.io import savemat

epochs = 128

nu = 1.0/40 #fluid viscosity
p  = 5      #points for Gaussian quadrature

n  = 8**3 #number of points to train on
xs = 2 * torch.pi * torch.rand( (n,3), requires_grad=True ) #need to diff with respect to these for flow

# Set PyTorch seed for reproducibility
seed_value = 420
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

stream_model = StreamfunctionNetwork()
hydro_model  = HydroNetwork( stream_model )

hydro_model = torch.load('trained_model.pth')
pinn        = WeakPINN( hydro_model, nu, p )


# Define the Mean Squared Error (MSE) loss
criterion = nn.MSELoss()

# Use an optimizer (e.g., Adam) to update the model parameters
#optimizer = optim.Adam(hydro_model.parameters(), lr=0.005)
optimizer = optim.LBFGS(hydro_model.parameters(), lr=0.005)

loss_history = torch.zeros( (epochs) )

def closure():
    optimizer.zero_grad()  # Clear gradients
    err = pinn.forward(xs)
    # Compute the MSE loss
    loss = criterion(err, torch.zeros_like(err))
    #loss.backward()  # Backward pass
    loss.backward(retain_graph=True)   # compute gradients
    return loss

for epoch in range(epochs):
    #generate new training data
    xs = 2 * torch.pi * torch.rand( (n,3), requires_grad=True ) #need to diff with respect to these for flow

    # Forward pass
    err = pinn.forward(xs)
    # Compute the MSE loss
    loss = criterion(err, torch.zeros_like(err))  # assuming you want to minimize pinn.forward(xs) to zero
    loss_history[epoch] = loss.detach()

    # Backward pass and optimization step
    #optimizer.zero_grad()  # clear previous gradients
    #loss.backward(retain_graph=True)   # compute gradients
    optimizer.step(closure)  # update model parameters

    # Print the loss every few epochs
    #if epoch % 100 == 0:
    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")


torch.save( hydro_model, 'trained_model2.pth' )


# After training, you can use the trained model for predictions
ns=(64,64,16)

x_grid = torch.linspace( 0, 2*torch.pi, ns[0], requires_grad=True )
y_grid = torch.linspace( 0, 2*torch.pi, ns[1], requires_grad=True )
t_grid = torch.linspace( 0, 2*torch.pi, ns[2], requires_grad=True )
[x,y,t] = torch.meshgrid( (x_grid, y_grid, t_grid) )

x = torch.reshape( x, [-1,1] )
y = torch.reshape( y, [-1,1] )
t = torch.reshape( t, [-1,1] )

xs      = torch.cat( (x,y,t), dim=1 )
f_final = hydro_model.forward(xs)

f_final = f_final.detach().numpy()
x_grid  = x_grid.detach().numpy()
y_grid  = y_grid.detach().numpy()
t_grid  = t_grid.detach().numpy()

f_final = np.reshape( f_final, [ns[0], ns[1], ns[2], -1] )

out_dict =  {"f": f_final, "x_grid": x_grid, "y_grid": y_grid, "t_grid": t_grid, "loss_history": loss_history }

savemat("torch_output.mat", out_dict)