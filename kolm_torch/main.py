import numpy as np
import os
import torch

import torch.optim as optim
from torch import nn


from lib.model import StreamfunctionNetwork_RBF, HydroNetwork, WeakPINN
from lib.utils import generate_uniform_grid

from scipy.io import savemat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Set PyTorch seed for reproducibility
seed_value = 420
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

epochs = 4*1024

nu = 1.0/40 #fluid viscosity
p  = 4      #points in each direction for Gaussian quadrature
#L  = 16    #number of nodes in hidden layers
n  = 4**3   #number of subdomain to train on


#RBF parameters
length = 0.5 #characteristic length of rbfs in sin/cos space No truncation needed if length >= 2
power  = 2 #power of envelope polynomial used as rbf
n_cent = 4 #centers per dimnension for rbf. Total number will be n_cent^3



def generate_samples(n):
    #n is the TOTAL number of subdomains we randomly sample
    xs = 2 * torch.pi * torch.rand( (n,3), requires_grad=True ) #need to diff with respect to these for flow
    return xs 

def generate_uniform_grid(n):
    #To penalize no time derivative, I will compute \partial_t \omega on a uniform grid
    #n is now points per side instead of total number.
    #This will not be in weak form.
    x1 = torch.linspace( 0, n-1, n, requires_grad=True ) / n * 2 * np.pi
    [x,y,t] = torch.meshgrid( x1, x1, x1 )
    x = torch.reshape( x, [-1,1])
    y = torch.reshape( y, [-1,1])
    t = torch.reshape( t, [-1,1])
    xs = torch.cat((x,y,t), axis=1)
    return xs

def save_network_output( hydro_model, out_name ):
    # After training, you can use the trained model for predictions
    ns=(64,64,32)

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
    savemat(out_name, out_dict)


xs = generate_samples(n).to(device)
#I'm hoping a sparse grid is fine to compute penalty
xs_uniform = generate_uniform_grid(3).to(device) 

stream_model = StreamfunctionNetwork_RBF( n_cent, power, length )
hydro_model  = HydroNetwork( stream_model )

#hydro_model = torch.load('trained_model.pth')

pinn         = WeakPINN( hydro_model, nu, p )


psi = stream_model.forward(xs)
f   = hydro_model.forward(xs)
err = pinn.forward(xs, xs_uniform)

# Define the Mean Squared Error (MSE) loss
criterion = nn.MSELoss()

# Use an optimizer (e.g., Adam) to update the model parameters
optimizer = optim.Adam(hydro_model.parameters(), lr=0.001)

loss_history = torch.zeros( (epochs) )


for epoch in range(epochs):
    #generate new training data
    xs = generate_samples(n)

    # Forward pass
    err = pinn.forward(xs, xs_uniform)

    # Compute the MSE loss
    loss = criterion(err, torch.zeros_like(err))  # assuming you want to minimize pinn.forward(xs) to zero

    loss_history[epoch] = loss.detach()

    # Backward pass and optimization step
    optimizer.zero_grad()  # clear previous gradients
    loss.backward(retain_graph=True)   # compute gradients
    optimizer.step()  # update model parameters

    # Print the loss every few epochs
    if epoch % 100 == 0:
        save_network_output( hydro_model, "torch_output_%d.mat" % (epoch) )
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")


torch.save( hydro_model, 'trained_model.pth' )