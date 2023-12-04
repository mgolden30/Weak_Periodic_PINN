'''
Here are a bunch of functions I didn't know where to put
'''

import torch
import numpy as np
from scipy.io import savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_samples_NN(n, pick):
    #n is the TOTAL number of subdomains we randomly sample
    xs = 2 * torch.pi * torch.rand( (n,3), requires_grad=True ) #need to diff with respect to these for flow
    #xs = generate_uniform_grid( int(np.ceil(n**(1.0/3))) )
    xs = xs.to(device)

    #xs = generate_uniform_grid(12)

    xs_NN = pick.forward(xs)
    
    return xs, xs_NN

def generate_uniform_grid(n):
    #To penalize no time derivative, I will compute \partial_t \omega on a uniform grid
    #n is now points per side instead of total number.
    #This will not be in weak form.
    x1 = torch.linspace( 0, n-1, n, requires_grad=True ) / n * 2 * np.pi
    [x,y,t] = torch.meshgrid( x1, x1, x1 )
    x = torch.reshape( x, [-1,1])
    y = torch.reshape( y, [-1,1])
    t = torch.reshape( t, [-1,1])
    xs = torch.cat((x,y,t), axis=1).to(device)
    return xs

def generate_uniform_grid2(ns):
    #To penalize no time derivative, I will compute \partial_t \omega on a uniform grid
    #n is now points per side instead of total number.
    #This will not be in weak form.
    x1 = torch.linspace( 0, ns[0]-1, ns[0], requires_grad=True ) / ns[0] * 2 * np.pi
    x2 = torch.linspace( 0, ns[1]-1, ns[1], requires_grad=True ) / ns[1] * 2 * np.pi
    x3 = torch.linspace( 0, ns[2]-1, ns[2], requires_grad=True ) / ns[2] * 2 * np.pi

    [x,y,t] = torch.meshgrid( x1, x2, x3 )
    x = torch.reshape( x, [-1,1])
    y = torch.reshape( y, [-1,1])
    t = torch.reshape( t, [-1,1])
    xs = torch.cat((x,y,t), axis=1).to(device)
    return xs


def save_network_output( hydro_model, out_name, model_name, loss_history, xs, xs_NN ):
    # After training, you can use the trained model for predictions
    ns=(128,128,4)

    x_grid = torch.linspace( 0, 2*torch.pi, ns[0], requires_grad=True )
    y_grid = torch.linspace( 0, 2*torch.pi, ns[1], requires_grad=True )
    t_grid = torch.linspace( 0, 2*torch.pi, ns[2], requires_grad=True )
    [x,y,t] = torch.meshgrid( (x_grid, y_grid, t_grid) )

    x = torch.reshape( x, [-1,1] )
    y = torch.reshape( y, [-1,1] )
    t = torch.reshape( t, [-1,1] )

    xs_fine      = torch.cat( (x,y,t), dim=1 ).to(device)
    f_final = hydro_model.forward(xs_fine)

    f_final = f_final.cpu().detach().numpy()
    x_grid  = x_grid.cpu().detach().numpy()
    y_grid  = y_grid.cpu().detach().numpy()
    t_grid  = t_grid.cpu().detach().numpy()

    f_final = np.reshape( f_final, [ns[0], ns[1], ns[2], -1] )

    loss_history = loss_history.cpu().detach().numpy()
    xs    = xs.cpu().detach().numpy()
    xs_NN = xs_NN.cpu().detach().numpy()
    

    out_dict =  {"f": f_final, "x_grid": x_grid, "y_grid": y_grid, "t_grid": t_grid, "loss_history": loss_history, \
                 "xs": xs, "xs_NN": xs_NN }
    
    savemat(out_name, out_dict)
    torch.save( hydro_model, model_name )

def reset_torch_seed( seed_value=142):
    # Set PyTorch seed for reproducibility
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)