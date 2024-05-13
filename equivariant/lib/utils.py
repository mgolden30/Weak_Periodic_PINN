import torch
import numpy as np
from scipy.io import loadmat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vorticity_data():
    #General purpose function for loading vorticity training data
    #Note that we will merge the batch and time axis before returning
    # w is then of size [b,n,n] where b is a superbatch index 

    # Load data
    data = loadmat("w_traj.mat")
    w = torch.tensor(data["w"], dtype=torch.float32)
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)

    #w is saved in shape [b,nt,n,n]
    n = w.shape[-1]
    w = torch.reshape( w, [-1,n,n] )

    return x,y,w


def reset_torch_seed( seed_value=142 ):
    # Set PyTorch seed for reproducibility
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def enstrophy_statistics( w ):
    # I want to force the network to learn extreme events in addition to slowly evolving state.
    # To accomplish this, I will create a histogram of enstrophy (mean vorticity squared)

    #compute the mean of the entire dataset
    enstrophy = torch.mean( w*w, dim=[1,2], keepdim=True )

    #Compute a histogram manually
    max_enst = torch.max( enstrophy )
    min_enst = torch.min( enstrophy )
    print( f"Enstrophy ranges from {min_enst} to {max_enst}" )