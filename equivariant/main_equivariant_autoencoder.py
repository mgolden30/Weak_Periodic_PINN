import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lib.equivariant_networks import EquivariantAutoencoder
from scipy.io import savemat, loadmat
import numpy as np

#set device as "cuda" or "cpu"
device = "cuda"

#Parameters
batch_size =  256  #train on this many snapshots at a time
num_epochs =  256
lr = 1e-6

# Set PyTorch seed for reproducibility
seed_value = 123
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Load data
data = loadmat("w_traj.mat")
w = torch.tensor(data["w"], dtype=torch.float32)
x = torch.tensor(data["x"], dtype=torch.float32)
y = torch.tensor(data["y"], dtype=torch.float32)

#w is saved as [n,n,nt,tr] where n is grid resolution, nt is timepoints, tr is number of trials
#For our purposes, we can combint nt and tr
n = w.shape[0]
w = torch.reshape( w, [n,n,-1] )

# Permute dimensions for training [n,n,b] -> [b,n,n]
w = w.permute(2, 0, 1)

#For the equivariant autoencoder, I am also going to pass the forcing 4 cos (4y)
force = 4*torch.cos(4*y)
force = force.unsqueeze(0) #add batch index
force = force.unsqueeze(1) #add channel index
force = force.repeat((batch_size,1,1,1)) #repeat over batch dimension

# Define dataset and dataloader
dataset = TensorDataset(w)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

##############################
# Define autoencoder
##############################
lc = 2 #change to whatever you want
ch = 16
enc_res = [64, 32, 16,  8] #encoder resolution sequence
enc_c   = [ 2, ch, ch, ch] #output conv channels
dec_res = [ 8, 16, 32, 64] #decoder resolution sequence
dec_c   = [lc, ch, ch, ch] #output conv channels 

dropout = 0 #to avoid overfitting
network = EquivariantAutoencoder( lc, enc_res, dec_res, enc_c, dec_c )
network.load_state_dict(torch.load("gpu_equivariant_autoencoder.pth"))


#Send everything to the GPU
network = network.to(device)

# Define loss function (e.g., mean squared error)
criterion = nn.MSELoss()

# Define optimizer
optimizer = torch.optim.Adam( network.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        w_batch = batch[0].unsqueeze(1) #add a channel index

        input   = torch.cat( (w_batch, force), dim=1 ) #stack over channel
        input = input.to(device)

        optimizer.zero_grad()
        l_batch = network.encode(input,   dropout=dropout) #encode the state
        w_out   = network.decode(l_batch, dropout=dropout) #decode

        #make sure these are the same shape
        w_out = torch.reshape(w_out, w_batch.shape )

        #Compute the mismatch in vorticity
        w_diff = w_out - w_batch.to(device)

        #Also compute its approximate derivatives
        #dx = 2*torch.pi/64
        #wx_diff = (torch.roll(w_diff,shifts=1,dims=[2]) - w_diff)/dx
        #wy_diff = (torch.roll(w_diff,shifts=1,dims=[3]) - w_diff)/dx

        #err = torch.cat( (w_diff, wx_diff, wy_diff), dim=1 )
        err = w_diff
        loss = criterion( err, 0*err )
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save trained models if needed
torch.save(network.state_dict(), "gpu_equivariant_autoencoder.pth")
