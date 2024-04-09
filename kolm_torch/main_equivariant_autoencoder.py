import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lib.equivariant_layer import EquivariantAutoencoder
from lib.model import HydroNetwork
from scipy.io import savemat, loadmat
import numpy as np

#Parameters
batch_size = 64  #train on this many snapshots at a time
num_epochs = 512
lr = 0.001


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
force = force.unsqueeze(3) #add channel index
force = force.repeat((batch_size,1,1,1)) #repeat over batch dimension

# Define dataset and dataloader
dataset = TensorDataset(w)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define encoder and decoder
network = EquivariantAutoencoder()
#network.load_state_dict(torch.load("equivariant_autoencoder.pth"))

# Define loss function (e.g., mean squared error)
criterion = nn.MSELoss()

# Define optimizer
optimizer = torch.optim.Adam( network.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        w_batch = batch[0].unsqueeze(3) #add a channel index
        input   = torch.cat( (w_batch, force), dim=3 ) #stack over channel

        optimizer.zero_grad()
        l_batch = network.encode(input)   #encode the state

        #drop = torch.nn.Dropout(p=0.2)
        #l_batch = drop(l_batch)

        w_out   = network.decode(l_batch) #decode

        #make sure these are the same shape
        w_out = torch.reshape(w_out, w_batch.shape )
        
        loss = criterion(w_out, w_batch )
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save trained models if needed
torch.save(network.state_dict(), "equivariant_autoencoder.pth")