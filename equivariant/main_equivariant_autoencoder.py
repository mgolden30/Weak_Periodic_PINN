import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lib.equivariant_networks import EquivariantAutoencoder
from scipy.io import savemat, loadmat
import numpy as np

torch.autograd.set_detect_anomaly(True)

#Parameters
batch_size = 128  #train on this many snapshots at a time
num_epochs =  64
lr = 1e-3

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
        w_batch = batch[0].unsqueeze(1) #add a channel index
        input   = torch.cat( (w_batch, force), dim=1 ) #stack over channel

        optimizer.zero_grad()
        l_batch = network.encode(input)   #encode the state

        w_out   = network.decode(l_batch) #decode

        #make sure these are the same shape
        w_out = torch.reshape(w_out, w_batch.shape )
        
        loss = criterion(w_out, w_batch )
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save trained models if needed
torch.save(network.state_dict(), "equivariant_autoencoder.pth")