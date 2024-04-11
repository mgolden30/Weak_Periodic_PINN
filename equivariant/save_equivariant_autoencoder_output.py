import torch
from torch.utils.data import DataLoader, TensorDataset
from lib.equivariant_networks import EquivariantAutoencoder
from scipy.io import loadmat, savemat
import numpy as np

# Load data
data = loadmat("w_traj.mat")
w = torch.tensor(data["w"][:], dtype=torch.float32)
x = torch.tensor(data["x"], dtype=torch.float32)
y = torch.tensor(data["y"], dtype=torch.float32)

batch_size = 8

#w is saved as [n,n,nt,tr] where n is grid resolution, nt is timepoints, tr is number of trials
#For training, combine the last two dimensions, but do not forget them!
nt = w.shape[2]
tr = w.shape[3]
n  = w.shape[0]
w  = torch.reshape( w, [n,n,-1] )

# Permute dimensions for training [n,n,b] -> [b,n,n]
w = w.permute(2, 0, 1)

# Define dataset and dataloader
dataset = TensorDataset(w)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Set shuffle to False for prediction

#For the equivariant autoencoder, I am also going to pass the forcing 4 cos (4y)
force = 4*torch.cos(4*y)
force = force.unsqueeze(0) #add batch index
force = force.unsqueeze(1) #add channel index
force = force.repeat((batch_size,1,1,1)) #repeat over batch dimension

# Load trained models
network = EquivariantAutoencoder()
network.load_state_dict(torch.load("equivariant_autoencoder.pth"))

# Create lists to store predictions and latent space values
predictions  = []
latent_space = []

print(w.shape) 
# Evaluate the model and store predictions and latent space values
with torch.no_grad():
    for batch in dataloader:
        w_batch = batch[0].unsqueeze(1) #add a channel index
        input   = torch.cat( (w_batch, force), dim=1 ) #stack over channel

        l_batch = network.encode(input)   #encode the state

        w_out   = network.decode(l_batch) #decode

        #make sure these are the same shape
        w_out = torch.reshape(w_out, w_batch.shape )
        
        predictions.append(w_out.cpu().numpy())
        latent_space.append(l_batch.cpu().numpy())

# Concatenate and reshape predictions and latent space arrays
predictions  = np.concatenate(predictions, axis=0)
latent_space = np.concatenate(latent_space, axis=0)

predictions = predictions.reshape([nt,tr,n,n])
latent_space = latent_space.reshape([nt,tr,8,8,-1])
w = w.reshape([nt,tr,n,n])

print( predictions.shape )
print( latent_space.shape )
print( w.shape)

# Save predictions, w_batch, and latent space to a matfile
savemat("equivariant_predictions.mat", {"predictions": predictions, "w": w.cpu().numpy(), "latent_space": latent_space})