import torch
from torch.utils.data import DataLoader, TensorDataset
from lib.equivariant_networks import EquivariantAutoencoder
from scipy.io import loadmat, savemat
import numpy as np

device = "cuda"
#device = "cpu"

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


lc = 2 #change to whatever you want
enc_res = [64, 32, 16,  8] #encoder resolution sequence
enc_c   = [ 2, 32, 32, 32] #output conv channels
dec_res = [ 8, 16, 32, 64] #decoder resolution sequence
dec_c   = [lc, 32, 32, 32] #output conv channels 

network = EquivariantAutoencoder( lc, enc_res, dec_res, enc_c, dec_c )
network.load_state_dict(torch.load("gpu_equivariant_autoencoder.pth", map_location=torch.device('cpu')))
network = network.to(device)

# Create lists to store predictions and latent space values
predictions  = []
latent_space = []

print(w.shape) 

#Make a minigrid
gridm  = torch.linspace(0,2*torch.pi,steps=9)
xm, ym = torch.meshgrid( gridm[0:8], gridm[0:8] )
xm = torch.unsqueeze(xm,0)
ym = torch.unsqueeze(ym,0)


# Evaluate the model and store predictions and latent space values
with torch.no_grad():
    for batch in dataloader:
        w_batch = batch[0].unsqueeze(1) #add a channel index
        input   = torch.cat( (w_batch, force), dim=1 ) #stack over channel

        input = input.to(device)

        l_batch = network.encode(input)

        #turn off all latent space
        #l_batch = 0*l_batch
        #l_batch[:,0,1,1] = 30
        #l_batch[:,0,:,:] = 100*torch.cos(xm)
        
        w_out   = network.decode(l_batch)

        #make sure these are the same shape
        w_out = torch.reshape(w_out, w_batch.shape )
        
        predictions.append(w_out.cpu().numpy())
        latent_space.append(l_batch.cpu().numpy())

# Concatenate and reshape predictions and latent space arrays
predictions  = np.concatenate(predictions, axis=0)
latent_space = np.concatenate(latent_space, axis=0)

predictions = predictions.reshape([nt,tr,n,n])
latent_space = latent_space.reshape([nt,tr,-1,8,8])
w = w.reshape([nt,tr,n,n])

print( predictions.shape )
print( latent_space.shape )
print( w.shape)

# Save predictions, w_batch, and latent space to a matfile
savemat("equivariant_predictions.mat", {"predictions": predictions, "w": w.cpu().numpy(), "latent_space": latent_space})