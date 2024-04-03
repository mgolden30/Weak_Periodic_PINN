import torch
from torch.utils.data import DataLoader, TensorDataset
from lib.autoencoder import Encoder, Decoder
from scipy.io import loadmat, savemat
import numpy as np


# Load data
data = loadmat("w_traj.mat")
w = torch.tensor(data["psi"][:, :, 1000:2000], dtype=torch.float32)
x = torch.tensor(data["x"], dtype=torch.float32)
y = torch.tensor(data["y"], dtype=torch.float32)

# Permute dimensions for training
w = w.permute(2, 0, 1)
x = torch.reshape( x, [-1, 1] )
y = torch.reshape( y, [-1, 1] )
xs = torch.cat((x, y), dim=1)

# Define dataset and dataloader
dataset = TensorDataset(w)
batch_size = 16  # Adjust batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # Set shuffle to False for prediction

# Define encoder and decoder
latent_dim = 64
enc = Encoder(channels=3, latent=latent_dim)
dec = Decoder(latent_dim=latent_dim)

# Load trained models
enc.load_state_dict(torch.load("encoder.pth"))
dec.load_state_dict(torch.load("decoder.pth"))

# Create lists to store predictions and latent space values
predictions = []
latent_space = []

# Evaluate the model and store predictions and latent space values
with torch.no_grad():
    for batch in dataloader:
        w_batch = batch[0]
        l_batch = enc(w_batch)
        w_out = dec(l_batch, xs)
        predictions.append(w_out.cpu().numpy())
        latent_space.append(l_batch.cpu().numpy())

# Concatenate and reshape predictions and latent space arrays
predictions = np.concatenate(predictions, axis=0)
latent_space = np.concatenate(latent_space, axis=0)

# Save predictions, w_batch, and latent space to a matfile
savemat("predictions.mat", {"predictions": predictions, "w_batch": w.permute(1, 2, 0).cpu().numpy(), "latent_space": latent_space})