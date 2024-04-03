import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lib.autoencoder import Encoder, Decoder
from scipy.io import savemat, loadmat
import numpy as np

# Load data
data = loadmat("w_traj.mat")
w = torch.tensor(data["psi"][:, :, 1000::30], dtype=torch.float32)
x = torch.tensor(data["x"], dtype=torch.float32)
y = torch.tensor(data["y"], dtype=torch.float32)

# Permute dimensions for training
w = w.permute(2, 0, 1)
x = torch.reshape( x, [-1, 1] )
y = torch.reshape( y, [-1, 1] )
xs = torch.cat((x, y), dim=1)

# Define dataset and dataloader
dataset = TensorDataset(w)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define encoder and decoder
latent_dim = 64
enc = Encoder(channels=3, latent=latent_dim)
dec = Decoder(latent_dim=latent_dim)

# Define loss function (e.g., mean squared error)
criterion = nn.MSELoss()

# Define optimizer
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.001)

# Training loop
num_epochs = 64
for epoch in range(num_epochs):
    for batch in dataloader:
        w_batch = batch[0]

        optimizer.zero_grad()
        l_batch = enc(w_batch)
        w_out = dec(l_batch, xs)

        #Make them the same size
        w_out = torch.reshape(w_out, w_batch.shape )

        loss = criterion(w_out, w_batch )
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")



# After training, it is probable that not all available latent degrees of freedom are used.
# For the purposes of evaluation, we should be certain to never let the static latent dimensions deviate from their mean value, since such values were never seen.
# Evaluate latent space on all training data
latent_space = []
with torch.no_grad():
    for w_batch in dataloader:
        l_batch = enc(w_batch[0])
        latent_space.append(l_batch.numpy())  # Convert to numpy array for easier processing

# Concatenate latent space across batches
latent_space = np.concatenate(latent_space, axis=0)

# Compute mean and standard deviation of each latent dimension
latent_mean = np.mean(latent_space, axis=0)
latent_std  = np.std(latent_space, axis=0)

# Create mask based on standard deviation
static_mask = latent_std < 1e-2  # Adjust threshold as needed
static_vals = latent_mean[static_mask]

print(static_mask)

# Save trained models if needed
torch.save(enc.state_dict(), "encoder.pth")
torch.save(dec.state_dict(), "decoder.pth")

#save the mask and mean values for evaluation
my_dict = {"static_mask": static_mask, "static_mean": static_vals}
savemat("latent_mask.mat", my_dict)
