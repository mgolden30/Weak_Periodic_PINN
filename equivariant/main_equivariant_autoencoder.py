import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import savemat, loadmat
import numpy as np

#Custom stuff
from lib.EquivariantAutoencoder import EquivariantAutoencoder
from lib.dns import time_deriv
import lib.utils as ut

#set device as "cuda" or "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Parameters
batch_size =  256
num_epochs =  64
lr = 1e-3
dropout = 0 #to avoid overfitting

# Define loss function (e.g., mean squared error)
criterion = nn.L1Loss()

# Set PyTorch seed for reproducibility
seed = 123
ut.reset_torch_seed( seed_value=seed )

#Load training data
x, y, w = ut.load_vorticity_data()


#split data into training and testing
b = w.shape[0]
train_size = round(b*0.8) #Do 80-20 split
train_w = w[:train_size,:,:]
test_w  = w[train_size:,:,:]
print(f"Splitting {b} training images into {train_w.shape[0]} training and {test_w.shape[0]} testing.\n")


# Define dataset and dataloader
dataset    = TensorDataset(train_w)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset2    = TensorDataset(test_w)
dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

ut.enstrophy_statistics( w )





##############################
# Define autoencoder
##############################
lc = 5 #Number of latent images
ch = 4
enc_res = [ 64, 32, 16,  8,  4 ] # encoder resolution sequence
enc_c   = [  2, ch, ch, ch, ch ] # output conv channels
dec_res = [  4,  8, 16, 32, 64 ] # decoder resolution sequence
dec_c   = [ lc, ch, ch, ch, ch ] # output conv channels 

network = EquivariantAutoencoder( lc, enc_res, dec_res, enc_c, dec_c )
#network.load_state_dict(torch.load(f"models/model_16.pth"))
network = network.to(device)



# Define optimizer
optimizer = torch.optim.Adam( network.parameters(), lr=lr)


train_loss = np.zeros((num_epochs))
test_loss  = np.zeros((num_epochs))

#For the equivariant autoencoder, I am also going to pass the forcing 4 cos (4y)
force = 4*torch.cos(4*y)
force = force.unsqueeze(0) #add batch index
force = force.unsqueeze(1) #add channel index


def evaluate_loss( batch, force ):
    '''
    Write a wrapper for batched evaluation
    '''

    #Reshape the forcing to match batch dimension
    w_batch = batch[0].unsqueeze(1) #add a channel index
    force_b = force.repeat((w_batch.shape[0],1,1,1)) #repeat over batch dimension       
    
    #Compute the state-space velocity
    nu = 1.0/40.0
    v  = time_deriv( w_batch, nu, force )

    #stack over channel
    input   = torch.cat( (w_batch,   force_b), dim=1 ) 
    v_input = torch.cat( (v,       0*force_b), dim=1 ) 

    #Move to GPU if you want
    input   = input.to(device)
    v_input = v_input.to(device)

    latent, latent_v = network.encode(input,  v=v_input,  dropout=dropout) #encode the state
    w_out,  v_out    = network.decode(latent, v=latent_v, dropout=dropout) #decode

    #make sure these are the same shape
    w_out = torch.reshape(w_out, w_batch.shape )
    v_out = torch.reshape(v_out, v.shape )

    #Compute the mismatch in vorticity
    w_diff = w_out - w_batch.to(device)

    #Compute the mismatch in state space velocity
    v_diff = v_out - v.to(device)

    #Stack these to get a total error
    err = torch.cat( (w_diff, v_diff), dim=0 )

    #shoot for err = 0
    loss = criterion( err, 0*err )
    return loss





# Training loop
for epoch in range(num_epochs):
    if epoch % 8  == 0:
        torch.save(network.state_dict(), f"models/model_{epoch}.pth")

    #Loop over training data
    for batch in dataloader:
        optimizer.zero_grad()
        loss = evaluate_loss( batch, force )
        loss.backward()
        optimizer.step()
        train_loss[epoch] = train_loss[epoch] + loss.item() * batch[0].shape[0]
    train_loss[epoch] = train_loss[epoch]/train_w.shape[0]

    #Loop over testing data
    for batch in dataloader2:
        loss = evaluate_loss( batch, force )
        test_loss[epoch] = test_loss[epoch] + loss.item() * batch[0].shape[0]
    test_loss[epoch] = test_loss[epoch]/test_w.shape[0]


    print(f"Epoch {epoch+1}, training loss: {train_loss[epoch]}, test loss: {test_loss[epoch]}")

# Save trained models if needed
torch.save(network.state_dict(), "gpu_equivariant_autoencoder.pth")

#Save losses
loss_dict = { "train_loss": train_loss, "test_loss": test_loss }
savemat( "loss.mat", loss_dict)
